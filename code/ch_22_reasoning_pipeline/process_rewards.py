"""
Process Reward Models for step-by-step evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json


@dataclass
class PRMConfig:
    """Configuration for Process Reward Model."""
    base_model_name: str = "microsoft/deberta-v3-base"
    
    # Model architecture
    hidden_size: int = 768
    num_labels: int = 1  # Regression for step quality
    dropout_rate: float = 0.1
    
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Step evaluation settings
    min_step_length: int = 10  # Minimum words in a step
    max_step_length: int = 200  # Maximum words in a step
    context_window: int = 3  # Previous steps to consider
    
    # Data generation
    num_perturbations: int = 5  # Negative examples per positive
    perturbation_types: List[str] = None
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.perturbation_types is None:
            self.perturbation_types = [
                'arithmetic_error',
                'logic_error', 
                'irrelevant_step',
                'incomplete_step',
                'circular_reasoning'
            ]


class ProcessRewardModel(nn.Module):
    """Neural Process Reward Model for evaluating reasoning steps."""
    
    def __init__(self, config: PRMConfig):
        super().__init__()
        self.config = config
        
        # Load base transformer
        self.encoder = AutoModel.from_pretrained(config.base_model_name)
        
        # Step quality prediction head
        self.quality_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.num_labels)
        )
        
        # Step type classifier (optional)
        self.step_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 5)  # 5 step types
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass for step evaluation."""
        # Encode text
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Pool hidden states
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Predict step quality
        quality_score = self.quality_head(pooled_output)
        quality_score = self.sigmoid(quality_score)
        
        # Predict step type
        step_type_logits = self.step_classifier(pooled_output)
        
        return {
            'quality_score': quality_score,
            'step_type_logits': step_type_logits,
            'hidden_states': pooled_output
        }


class PRMDataGenerator:
    """Generate training data for Process Reward Model."""
    
    def __init__(self, outcome_reward_fn, config: PRMConfig):
        self.outcome_reward_fn = outcome_reward_fn
        self.config = config
        self.perturbation_functions = self._setup_perturbations()
    
    def generate_prm_dataset(self, problems_with_solutions: List[Dict]) -> List[Dict]:
        """Generate PRM training data from solved problems."""
        prm_data = []
        
        for item in tqdm(problems_with_solutions, desc="Generating PRM data"):
            problem = item['problem']
            solution = item['solution']
            
            # Parse solution into steps
            steps = self._parse_solution_steps(solution)
            if not steps:
                continue
            
            # Get outcome reward for correct solution
            correct_reward = self.outcome_reward_fn(problem, solution)
            
            # Generate training examples
            for i, step in enumerate(steps):
                # Positive example (correct step)
                context = self._get_step_context(steps, i)
                prm_data.append({
                    'problem': problem,
                    'context': context,
                    'step': step,
                    'label': correct_reward,  # High quality
                    'step_index': i,
                    'total_steps': len(steps),
                    'is_correct': True
                })
                
                # Negative examples (perturbed steps)
                for pert_type in self.config.perturbation_types[:self.config.num_perturbations]:
                    perturbed_step = self.perturbation_functions[pert_type](step, context)
                    
                    # Evaluate perturbed solution
                    perturbed_solution = self._reconstruct_solution(steps, i, perturbed_step)
                    perturbed_reward = self.outcome_reward_fn(problem, perturbed_solution)
                    
                    prm_data.append({
                        'problem': problem,
                        'context': context,
                        'step': perturbed_step,
                        'label': perturbed_reward,  # Usually lower
                        'step_index': i,
                        'total_steps': len(steps),
                        'is_correct': False,
                        'perturbation_type': pert_type
                    })
        
        return prm_data
    
    def _parse_solution_steps(self, solution: str) -> List[str]:
        """Parse solution into individual steps."""
        # Try multiple parsing strategies
        steps = []
        
        # Strategy 1: Numbered steps
        numbered_pattern = r'(?:^|\n)\s*(?:\d+[\.\)]|Step \d+:)\s*(.+?)(?=(?:\n\s*(?:\d+[\.\)]|Step \d+:))|$)'
        numbered_steps = re.findall(numbered_pattern, solution, re.DOTALL | re.MULTILINE)
        if numbered_steps:
            steps = [s.strip() for s in numbered_steps if len(s.strip()) > self.config.min_step_length]
        
        # Strategy 2: Line-based (if no numbered steps)
        if not steps:
            lines = solution.split('\n')
            current_step = ""
            for line in lines:
                line = line.strip()
                if line:
                    current_step += " " + line
                    # Check if this looks like end of step
                    if any(line.endswith(p) for p in ['.', '!', '?']) and len(current_step.split()) > 10:
                        steps.append(current_step.strip())
                        current_step = ""
            
            if current_step.strip():
                steps.append(current_step.strip())
        
        # Strategy 3: Sentence-based (fallback)
        if not steps:
            sentences = re.split(r'(?<=[.!?])\s+', solution)
            steps = [s.strip() for s in sentences if len(s.split()) > self.config.min_step_length]
        
        return steps
    
    def _get_step_context(self, steps: List[str], current_index: int) -> str:
        """Get context (previous steps) for current step."""
        start_idx = max(0, current_index - self.config.context_window)
        context_steps = steps[start_idx:current_index]
        return " ".join(context_steps)
    
    def _setup_perturbations(self) -> Dict:
        """Setup perturbation functions for generating negative examples."""
        
        def arithmetic_error(step: str, context: str) -> str:
            """Introduce arithmetic errors."""
            # Find numbers and change them
            import random
            numbers = re.findall(r'\b\d+\b', step)
            if numbers:
                num_to_change = random.choice(numbers)
                wrong_num = str(int(num_to_change) + random.choice([-1, 1, 2, -2]))
                return step.replace(num_to_change, wrong_num, 1)
            return step + " (error in calculation)"
        
        def logic_error(step: str, context: str) -> str:
            """Introduce logical errors."""
            logic_flips = {
                'therefore': 'however',
                'because': 'despite',
                'if': 'unless',
                'all': 'none',
                'must': 'might',
                'always': 'never'
            }
            
            for correct, wrong in logic_flips.items():
                if correct in step.lower():
                    return re.sub(r'\b' + correct + r'\b', wrong, step, flags=re.IGNORECASE)
            
            return step + " which contradicts the previous step"
        
        def irrelevant_step(step: str, context: str) -> str:
            """Replace with irrelevant content."""
            irrelevant_templates = [
                "We need to consider the weather conditions.",
                "This reminds me of a different problem.",
                "Let's take a break and come back to this.",
                "The color of the objects doesn't matter here.",
                "We should verify this with additional sources."
            ]
            return np.random.choice(irrelevant_templates)
        
        def incomplete_step(step: str, context: str) -> str:
            """Make step incomplete."""
            words = step.split()
            if len(words) > 5:
                cutoff = len(words) // 2
                return " ".join(words[:cutoff]) + "..."
            return step[:len(step)//2] + "..."
        
        def circular_reasoning(step: str, context: str) -> str:
            """Introduce circular reasoning."""
            if context:
                # Reference something from context incorrectly
                return f"As we established, {step.lower()} because we already know {step.lower()}"
            return f"This is true because {step.lower()}"
        
        return {
            'arithmetic_error': arithmetic_error,
            'logic_error': logic_error,
            'irrelevant_step': irrelevant_step,
            'incomplete_step': incomplete_step,
            'circular_reasoning': circular_reasoning
        }
    
    def _reconstruct_solution(self, steps: List[str], changed_idx: int, new_step: str) -> str:
        """Reconstruct solution with changed step."""
        modified_steps = steps.copy()
        modified_steps[changed_idx] = new_step
        return "\n".join(modified_steps)


class PRMTrainer:
    """Trainer for Process Reward Model."""
    
    def __init__(self, config: PRMConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.model = ProcessRewardModel(config).to(config.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.loss_fn = nn.MSELoss()
    
    def train(self, train_data: List[Dict], val_data: List[Dict] = None):
        """Train the PRM."""
        train_dataset = PRMDataset(train_data, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        print(f"Training PRM on {len(train_data)} examples")
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
                # Move to device
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                quality_scores = outputs['quality_score'].squeeze()
                
                # Compute loss
                loss = self.loss_fn(quality_scores, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Validation
            if val_data:
                val_metrics = self.evaluate(val_data)
                print(f"Validation Metrics: {val_metrics}")
    
    def evaluate(self, eval_data: List[Dict]) -> Dict:
        """Evaluate PRM performance."""
        self.model.eval()
        
        correct_scores = []
        incorrect_scores = []
        
        with torch.no_grad():
            for item in eval_data:
                # Prepare input
                text = f"Problem: {item['problem']}\nContext: {item['context']}\nStep: {item['step']}"
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding='max_length'
                ).to(self.config.device)
                
                # Get prediction
                outputs = self.model(**inputs)
                score = outputs['quality_score'].item()
                
                if item['is_correct']:
                    correct_scores.append(score)
                else:
                    incorrect_scores.append(score)
        
        # Compute metrics
        metrics = {
            'avg_correct_score': np.mean(correct_scores),
            'avg_incorrect_score': np.mean(incorrect_scores),
            'score_gap': np.mean(correct_scores) - np.mean(incorrect_scores),
            'discrimination_accuracy': self._compute_discrimination_accuracy(
                correct_scores, incorrect_scores
            )
        }
        
        return metrics
    
    def _compute_discrimination_accuracy(self, correct_scores: List[float], 
                                       incorrect_scores: List[float]) -> float:
        """Compute how well PRM discriminates correct vs incorrect steps."""
        threshold = 0.5
        
        correct_predictions = sum(1 for s in correct_scores if s > threshold)
        incorrect_predictions = sum(1 for s in incorrect_scores if s <= threshold)
        
        total_correct = correct_predictions + incorrect_predictions
        total = len(correct_scores) + len(incorrect_scores)
        
        return total_correct / total if total > 0 else 0


class PRMDataset(Dataset):
    """Dataset for PRM training."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Construct input text
        text = f"Problem: {item['problem']}\nContext: {item['context']}\nStep: {item['step']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.float)
        }


class ProcessRewardEvaluator:
    """Use PRM to evaluate reasoning chains."""
    
    def __init__(self, prm_model: ProcessRewardModel, tokenizer):
        self.prm = prm_model
        self.tokenizer = tokenizer
        self.prm.eval()
    
    def evaluate_solution(self, problem: str, solution: str) -> Dict:
        """Evaluate complete solution using PRM."""
        # Parse steps
        parser = PRMDataGenerator(None, PRMConfig())
        steps = parser._parse_solution_steps(solution)
        
        if not steps:
            return {
                'overall_score': 0.0,
                'step_scores': [],
                'weakest_step': None,
                'strongest_step': None
            }
        
        step_scores = []
        step_details = []
        
        with torch.no_grad():
            for i, step in enumerate(steps):
                # Get context
                context = " ".join(steps[max(0, i-3):i])
                
                # Prepare input
                text = f"Problem: {problem}\nContext: {context}\nStep: {step}"
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(self.prm.config.device)
                
                # Get score
                outputs = self.prm(**inputs)
                score = outputs['quality_score'].item()
                
                step_scores.append(score)
                step_details.append({
                    'step_index': i,
                    'step_text': step[:100] + "..." if len(step) > 100 else step,
                    'score': score
                })
        
        # Compute overall score (geometric mean to penalize weak steps)
        overall_score = np.exp(np.mean(np.log(np.array(step_scores) + 1e-6)))
        
        # Find weakest and strongest steps
        weakest_idx = np.argmin(step_scores)
        strongest_idx = np.argmax(step_scores)
        
        return {
            'overall_score': overall_score,
            'step_scores': step_scores,
            'step_details': step_details,
            'weakest_step': step_details[weakest_idx],
            'strongest_step': step_details[strongest_idx],
            'num_steps': len(steps)
        }
    
    def compare_solutions(self, problem: str, solutions: List[str]) -> List[Dict]:
        """Compare multiple solutions using PRM."""
        results = []
        
        for i, solution in enumerate(solutions):
            evaluation = self.evaluate_solution(problem, solution)
            evaluation['solution_index'] = i
            results.append(evaluation)
        
        # Sort by overall score
        results.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return results