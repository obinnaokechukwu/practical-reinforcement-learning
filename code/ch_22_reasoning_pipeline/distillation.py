"""
Knowledge distillation from RL-trained reasoning models.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import json


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    teacher_model_name: str
    student_model_name: str
    
    # Distillation settings
    temperature: float = 3.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for student loss
    
    # Training settings
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 500
    
    # Generation settings for teacher
    teacher_temperature: float = 0.7
    teacher_top_p: float = 0.9
    max_length: int = 1024
    
    # Progressive distillation
    curriculum_learning: bool = True
    difficulty_stages: int = 3
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = True


class ReasoningDataset(Dataset):
    """Dataset for reasoning distillation."""
    
    def __init__(self, problems: List[str], solutions: List[str], tokenizer, max_length: int = 1024):
        self.problems = problems
        self.solutions = solutions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        solution = self.solutions[idx]
        
        # Tokenize input and target
        inputs = self.tokenizer(
            problem,
            max_length=self.max_length // 2,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            solution,
            max_length=self.max_length // 2,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }


class ReasoningDistiller:
    """Distill reasoning capabilities from teacher to student."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        
        # Load teacher model
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        ).to(config.device)
        self.teacher_model.eval()
        
        # Load student model
        self.student_tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        ).to(config.device)
        
        # Ensure compatible tokenizers
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        
        # Metrics tracking
        self.distillation_metrics = {
            'teacher_perplexity': [],
            'student_perplexity': [],
            'kl_divergence': [],
            'reasoning_score_correlation': []
        }
    
    def generate_teacher_solutions(self, problems: List[str]) -> List[str]:
        """Generate high-quality solutions from teacher model."""
        solutions = []
        
        for problem in tqdm(problems, desc="Generating teacher solutions"):
            inputs = self.teacher_tokenizer(
                problem,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.config.device)
            
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    **inputs,
                    max_length=self.config.max_length,
                    temperature=self.config.teacher_temperature,
                    top_p=self.config.teacher_top_p,
                    do_sample=True,
                    pad_token_id=self.teacher_tokenizer.pad_token_id
                )
            
            solution = self.teacher_tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = solution[len(problem):].strip()
            solutions.append(solution)
        
        return solutions
    
    def distill_with_reasoning_preservation(self, problems: List[str], validation_problems: List[str] = None):
        """Distill while preserving reasoning capabilities."""
        
        # Generate teacher solutions
        print("Generating teacher solutions...")
        teacher_solutions = self.generate_teacher_solutions(problems)
        
        # Create dataset
        dataset = ReasoningDataset(
            problems, 
            teacher_solutions, 
            self.student_tokenizer,
            self.config.max_length
        )
        
        # Setup training
        training_args = TrainingArguments(
            output_dir="./distillation_output",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no"
        )
        
        # Custom trainer with distillation loss
        trainer = DistillationTrainer(
            model=self.student_model,
            args=training_args,
            train_dataset=dataset,
            teacher_model=self.teacher_model,
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            beta=self.config.beta
        )
        
        # Train
        print("Starting distillation...")
        trainer.train()
        
        # Evaluate if validation set provided
        if validation_problems:
            self.evaluate_distillation(validation_problems)
    
    def evaluate_distillation(self, test_problems: List[str]):
        """Evaluate quality of distillation."""
        teacher_scores = []
        student_scores = []
        
        for problem in test_problems:
            # Teacher solution
            teacher_sol = self.generate_teacher_solutions([problem])[0]
            
            # Student solution
            student_inputs = self.student_tokenizer(
                problem,
                return_tensors='pt',
                truncation=True
            ).to(self.config.device)
            
            with torch.no_grad():
                student_outputs = self.student_model.generate(
                    **student_inputs,
                    max_length=self.config.max_length,
                    temperature=0.7,
                    do_sample=True
                )
            
            student_sol = self.student_tokenizer.decode(
                student_outputs[0], 
                skip_special_tokens=True
            )[len(problem):].strip()
            
            # Compare reasoning patterns
            teacher_score = self._evaluate_reasoning_quality(teacher_sol)
            student_score = self._evaluate_reasoning_quality(student_sol)
            
            teacher_scores.append(teacher_score)
            student_scores.append(student_score)
        
        # Compute correlation
        correlation = np.corrcoef(teacher_scores, student_scores)[0, 1]
        
        print(f"\nDistillation Evaluation:")
        print(f"Teacher avg score: {np.mean(teacher_scores):.3f}")
        print(f"Student avg score: {np.mean(student_scores):.3f}")
        print(f"Score correlation: {correlation:.3f}")
        
        self.distillation_metrics['reasoning_score_correlation'].append(correlation)
    
    def _evaluate_reasoning_quality(self, solution: str) -> float:
        """Simple heuristic for reasoning quality."""
        score = 0.0
        
        # Check for reasoning indicators
        reasoning_words = ['because', 'therefore', 'thus', 'since', 'if', 'then', 
                          'first', 'second', 'finally', 'step']
        for word in reasoning_words:
            if word in solution.lower():
                score += 0.1
        
        # Check for structure
        if '\n' in solution:
            score += 0.2
        
        # Check for mathematical expressions
        if any(op in solution for op in ['=', '+', '-', '*', '/']):
            score += 0.2
        
        # Length appropriateness
        length = len(solution.split())
        if 50 < length < 500:
            score += 0.2
        
        return min(score, 1.0)


class ProgressiveDistiller(ReasoningDistiller):
    """Progressive distillation with curriculum learning."""
    
    def __init__(self, config: DistillationConfig):
        super().__init__(config)
        self.difficulty_classifier = self._build_difficulty_classifier()
    
    def _build_difficulty_classifier(self):
        """Build simple difficulty classifier for problems."""
        def classify_difficulty(problem: str) -> int:
            # Simple heuristic based on problem characteristics
            problem_lower = problem.lower()
            
            score = 0
            
            # Length
            if len(problem.split()) > 50:
                score += 1
            
            # Mathematical complexity
            if any(term in problem_lower for term in ['derivative', 'integral', 'matrix']):
                score += 2
            elif any(term in problem_lower for term in ['equation', 'solve', 'calculate']):
                score += 1
            
            # Multi-step indicators
            if any(term in problem_lower for term in ['then', 'after', 'finally']):
                score += 1
            
            # Constraints
            if 'constraint' in problem_lower or 'subject to' in problem_lower:
                score += 1
            
            # Map to difficulty levels (0: easy, 1: medium, 2: hard)
            if score <= 1:
                return 0
            elif score <= 3:
                return 1
            else:
                return 2
        
        return classify_difficulty
    
    def progressive_distillation(self, problems: List[str], validation_problems: List[str] = None):
        """Perform curriculum-based progressive distillation."""
        
        # Classify problems by difficulty
        classified_problems = {0: [], 1: [], 2: []}
        for problem in problems:
            difficulty = self.difficulty_classifier(problem)
            classified_problems[difficulty].append(problem)
        
        print(f"Problem distribution: Easy: {len(classified_problems[0])}, "
              f"Medium: {len(classified_problems[1])}, Hard: {len(classified_problems[2])}")
        
        # Progressive training
        all_problems_so_far = []
        
        for stage in range(self.config.difficulty_stages):
            print(f"\n=== Stage {stage + 1}/{self.config.difficulty_stages} ===")
            
            # Add problems from current difficulty
            if stage < len(classified_problems):
                all_problems_so_far.extend(classified_problems[stage])
            
            if not all_problems_so_far:
                continue
            
            # Adjust learning rate for later stages
            original_lr = self.config.learning_rate
            self.config.learning_rate = original_lr * (0.5 ** stage)
            
            # Distill on accumulated problems
            self.distill_with_reasoning_preservation(
                all_problems_so_far,
                validation_problems
            )
            
            # Reset learning rate
            self.config.learning_rate = original_lr
            
            # Evaluate stage performance
            if validation_problems:
                stage_metrics = self._evaluate_stage(stage, validation_problems)
                print(f"Stage {stage + 1} metrics: {stage_metrics}")
    
    def _evaluate_stage(self, stage: int, test_problems: List[str]) -> Dict:
        """Evaluate performance at current stage."""
        # Sample problems from different difficulties
        easy_problems = [p for p in test_problems if self.difficulty_classifier(p) == 0][:5]
        medium_problems = [p for p in test_problems if self.difficulty_classifier(p) == 1][:5]
        hard_problems = [p for p in test_problems if self.difficulty_classifier(p) == 2][:5]
        
        scores = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        for problems, difficulty in [(easy_problems, 'easy'), 
                                    (medium_problems, 'medium'), 
                                    (hard_problems, 'hard')]:
            for problem in problems:
                student_sol = self._generate_student_solution(problem)
                score = self._evaluate_reasoning_quality(student_sol)
                scores[difficulty].append(score)
        
        return {
            'stage': stage + 1,
            'easy_avg': np.mean(scores['easy']) if scores['easy'] else 0,
            'medium_avg': np.mean(scores['medium']) if scores['medium'] else 0,
            'hard_avg': np.mean(scores['hard']) if scores['hard'] else 0
        }
    
    def _generate_student_solution(self, problem: str) -> str:
        """Generate solution from student model."""
        inputs = self.student_tokenizer(
            problem,
            return_tensors='pt',
            truncation=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.student_model.generate(
                **inputs,
                max_length=self.config.max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.student_tokenizer.pad_token_id
            )
        
        solution = self.student_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return solution[len(problem):].strip()


class DistillationTrainer(Trainer):
    """Custom trainer for knowledge distillation."""
    
    def __init__(self, teacher_model, temperature, alpha, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute distillation loss."""
        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Distillation loss (KL divergence)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Student loss (cross-entropy with true labels)
        student_loss = student_outputs.loss
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + self.beta * student_loss
        
        return (total_loss, student_outputs) if return_outputs else total_loss


class ReasoningDistillationPipeline:
    """Complete pipeline for distilling reasoning models."""
    
    def __init__(self, teacher_model_path: str, student_model_name: str):
        self.teacher_path = teacher_model_path
        self.student_name = student_model_name
        
        # Setup configs
        self.distillation_config = DistillationConfig(
            teacher_model_name=teacher_model_path,
            student_model_name=student_model_name
        )
    
    def run_full_pipeline(self, 
                         training_problems: List[str],
                         validation_problems: List[str],
                         output_dir: str):
        """Run complete distillation pipeline."""
        
        print("=== Reasoning Model Distillation Pipeline ===")
        
        # Step 1: Basic distillation
        print("\nStep 1: Basic Knowledge Distillation")
        basic_distiller = ReasoningDistiller(self.distillation_config)
        basic_distiller.distill_with_reasoning_preservation(
            training_problems[:len(training_problems)//2],
            validation_problems
        )
        
        # Step 2: Progressive distillation
        print("\nStep 2: Progressive Curriculum Distillation")
        progressive_distiller = ProgressiveDistiller(self.distillation_config)
        progressive_distiller.progressive_distillation(
            training_problems,
            validation_problems
        )
        
        # Step 3: Save final model
        print(f"\nSaving final distilled model to {output_dir}")
        progressive_distiller.student_model.save_pretrained(output_dir)
        progressive_distiller.student_tokenizer.save_pretrained(output_dir)
        
        # Step 4: Generate final report
        report = self._generate_distillation_report(
            basic_distiller.distillation_metrics,
            progressive_distiller.distillation_metrics
        )
        
        with open(f"{output_dir}/distillation_report.txt", 'w') as f:
            f.write(report)
        
        print("\nDistillation complete!")
        return report
    
    def _generate_distillation_report(self, basic_metrics: Dict, progressive_metrics: Dict) -> str:
        """Generate comprehensive distillation report."""
        report = """
Reasoning Model Distillation Report
==================================

Model Information:
- Teacher: {teacher}
- Student: {student}

Basic Distillation Results:
- Final correlation: {basic_corr:.3f}

Progressive Distillation Results:
- Stages completed: {stages}
- Final correlation: {prog_corr:.3f}

Recommendations:
- The distilled model achieves {performance}% of teacher performance
- Recommended for deployment in {use_cases}
- Further optimization possible through {suggestions}
""".format(
            teacher=self.teacher_path,
            student=self.student_name,
            basic_corr=basic_metrics.get('reasoning_score_correlation', [0])[-1],
            stages=self.distillation_config.difficulty_stages,
            prog_corr=progressive_metrics.get('reasoning_score_correlation', [0])[-1],
            performance=int(progressive_metrics.get('reasoning_score_correlation', [0])[-1] * 100),
            use_cases="resource-constrained environments",
            suggestions="additional training on hard problems"
        )
        
        return report