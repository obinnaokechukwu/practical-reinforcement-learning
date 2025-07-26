"""
Chain-of-Thought emergence through reinforcement learning.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import numpy as np
import re
from dataclasses import dataclass
from tqdm import tqdm
import wandb


@dataclass
class CoTConfig:
    """Configuration for Chain-of-Thought training."""
    model_name: str
    thinking_reward_weight: float = 0.3
    answer_reward_weight: float = 0.7
    
    # CoT-specific rewards
    step_clarity_weight: float = 0.2
    logical_flow_weight: float = 0.3
    completeness_weight: float = 0.2
    
    # Generation settings
    max_thinking_length: int = 1024
    max_answer_length: int = 256
    thinking_temperature: float = 0.8
    answer_temperature: float = 0.3
    
    # Training settings
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # CoT emergence settings
    force_thinking_steps: int = 3  # Minimum thinking steps
    thinking_token: str = "<think>"
    answer_token: str = "<answer>"
    step_token: str = "<step>"
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_wandb: bool = False


class ChainOfThoughtTrainer:
    """Trainer for emergent Chain-of-Thought reasoning."""
    
    def __init__(self, config: CoTConfig, reward_fn):
        self.config = config
        self.reward_fn = reward_fn
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._add_special_tokens()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name
        ).to(config.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Metrics tracking
        self.cot_emergence_metrics = {
            'avg_thinking_steps': [],
            'step_clarity_scores': [],
            'logical_flow_scores': [],
            'answer_quality_scores': []
        }
        
        if config.use_wandb:
            wandb.init(project="cot-emergence", config=config.__dict__)
    
    def _add_special_tokens(self):
        """Add CoT-specific special tokens."""
        special_tokens = {
            'additional_special_tokens': [
                self.config.thinking_token,
                self.config.answer_token,
                self.config.step_token
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
    
    def train_with_cot_rewards(self, problems: List[str], num_iterations: int = 1000):
        """Train model to develop Chain-of-Thought reasoning."""
        
        print("Training for Chain-of-Thought Emergence")
        print("=" * 50)
        
        for iteration in range(num_iterations):
            batch_problems = self._sample_batch(problems)
            
            total_loss = 0
            total_thinking_reward = 0
            total_answer_reward = 0
            
            for problem in batch_problems:
                # Generate CoT response
                response_data = self._generate_cot_response(problem)
                
                # Evaluate thinking and answer separately
                thinking_reward = self._evaluate_thinking_process(
                    response_data['thinking_steps']
                )
                answer_reward = self.reward_fn(
                    problem, 
                    response_data['final_answer']
                )
                
                # Compute combined reward
                total_reward = (
                    self.config.thinking_reward_weight * thinking_reward +
                    self.config.answer_reward_weight * answer_reward
                )
                
                # Reinforcement learning update
                loss = self._compute_rl_loss(response_data, total_reward)
                
                # Accumulate gradients
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item()
                total_thinking_reward += thinking_reward
                total_answer_reward += answer_reward
            
            # Gradient update
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Log metrics
            if iteration % 10 == 0:
                self._log_iteration(
                    iteration,
                    total_loss,
                    total_thinking_reward / len(batch_problems),
                    total_answer_reward / len(batch_problems)
                )
    
    def _generate_cot_response(self, problem: str) -> Dict:
        """Generate response with explicit thinking steps."""
        # Prepare prompt with CoT structure
        prompt = f"{problem}\n{self.config.thinking_token}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.config.device)
        
        thinking_steps = []
        current_step = ""
        all_tokens = []
        all_log_probs = []
        
        # Generate thinking process
        with torch.no_grad():
            # First, generate thinking steps
            for _ in range(self.config.max_thinking_length):
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
                
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits / self.config.thinking_temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Track log probability
                log_prob = torch.log(probs[0, next_token.item()])
                all_log_probs.append(log_prob.item())
                all_tokens.append(next_token.item())
                
                # Decode token
                token_text = self.tokenizer.decode(next_token[0])
                
                # Check for step boundaries
                if self.config.step_token in token_text:
                    if current_step.strip():
                        thinking_steps.append(current_step.strip())
                    current_step = ""
                elif self.config.answer_token in token_text:
                    if current_step.strip():
                        thinking_steps.append(current_step.strip())
                    break
                else:
                    current_step += token_text
                
                # Update inputs
                inputs['input_ids'] = torch.cat([
                    inputs['input_ids'],
                    next_token
                ], dim=1)
                
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = torch.cat([
                        inputs['attention_mask'],
                        torch.ones(1, 1).to(self.config.device)
                    ], dim=1)
            
            # Generate final answer
            answer_prompt = f"{self.config.answer_token}"
            answer_inputs = self.tokenizer(
                answer_prompt,
                return_tensors='pt'
            ).to(self.config.device)
            
            inputs['input_ids'] = torch.cat([
                inputs['input_ids'],
                answer_inputs['input_ids']
            ], dim=1)
            
            final_answer = ""
            for _ in range(self.config.max_answer_length):
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                
                probs = F.softmax(logits / self.config.answer_temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                log_prob = torch.log(probs[0, next_token.item()])
                all_log_probs.append(log_prob.item())
                all_tokens.append(next_token.item())
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                token_text = self.tokenizer.decode(next_token[0])
                final_answer += token_text
                
                inputs['input_ids'] = torch.cat([
                    inputs['input_ids'],
                    next_token
                ], dim=1)
        
        # Ensure minimum thinking steps
        while len(thinking_steps) < self.config.force_thinking_steps:
            thinking_steps.append(f"Step {len(thinking_steps) + 1}: Further analysis needed.")
        
        return {
            'thinking_steps': thinking_steps,
            'final_answer': final_answer.strip(),
            'all_tokens': all_tokens,
            'all_log_probs': all_log_probs,
            'full_response': self.tokenizer.decode(all_tokens)
        }
    
    def _evaluate_thinking_process(self, thinking_steps: List[str]) -> float:
        """Evaluate quality of thinking process."""
        if not thinking_steps:
            return 0.0
        
        # Step clarity score
        clarity_score = self._compute_step_clarity(thinking_steps)
        
        # Logical flow score
        flow_score = self._compute_logical_flow(thinking_steps)
        
        # Completeness score
        completeness_score = min(len(thinking_steps) / self.config.force_thinking_steps, 1.0)
        
        # Combined thinking reward
        thinking_reward = (
            self.config.step_clarity_weight * clarity_score +
            self.config.logical_flow_weight * flow_score +
            self.config.completeness_weight * completeness_score
        )
        
        # Track metrics
        self.cot_emergence_metrics['avg_thinking_steps'].append(len(thinking_steps))
        self.cot_emergence_metrics['step_clarity_scores'].append(clarity_score)
        self.cot_emergence_metrics['logical_flow_scores'].append(flow_score)
        
        return thinking_reward
    
    def _compute_step_clarity(self, steps: List[str]) -> float:
        """Compute clarity score for thinking steps."""
        clarity_scores = []
        
        for step in steps:
            # Check for clear structure
            has_conclusion = any(word in step.lower() for word in 
                               ['therefore', 'thus', 'so', 'conclude', 'result'])
            has_reasoning = any(word in step.lower() for word in 
                              ['because', 'since', 'as', 'given', 'if'])
            
            # Check for mathematical/logical operators
            has_operators = bool(re.search(r'[=+\-*/><]', step))
            
            # Length check (not too short, not too long)
            good_length = 20 < len(step) < 200
            
            # Compute step clarity
            clarity = sum([
                has_conclusion * 0.3,
                has_reasoning * 0.3,
                has_operators * 0.2,
                good_length * 0.2
            ])
            
            clarity_scores.append(clarity)
        
        return np.mean(clarity_scores) if clarity_scores else 0.0
    
    def _compute_logical_flow(self, steps: List[str]) -> float:
        """Compute logical flow between steps."""
        if len(steps) < 2:
            return 0.5
        
        flow_scores = []
        
        for i in range(1, len(steps)):
            prev_step = steps[i-1].lower()
            curr_step = steps[i].lower()
            
            # Check for continuity indicators
            has_reference = any(word in curr_step for word in 
                              ['this', 'that', 'previous', 'above', 'we'])
            
            # Check for logical connectors
            has_connector = any(word in curr_step for word in 
                              ['next', 'then', 'now', 'furthermore', 'additionally'])
            
            # Check for shared concepts (simple word overlap)
            prev_words = set(prev_step.split())
            curr_words = set(curr_step.split())
            word_overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))
            
            # Compute flow score
            flow = has_reference * 0.3 + has_connector * 0.3 + word_overlap * 0.4
            flow_scores.append(flow)
        
        return np.mean(flow_scores) if flow_scores else 0.0
    
    def _compute_rl_loss(self, response_data: Dict, reward: float) -> torch.Tensor:
        """Compute reinforcement learning loss."""
        # Get log probabilities
        log_probs = torch.tensor(
            response_data['all_log_probs'],
            device=self.config.device,
            requires_grad=True
        )
        
        # Advantage (reward - baseline)
        # Simple baseline: running average of rewards
        if not hasattr(self, 'reward_baseline'):
            self.reward_baseline = reward
        else:
            self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * reward
        
        advantage = reward - self.reward_baseline
        
        # Policy gradient loss
        loss = -advantage * log_probs.mean()
        
        return loss
    
    def _sample_batch(self, problems: List[str]) -> List[str]:
        """Sample batch of problems."""
        if len(problems) > self.config.batch_size:
            return np.random.choice(problems, self.config.batch_size, replace=False).tolist()
        return problems
    
    def _log_iteration(self, iteration: int, loss: float, 
                      thinking_reward: float, answer_reward: float):
        """Log training metrics."""
        print(f"\nIteration {iteration}")
        print(f"Loss: {loss:.4f}")
        print(f"Thinking Reward: {thinking_reward:.3f}")
        print(f"Answer Reward: {answer_reward:.3f}")
        
        if self.cot_emergence_metrics['avg_thinking_steps']:
            avg_steps = np.mean(self.cot_emergence_metrics['avg_thinking_steps'][-100:])
            print(f"Avg Thinking Steps: {avg_steps:.1f}")
        
        if self.config.use_wandb:
            wandb.log({
                'iteration': iteration,
                'loss': loss,
                'thinking_reward': thinking_reward,
                'answer_reward': answer_reward,
                'avg_thinking_steps': avg_steps if 'avg_steps' in locals() else 0
            })


class ChainOfThoughtAnalyzer:
    """Analyzer for Chain-of-Thought patterns."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pattern_counts = {
            'step_based': 0,
            'because_therefore': 0,
            'equation_based': 0,
            'enumeration': 0,
            'recursive': 0
        }
    
    def analyze_cot_patterns(self, responses: List[str]) -> Dict:
        """Analyze patterns in Chain-of-Thought responses."""
        for response in responses:
            self._categorize_pattern(response)
        
        total = sum(self.pattern_counts.values())
        pattern_distribution = {
            pattern: count / total if total > 0 else 0
            for pattern, count in self.pattern_counts.items()
        }
        
        analysis = {
            'pattern_distribution': pattern_distribution,
            'dominant_pattern': max(pattern_distribution, key=pattern_distribution.get),
            'pattern_diversity': self._compute_diversity(pattern_distribution),
            'avg_step_length': self._compute_avg_step_length(responses),
            'reasoning_depth': self._compute_reasoning_depth(responses)
        }
        
        return analysis
    
    def _categorize_pattern(self, response: str):
        """Categorize CoT pattern type."""
        response_lower = response.lower()
        
        # Step-based reasoning
        if re.search(r'step \d+:|first,|second,|finally,', response_lower):
            self.pattern_counts['step_based'] += 1
        
        # Causal reasoning
        elif 'because' in response_lower and 'therefore' in response_lower:
            self.pattern_counts['because_therefore'] += 1
        
        # Equation-based
        elif re.search(r'[=+\-*/]', response) and response.count('=') > 2:
            self.pattern_counts['equation_based'] += 1
        
        # Enumeration
        elif re.search(r'\d+\)|[a-z]\)', response_lower):
            self.pattern_counts['enumeration'] += 1
        
        # Recursive/nested
        elif response.count('(') > 3 or 'recursively' in response_lower:
            self.pattern_counts['recursive'] += 1
    
    def _compute_diversity(self, distribution: Dict[str, float]) -> float:
        """Compute diversity of reasoning patterns (entropy)."""
        entropy = 0
        for p in distribution.values():
            if p > 0:
                entropy -= p * np.log(p)
        
        # Normalize to [0, 1]
        max_entropy = np.log(len(distribution))
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _compute_avg_step_length(self, responses: List[str]) -> float:
        """Compute average length of reasoning steps."""
        all_steps = []
        
        for response in responses:
            # Simple heuristic: split by newlines or step markers
            steps = re.split(r'\n|step \d+:|first,|second,|finally,', response, flags=re.I)
            steps = [s.strip() for s in steps if s.strip()]
            all_steps.extend(steps)
        
        if all_steps:
            return np.mean([len(step.split()) for step in all_steps])
        return 0
    
    def _compute_reasoning_depth(self, responses: List[str]) -> float:
        """Estimate depth of reasoning (number of logical connections)."""
        depths = []
        
        reasoning_markers = [
            'therefore', 'thus', 'hence', 'so', 'because', 'since',
            'if', 'then', 'implies', 'leads to', 'results in'
        ]
        
        for response in responses:
            response_lower = response.lower()
            depth = sum(1 for marker in reasoning_markers if marker in response_lower)
            depths.append(depth)
        
        return np.mean(depths) if depths else 0
    
    def generate_cot_report(self, test_problems: List[str], model) -> str:
        """Generate comprehensive CoT analysis report."""
        responses = []
        
        # Generate responses for test problems
        for problem in test_problems:
            with torch.no_grad():
                inputs = self.tokenizer(problem, return_tensors='pt')
                outputs = model.generate(
                    **inputs,
                    max_length=1024,
                    temperature=0.7,
                    do_sample=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
        
        # Analyze patterns
        analysis = self.analyze_cot_patterns(responses)
        
        # Generate report
        report = f"""
Chain-of-Thought Analysis Report
================================

Pattern Distribution:
"""
        for pattern, freq in analysis['pattern_distribution'].items():
            report += f"- {pattern}: {freq:.1%}\n"
        
        report += f"""
Dominant Pattern: {analysis['dominant_pattern']}
Pattern Diversity: {analysis['pattern_diversity']:.3f}
Average Step Length: {analysis['avg_step_length']:.1f} words
Reasoning Depth: {analysis['reasoning_depth']:.2f} logical connections

Sample Chain-of-Thought Response:
---------------------------------
{responses[0] if responses else 'No responses generated'}
"""
        
        return report