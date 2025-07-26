"""
Core GRPO trainer implementation.
Eliminates value function for efficient LLM training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import time
from tqdm import tqdm
import json


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model settings
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    
    # GRPO hyperparameters
    group_size: int = 8
    clip_epsilon: float = 0.2
    kl_coef: float = 0.04
    target_kl: Optional[float] = 0.01
    
    # Generation settings
    max_prompt_length: int = 512
    max_response_length: int = 2048
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    
    # Training settings
    learning_rate: float = 5e-6
    batch_size: int = 4  # Number of prompts per batch
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Optimization
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = False
    
    # Logging
    log_interval: int = 10
    save_interval: int = 1000


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    Implements the core GRPO algorithm without value functions.
    """
    
    def __init__(self, config: GRPOConfig, reward_fn: Callable):
        self.config = config
        self.reward_fn = reward_fn
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path or config.model_name_or_path
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Policy model (trainable)
        self.policy = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        ).to(config.device)
        
        # Reference model (frozen)
        self.ref_policy = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        ).to(config.device)
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
        
        # Adaptive KL controller
        self.kl_controller = AdaptiveKLController(
            target_kl=config.target_kl,
            initial_coef=config.kl_coef
        )
        
        # Training state
        self.global_step = 0
        self.best_reward = -float('inf')
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
    
    def generate_groups(self, prompts: List[str]) -> List[Dict]:
        """
        Generate groups of responses for each prompt.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            List of group dictionaries containing responses and metadata
        """
        all_groups = []
        
        for prompt in prompts:
            # Tokenize prompt
            prompt_encoding = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_prompt_length,
                padding=True
            ).to(self.config.device)
            
            group_data = {
                'prompt': prompt,
                'prompt_ids': prompt_encoding['input_ids'],
                'prompt_mask': prompt_encoding['attention_mask'],
                'responses': [],
                'response_ids': [],
                'response_masks': [],
                'rewards': []
            }
            
            # Generate multiple responses
            for _ in range(self.config.group_size):
                with torch.no_grad():
                    # Generate with sampling
                    output = self.policy.generate(
                        input_ids=prompt_encoding['input_ids'],
                        attention_mask=prompt_encoding['attention_mask'],
                        max_new_tokens=self.config.max_response_length,
                        temperature=self.config.temperature,
                        do_sample=True,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Extract response (remove prompt)
                    prompt_length = prompt_encoding['input_ids'].shape[1]
                    response_ids = output[:, prompt_length:]
                    
                    # Decode response
                    response_text = self.tokenizer.decode(
                        response_ids[0],
                        skip_special_tokens=True
                    )
                    
                    # Compute reward
                    reward = self.reward_fn(prompt, response_text)
                    
                    # Create attention mask for full sequence
                    full_mask = torch.ones_like(output)
                    
                    group_data['responses'].append(response_text)
                    group_data['response_ids'].append(output)
                    group_data['response_masks'].append(full_mask)
                    group_data['rewards'].append(reward)
            
            # Compute group-relative advantages
            rewards = np.array(group_data['rewards'])
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            
            # Normalize advantages
            advantages = (rewards - mean_reward) / std_reward
            group_data['advantages'] = advantages
            group_data['mean_reward'] = mean_reward
            group_data['std_reward'] = std_reward
            
            all_groups.append(group_data)
        
        return all_groups
    
    def compute_loss(self, group_data: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GRPO loss for a group of responses.
        
        Args:
            group_data: Dictionary containing group information
            
        Returns:
            loss: Total loss for the group
            metrics: Dictionary of metrics
        """
        total_loss = 0
        metrics = {
            'pg_loss': 0,
            'kl_div': 0,
            'kl_penalty': 0,
            'clip_fraction': 0,
            'mean_reward': group_data['mean_reward'],
            'responses_used': 0
        }
        
        # Process each response in the group
        for i in range(self.config.group_size):
            advantage = group_data['advantages'][i]
            
            # Skip responses with small advantages
            if abs(advantage) < 0.1:
                continue
            
            response_ids = group_data['response_ids'][i]
            response_mask = group_data['response_masks'][i]
            
            # Forward pass through policy
            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                policy_outputs = self.policy(
                    input_ids=response_ids,
                    attention_mask=response_mask,
                    labels=response_ids  # For loss computation
                )
                
                # Get logits
                logits = policy_outputs.logits
            
            # Compute log probabilities
            log_probs = self._compute_log_probs(logits, response_ids)
            
            # Get reference log probabilities
            with torch.no_grad():
                ref_outputs = self.ref_policy(
                    input_ids=response_ids,
                    attention_mask=response_mask
                )
                ref_logits = ref_outputs.logits
                ref_log_probs = self._compute_log_probs(ref_logits, response_ids)
            
            # Importance sampling ratio
            ratio = torch.exp(log_probs - ref_log_probs)
            
            # PPO-style clipped objective
            pg_loss1 = -advantage * ratio
            pg_loss2 = -advantage * torch.clamp(
                ratio,
                1 - self.config.clip_epsilon,
                1 + self.config.clip_epsilon
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # KL divergence
            kl_div = (log_probs - ref_log_probs).mean()
            
            # KL penalty
            kl_penalty = self.kl_controller.kl_coef * kl_div
            
            # Total loss for this response
            loss = pg_loss + kl_penalty
            total_loss += loss
            
            # Update metrics
            metrics['pg_loss'] += pg_loss.item()
            metrics['kl_div'] += kl_div.item()
            metrics['kl_penalty'] += kl_penalty.item()
            metrics['clip_fraction'] += (
                (ratio < 1 - self.config.clip_epsilon) |
                (ratio > 1 + self.config.clip_epsilon)
            ).float().mean().item()
            metrics['responses_used'] += 1
        
        # Average over valid responses
        if metrics['responses_used'] > 0:
            total_loss /= metrics['responses_used']
            for key in ['pg_loss', 'kl_div', 'kl_penalty', 'clip_fraction']:
                metrics[key] /= metrics['responses_used']
        
        return total_loss, metrics
    
    def _compute_log_probs(self, logits: torch.Tensor, 
                          labels: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of labels under the model.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab]
            labels: Token IDs [batch, seq_len]
            
        Returns:
            Log probabilities summed over sequence
        """
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs of actual tokens
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        masked_log_probs = gathered_log_probs * mask
        
        # Sum over sequence
        return masked_log_probs.sum(dim=-1)
    
    def train_step(self, batch: List[str]) -> Dict:
        """
        Single training step of GRPO.
        
        Args:
            batch: List of prompts
            
        Returns:
            Dictionary of metrics
        """
        # Generate response groups
        groups = self.generate_groups(batch)
        
        # Accumulate losses
        total_loss = 0
        all_metrics = {}
        
        for group_data in groups:
            # Compute loss for this group
            loss, metrics = self.compute_loss(group_data)
            
            # Scale by gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
            
            # Accumulate metrics
            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
        
        # Optimization step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        # Update KL controller
        mean_kl = np.mean(all_metrics.get('kl_div', [0]))
        self.kl_controller.update(mean_kl)
        
        # Average metrics
        avg_metrics = {
            k: np.mean(v) for k, v in all_metrics.items()
        }
        avg_metrics['loss'] = total_loss
        avg_metrics['kl_coef'] = self.kl_controller.kl_coef
        avg_metrics['step'] = self.global_step
        
        self.global_step += 1
        
        return avg_metrics
    
    def train(self, 
              train_dataloader: DataLoader,
              num_epochs: int = 1,
              eval_dataloader: Optional[DataLoader] = None,
              output_dir: str = './output'):
        """
        Main training loop.
        
        Args:
            train_dataloader: DataLoader providing prompts
            num_epochs: Number of training epochs
            eval_dataloader: Optional validation dataloader
            output_dir: Directory to save checkpoints
        """
        self.policy.train()
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            progress_bar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Extract prompts from batch
                if isinstance(batch, dict):
                    prompts = batch['prompts']
                else:
                    prompts = batch
                
                # Training step
                metrics = self.train_step(prompts)
                epoch_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'reward': f"{metrics['mean_reward']:.2f}",
                    'kl': f"{metrics['kl_div']:.4f}"
                })
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(metrics)
                
                # Saving
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(
                        f"{output_dir}/checkpoint-{self.global_step}"
                    )
                
                # Evaluation
                if eval_dataloader and self.global_step % 1000 == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    print(f"\nEval metrics: {eval_metrics}\n")
            
            # End of epoch summary
            avg_epoch_metrics = {
                k: np.mean([m[k] for m in epoch_metrics if k in m])
                for k in epoch_metrics[0].keys()
            }
            print(f"\nEpoch {epoch+1} summary: {avg_epoch_metrics}\n")
        
        # Save final model
        self.save_checkpoint(f"{output_dir}/final")
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict:
        """Evaluate model on validation set."""
        self.policy.eval()
        
        all_rewards = []
        all_lengths = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                if isinstance(batch, dict):
                    prompts = batch['prompts']
                else:
                    prompts = batch
                
                # Generate single response per prompt
                for prompt in prompts:
                    prompt_encoding = self.tokenizer(
                        prompt,
                        return_tensors='pt',
                        truncation=True,
                        max_length=self.config.max_prompt_length
                    ).to(self.config.device)
                    
                    # Generate with low temperature for consistency
                    output = self.policy.generate(
                        **prompt_encoding,
                        max_new_tokens=self.config.max_response_length,
                        temperature=0.1,
                        do_sample=True,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p
                    )
                    
                    # Decode response
                    response = self.tokenizer.decode(
                        output[0, prompt_encoding['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # Compute reward
                    reward = self.reward_fn(prompt, response)
                    all_rewards.append(reward)
                    all_lengths.append(len(response.split()))
        
        self.policy.train()
        
        return {
            'eval_reward_mean': np.mean(all_rewards),
            'eval_reward_std': np.std(all_rewards),
            'eval_length_mean': np.mean(all_lengths),
            'eval_length_std': np.std(all_lengths)
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'best_reward': self.best_reward,
            'kl_coef': self.kl_controller.kl_coef,
            'optimizer_state': self.optimizer.state_dict()
        }
        torch.save(state, f"{path}/training_state.pt")
        
        # Save config
        with open(f"{path}/grpo_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        # Load model
        self.policy = AutoModelForCausalLM.from_pretrained(path).to(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load training state
        state = torch.load(f"{path}/training_state.pt", map_location=self.config.device)
        self.global_step = state['global_step']
        self.best_reward = state['best_reward']
        self.kl_controller.kl_coef = state['kl_coef']
        self.optimizer.load_state_dict(state['optimizer_state'])
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics (can be extended for wandb, tensorboard, etc.)."""
        log_str = f"Step {metrics['step']}: "
        log_str += f"loss={metrics['loss']:.4f}, "
        log_str += f"reward={metrics['mean_reward']:.2f}, "
        log_str += f"kl={metrics['kl_div']:.4f}, "
        log_str += f"kl_coef={metrics['kl_coef']:.4f}"
        print(log_str)


class AdaptiveKLController:
    """
    Adaptive KL coefficient controller.
    Adjusts KL penalty based on divergence from reference.
    """
    
    def __init__(self, target_kl: Optional[float] = 0.01, 
                 initial_coef: float = 0.04):
        self.target_kl = target_kl
        self.kl_coef = initial_coef
        
        # Control parameters
        self.kl_factor = 2.0  # Multiplicative factor for updates
        self.kl_min = 0.001
        self.kl_max = 1.0
        
        # History for smoothing
        self.kl_history = []
        self.history_size = 10
    
    def update(self, current_kl: float):
        """Update KL coefficient based on current divergence."""
        if self.target_kl is None:
            return
        
        # Add to history
        self.kl_history.append(current_kl)
        if len(self.kl_history) > self.history_size:
            self.kl_history.pop(0)
        
        # Use smoothed KL
        smooth_kl = np.mean(self.kl_history)
        
        # Proportional control
        if smooth_kl > self.target_kl * 1.5:
            # KL too high, increase penalty
            self.kl_coef = min(self.kl_coef * self.kl_factor, self.kl_max)
        elif smooth_kl < self.target_kl * 0.5:
            # KL too low, decrease penalty
            self.kl_coef = max(self.kl_coef / self.kl_factor, self.kl_min)
        
        return self.kl_coef