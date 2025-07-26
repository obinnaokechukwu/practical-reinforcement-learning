"""
PPO trainer specifically designed for language models.
Implements memory-efficient training with KL penalties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import copy
from tqdm import tqdm


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Model paths
    model_name_or_path: str
    reward_model_path: str
    
    # PPO hyperparameters
    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    
    # PPO specific
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    target_kl: float = 0.01
    
    # Coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    
    # Generation parameters
    max_length: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    
    # Training parameters
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class PPOTrainer:
    """PPO trainer for language models with RLHF."""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # Load models
        self.policy = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        self.policy.to(config.device)
        
        # Reference model for KL penalty
        self.ref_policy = copy.deepcopy(self.policy)
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad = False
        
        # Value model (separate head or model)
        self.value_model = self._init_value_model()
        
        # Reward model
        from .reward_model import RewardModel
        self.reward_model = RewardModel.from_pretrained(config.reward_model_path)
        self.reward_model.to(config.device)
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(), lr=config.value_learning_rate
        )
        
        # Training state
        self.global_step = 0
        self.kl_coef = config.kl_coef
    
    def _init_value_model(self) -> nn.Module:
        """Initialize value model as a separate head."""
        class ValueHead(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.dense = nn.Linear(hidden_size, hidden_size)
                self.dropout = nn.Dropout(0.1)
                self.out = nn.Linear(hidden_size, 1)
            
            def forward(self, hidden_states):
                output = self.dropout(F.relu(self.dense(hidden_states)))
                return self.out(output)
        
        # Get hidden size from policy model
        hidden_size = self.policy.config.hidden_size
        value_model = ValueHead(hidden_size).to(self.config.device)
        
        return value_model
    
    def generate_batch(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Generate responses for a batch of prompts."""
        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length // 2  # Leave room for response
        ).to(self.config.device)
        
        # Track generation data
        batch_data = {
            'prompt_input_ids': prompt_encodings.input_ids,
            'prompt_attention_mask': prompt_encodings.attention_mask,
            'prompts': prompts,
            'responses': [],
            'response_input_ids': [],
            'response_attention_mask': [],
            'response_log_probs': [],
            'response_values': [],
            'response_rewards': []
        }
        
        # Generate responses
        with torch.no_grad():
            # Generate with sampling
            outputs = self.policy.generate(
                **prompt_encodings,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            response_ids = outputs.sequences
            
            # Decode responses
            responses = self.tokenizer.batch_decode(
                response_ids[:, prompt_encodings.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            batch_data['responses'] = responses
            batch_data['response_input_ids'] = response_ids
            
            # Compute initial log probabilities
            log_probs = self._compute_log_probs(
                response_ids, outputs.scores, prompt_encodings.input_ids.shape[1]
            )
            batch_data['response_log_probs'] = log_probs
            
            # Compute values
            values = self._compute_values(response_ids)
            batch_data['response_values'] = values
            
            # Compute rewards
            rewards = self._compute_rewards(response_ids, prompt_encodings.input_ids)
            batch_data['response_rewards'] = rewards
        
        return batch_data
    
    def _compute_log_probs(self, 
                          sequences: torch.Tensor, 
                          scores: Tuple[torch.Tensor],
                          prompt_length: int) -> torch.Tensor:
        """Compute log probabilities of generated tokens."""
        batch_size = sequences.shape[0]
        seq_length = len(scores)
        
        log_probs = torch.zeros(batch_size, seq_length).to(self.config.device)
        
        for t, score in enumerate(scores):
            # Get the token that was actually sampled
            token_id = sequences[:, prompt_length + t]
            
            # Get log probability of that token
            log_prob = F.log_softmax(score, dim=-1)
            token_log_prob = log_prob.gather(1, token_id.unsqueeze(1)).squeeze(1)
            
            log_probs[:, t] = token_log_prob
        
        return log_probs
    
    def _compute_values(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute value estimates for sequences."""
        with torch.no_grad():
            # Get hidden states from policy model
            outputs = self.policy(sequences, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # Compute values for each position
            values = self.value_model(hidden_states).squeeze(-1)
        
        return values
    
    def _compute_rewards(self, 
                        response_ids: torch.Tensor,
                        prompt_ids: torch.Tensor) -> torch.Tensor:
        """Compute rewards using reward model and KL penalty."""
        batch_size = response_ids.shape[0]
        
        # Get reward from reward model
        with torch.no_grad():
            rewards = self.reward_model.compute_reward(response_ids)
            rewards = rewards.squeeze(-1)  # [batch_size]
        
        # Compute KL penalty
        kl_penalty = self._compute_kl_penalty(response_ids)
        
        # Combine rewards
        total_rewards = rewards - self.kl_coef * kl_penalty
        
        # Create sparse reward (only at the end of generation)
        seq_length = response_ids.shape[1] - prompt_ids.shape[1]
        sparse_rewards = torch.zeros(batch_size, seq_length).to(self.config.device)
        sparse_rewards[:, -1] = total_rewards
        
        return sparse_rewards
    
    def _compute_kl_penalty(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from reference policy."""
        with torch.no_grad():
            # Get logits from current and reference policies
            current_logits = self.policy(sequences).logits
            ref_logits = self.ref_policy(sequences).logits
            
            # Compute KL divergence
            current_log_probs = F.log_softmax(current_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            # KL(current || ref) for each token
            kl = F.kl_div(ref_log_probs, current_log_probs, reduction='none', log_target=True)
            kl = kl.sum(dim=-1)  # Sum over vocabulary
            
            # Average over sequence
            kl_penalty = kl.mean(dim=-1)  # [batch_size]
        
        return kl_penalty
    
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns."""
        batch_size, seq_length = rewards.shape
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute GAE
        last_gae_lam = 0
        for t in reversed(range(seq_length)):
            if t == seq_length - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.config.gamma * next_value - values[:, t]
            last_gae_lam = delta + self.config.gamma * self.config.lam * last_gae_lam
            advantages[:, t] = last_gae_lam
            returns[:, t] = advantages[:, t] + values[:, t]
        
        return advantages, returns
    
    def ppo_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform PPO update on a batch."""
        # Extract data
        response_ids = batch_data['response_input_ids']
        old_log_probs = batch_data['response_log_probs']
        old_values = batch_data['response_values']
        rewards = batch_data['response_rewards']
        prompt_length = batch_data['prompt_input_ids'].shape[1]
        
        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, old_values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        update_count = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Mini-batch updates
            for start_idx in range(0, response_ids.shape[0], self.config.mini_batch_size):
                end_idx = min(start_idx + self.config.mini_batch_size, response_ids.shape[0])
                
                # Mini-batch data
                mb_response_ids = response_ids[start_idx:end_idx]
                mb_old_log_probs = old_log_probs[start_idx:end_idx, prompt_length-1:-1]
                mb_advantages = advantages[start_idx:end_idx]
                mb_returns = returns[start_idx:end_idx]
                
                # Forward pass
                outputs = self.policy(mb_response_ids, output_hidden_states=True)
                logits = outputs.logits[:, prompt_length-1:-1]  # Only response logits
                hidden_states = outputs.hidden_states[-1][:, prompt_length-1:-1]
                
                # Compute new log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get log probs of taken actions
                response_token_ids = mb_response_ids[:, prompt_length:]
                new_log_probs = log_probs.gather(
                    2, response_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = self.value_model(hidden_states).squeeze(-1)
                
                if self.config.value_clip_epsilon > 0:
                    # Clipped value loss
                    value_pred_clipped = old_values[start_idx:end_idx] + torch.clamp(
                        values - old_values[start_idx:end_idx],
                        -self.config.value_clip_epsilon,
                        self.config.value_clip_epsilon
                    )
                    value_losses = (values - mb_returns) ** 2
                    value_losses_clipped = (value_pred_clipped - mb_returns) ** 2
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy bonus
                entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_loss_coef * value_loss - 
                       self.config.entropy_coef * entropy)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_model.parameters(), self.config.max_grad_norm
                )
                
                # Optimizer steps
                if (update_count + 1) % self.config.gradient_accumulation_steps == 0:
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                
                # Track metrics
                with torch.no_grad():
                    kl = (new_log_probs - mb_old_log_probs).mean()
                    total_kl += kl.item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1
            
            # Early stopping based on KL
            if total_kl / update_count > self.config.target_kl:
                break
        
        # Update KL coefficient if needed
        if total_kl / update_count > self.config.target_kl * 2:
            self.kl_coef *= 1.5
        elif total_kl / update_count < self.config.target_kl / 2:
            self.kl_coef *= 0.5
        
        self.global_step += 1
        
        return {
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy / update_count,
            'kl': total_kl / update_count,
            'kl_coef': self.kl_coef,
            'ppo_epochs_used': epoch + 1
        }
    
    def train(self, 
             train_dataloader: DataLoader,
             eval_dataloader: Optional[DataLoader] = None,
             num_epochs: int = 1,
             save_steps: int = 100,
             eval_steps: int = 50,
             output_dir: str = './rlhf_model'):
        """Main training loop."""
        
        self.policy.train()
        self.value_model.train()
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Generate responses
                prompts = batch['prompts']
                batch_data = self.generate_batch(prompts)
                
                # PPO update
                metrics = self.ppo_step(batch_data)
                epoch_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['policy_loss']:.4f}",
                    'kl': f"{metrics['kl']:.4f}"
                })
                
                # Save checkpoint
                if self.global_step % save_steps == 0:
                    self.save_checkpoint(f"{output_dir}/checkpoint-{self.global_step}")
                
                # Evaluation
                if eval_dataloader and self.global_step % eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader)
                    print(f"\nEval metrics at step {self.global_step}: {eval_metrics}\n")
            
            # End of epoch summary
            avg_metrics = {
                k: np.mean([m[k] for m in epoch_metrics])
                for k in epoch_metrics[0].keys()
            }
            print(f"\nEpoch {epoch+1} average metrics: {avg_metrics}\n")
        
        # Save final model
        self.save_checkpoint(output_dir)
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.policy.eval()
        
        all_rewards = []
        all_kl_penalties = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                prompts = batch['prompts']
                batch_data = self.generate_batch(prompts)
                
                # Get rewards
                rewards = batch_data['response_rewards'][:, -1].cpu().numpy()
                all_rewards.extend(rewards)
                
                # Get KL penalties
                response_ids = batch_data['response_input_ids']
                kl_penalties = self._compute_kl_penalty(response_ids).cpu().numpy()
                all_kl_penalties.extend(kl_penalties)
        
        self.policy.train()
        
        return {
            'eval_reward_mean': np.mean(all_rewards),
            'eval_reward_std': np.std(all_rewards),
            'eval_kl_mean': np.mean(all_kl_penalties),
            'eval_kl_std': np.std(all_kl_penalties)
        }
    
    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save policy model
        self.policy.save_pretrained(f"{output_dir}/policy")
        
        # Save value model
        torch.save(self.value_model.state_dict(), f"{output_dir}/value_model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(f"{output_dir}/tokenizer")
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'kl_coef': self.kl_coef,
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, f"{output_dir}/training_state.pt")
        
        print(f"Checkpoint saved to {output_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model checkpoint."""
        # Load policy model
        self.policy = AutoModelForCausalLM.from_pretrained(f"{checkpoint_dir}/policy")
        self.policy.to(self.config.device)
        
        # Load value model
        self.value_model.load_state_dict(
            torch.load(f"{checkpoint_dir}/value_model.pt", map_location=self.config.device)
        )
        
        # Load training state
        state = torch.load(f"{checkpoint_dir}/training_state.pt", map_location=self.config.device)
        self.global_step = state['global_step']
        self.kl_coef = state['kl_coef']
        self.policy_optimizer.load_state_dict(state['policy_optimizer'])
        self.value_optimizer.load_state_dict(state['value_optimizer'])
        
        print(f"Checkpoint loaded from {checkpoint_dir}")