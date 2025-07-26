"""
Reward model implementation for RLHF.
Includes training utilities and inference methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Dict, Tuple
import numpy as np


class RewardModel(PreTrainedModel):
    """
    Reward model that predicts scalar rewards for text sequences.
    Built on top of a pretrained language model.
    """
    
    def __init__(self, config, base_model_name_or_path: str = None):
        super().__init__(config)
        
        # Load base model
        if base_model_name_or_path:
            self.base_model = AutoModel.from_pretrained(base_model_name_or_path)
        else:
            self.base_model = AutoModel(config)
        
        # Reward head
        self.reward_head = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self.reward_head.weight.data.normal_(mean=0.0, std=0.02)
        self.reward_head.bias.data.zero_()
        
        # Store config
        self.config = config
    
    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = None) -> torch.Tensor:
        """
        Forward pass to compute rewards.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            rewards: Scalar rewards [batch_size, 1]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        
        # Pool hidden states - use last non-padding token
        if attention_mask is not None:
            # Find the last non-padding position for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            batch_idx = torch.arange(batch_size, device=hidden_states.device)
            
            # Extract the hidden state at the last position
            pooled_hidden = hidden_states[batch_idx, sequence_lengths]
        else:
            # If no attention mask, use the last token
            pooled_hidden = hidden_states[:, -1]
        
        # Compute reward
        rewards = self.reward_head(pooled_hidden)
        
        if not return_dict:
            return rewards
        
        return SequenceClassifierOutput(
            loss=None,
            logits=rewards,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
    
    def compute_reward(self,
                      input_ids: torch.LongTensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute rewards for sequences (inference mode)."""
        with torch.no_grad():
            rewards = self.forward(input_ids, attention_mask)
            if isinstance(rewards, SequenceClassifierOutput):
                rewards = rewards.logits
        return rewards


class RewardModelTrainer:
    """Trainer for reward model using preference data."""
    
    def __init__(self,
                 model: RewardModel,
                 tokenizer: AutoTokenizer,
                 learning_rate: float = 1e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 100,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training config
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        
        # Metrics
        self.global_step = 0
        self.training_history = []
    
    def compute_loss(self,
                    chosen_input_ids: torch.LongTensor,
                    chosen_attention_mask: torch.LongTensor,
                    rejected_input_ids: torch.LongTensor,
                    rejected_attention_mask: torch.LongTensor,
                    margin: float = 0.01) -> Tuple[torch.Tensor, Dict]:
        """
        Compute preference ranking loss.
        
        The loss encourages: reward(chosen) > reward(rejected) + margin
        """
        # Compute rewards for both chosen and rejected
        chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
        if isinstance(chosen_rewards, SequenceClassifierOutput):
            chosen_rewards = chosen_rewards.logits
            
        rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
        if isinstance(rejected_rewards, SequenceClassifierOutput):
            rejected_rewards = rejected_rewards.logits
        
        # Compute ranking loss
        # We want chosen_rewards > rejected_rewards
        # Using logsigmoid for numerical stability
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards - margin).mean()
        
        # Compute accuracy (how often chosen > rejected)
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        # Average rewards for monitoring
        avg_chosen_reward = chosen_rewards.mean().item()
        avg_rejected_reward = rejected_rewards.mean().item()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'avg_chosen_reward': avg_chosen_reward,
            'avg_rejected_reward': avg_rejected_reward,
            'reward_margin': avg_chosen_reward - avg_rejected_reward
        }
        
        return loss, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Compute loss
        loss, metrics = self.compute_loss(
            batch['chosen_input_ids'],
            batch['chosen_attention_mask'],
            batch['rejected_input_ids'],
            batch['rejected_attention_mask']
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        self.global_step += 1
        
        # Add global step to metrics
        metrics['global_step'] = self.global_step
        
        return metrics
    
    def evaluate(self, eval_dataloader) -> Dict:
        """Evaluate the reward model."""
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        all_chosen_rewards = []
        all_rejected_rewards = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                loss, metrics = self.compute_loss(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask'],
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask']
                )
                
                batch_size = batch['chosen_input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_accuracy += metrics['accuracy'] * batch_size
                total_samples += batch_size
                
                # Collect rewards for analysis
                chosen_rewards = self.model(
                    batch['chosen_input_ids'], 
                    batch['chosen_attention_mask']
                )
                if isinstance(chosen_rewards, SequenceClassifierOutput):
                    chosen_rewards = chosen_rewards.logits
                    
                rejected_rewards = self.model(
                    batch['rejected_input_ids'], 
                    batch['rejected_attention_mask']
                )
                if isinstance(rejected_rewards, SequenceClassifierOutput):
                    rejected_rewards = rejected_rewards.logits
                
                all_chosen_rewards.extend(chosen_rewards.cpu().numpy().flatten())
                all_rejected_rewards.extend(rejected_rewards.cpu().numpy().flatten())
        
        # Compute statistics
        eval_metrics = {
            'eval_loss': total_loss / total_samples,
            'eval_accuracy': total_accuracy / total_samples,
            'eval_chosen_reward_mean': np.mean(all_chosen_rewards),
            'eval_chosen_reward_std': np.std(all_chosen_rewards),
            'eval_rejected_reward_mean': np.mean(all_rejected_rewards),
            'eval_rejected_reward_std': np.std(all_rejected_rewards),
            'eval_reward_margin': np.mean(all_chosen_rewards) - np.mean(all_rejected_rewards)
        }
        
        return eval_metrics
    
    def save_model(self, output_path: str):
        """Save the reward model."""
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save training history
        import json
        with open(f"{output_path}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)


class RewardModelEnsemble:
    """
    Ensemble of reward models for more robust reward estimation.
    Helps prevent reward hacking.
    """
    
    def __init__(self, model_paths: list, tokenizer: AutoTokenizer, device: str = 'cuda'):
        self.models = []
        self.tokenizer = tokenizer
        self.device = device
        
        # Load all models
        for path in model_paths:
            model = RewardModel.from_pretrained(path)
            model.to(device)
            model.eval()
            self.models.append(model)
    
    def compute_reward(self,
                      input_ids: torch.LongTensor,
                      attention_mask: Optional[torch.Tensor] = None,
                      return_all: bool = False) -> torch.Tensor:
        """
        Compute ensemble reward.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            return_all: If True, return all model predictions
            
        Returns:
            rewards: Mean reward across ensemble (or all rewards if return_all=True)
        """
        all_rewards = []
        
        with torch.no_grad():
            for model in self.models:
                rewards = model.compute_reward(input_ids, attention_mask)
                all_rewards.append(rewards)
        
        all_rewards = torch.stack(all_rewards)  # [n_models, batch_size, 1]
        
        if return_all:
            return all_rewards
        else:
            # Return mean reward
            return all_rewards.mean(dim=0)
    
    def compute_uncertainty(self,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reward uncertainty (std across ensemble)."""
        all_rewards = self.compute_reward(input_ids, attention_mask, return_all=True)
        return all_rewards.std(dim=0)