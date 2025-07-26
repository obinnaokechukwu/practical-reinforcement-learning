"""
Core PPO implementation.
Handles the main training loop and policy updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Tuple, Callable
import time

from .networks import PPONetwork, DiscreteActionPPONetwork
from .buffer import RolloutBuffer
from .utils import explained_variance


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    Supports continuous and discrete action spaces.
    """
    def __init__(self, 
                 obs_dim: int,
                 act_dim: int,
                 continuous: bool = True,
                 # Network parameters
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 activation: str = 'tanh',
                 # PPO parameters
                 lr_policy: float = 3e-4,
                 lr_value: float = 1e-3,
                 clip_epsilon: float = 0.2,
                 epochs: int = 10,
                 mini_batch_size: int = 64,
                 # GAE parameters
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 # Loss coefficients
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 # Other parameters
                 max_grad_norm: float = 0.5,
                 normalize_advantages: bool = True,
                 clip_value_loss: bool = True,
                 value_clip_epsilon: float = 0.2,
                 target_kl: Optional[float] = 0.01,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.continuous = continuous
        self.device = device
        
        # Create network
        if continuous:
            self.network = PPONetwork(
                obs_dim, act_dim, hidden_dims, activation
            ).to(device)
        else:
            self.network = DiscreteActionPPONetwork(
                obs_dim, act_dim, hidden_dims, activation
            ).to(device)
        
        # Create optimizer with separate learning rates
        self.optimizer = optim.Adam([
            {'params': self.network.shared_encoder.parameters(), 'lr': lr_policy},
            {'params': self.network.policy_mean.parameters() if continuous 
                      else self.network.policy_head.parameters(), 'lr': lr_policy},
            {'params': [self.network.policy_log_std] if continuous else [], 'lr': lr_policy},
            {'params': self.network.value_head.parameters(), 'lr': lr_value}
        ])
        
        # PPO parameters
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Loss coefficients
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Other parameters
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages
        self.clip_value_loss = clip_value_loss
        self.value_clip_epsilon = value_clip_epsilon
        self.target_kl = target_kl
        
        # Learning rate schedulers (optional)
        self.lr_scheduler_policy = None
        self.lr_scheduler_value = None
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Select action given observation.
        
        Args:
            obs: Observation array
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action, log_prob, value, _ = self.network(obs_tensor, deterministic)
            
            # Convert action to numpy
            action_np = action.cpu().numpy()
            
            # Clip actions if continuous (ensure within bounds)
            if self.continuous:
                action_np = np.clip(action_np, -1.0, 1.0)
        
        return action_np, log_prob, value
    
    def evaluate_value(self, obs: np.ndarray) -> torch.Tensor:
        """Get value estimate for observations."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            value = self.network.get_value(obs_tensor)
        return value
    
    def update(self, rollout_buffer: RolloutBuffer, 
               lr_scheduler_progress: Optional[float] = None) -> Dict[str, float]:
        """
        Update policy and value function using collected rollout.
        
        Args:
            rollout_buffer: Buffer containing rollout data
            lr_scheduler_progress: Progress for learning rate scheduling (0 to 1)
            
        Returns:
            Dictionary of training statistics
        """
        # Update learning rates if schedulers are set
        if self.lr_scheduler_policy is not None and lr_scheduler_progress is not None:
            for param_group in self.optimizer.param_groups:
                if 'policy' in str(param_group['params'][0]):
                    param_group['lr'] = self.lr_scheduler_policy(lr_scheduler_progress)
                else:
                    param_group['lr'] = self.lr_scheduler_value(lr_scheduler_progress)
        
        # Initialize statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divs = []
        clip_fractions = []
        
        # Perform multiple epochs of updates
        for epoch in range(self.epochs):
            approx_kl_divs = []
            
            # Iterate through mini-batches
            for batch_data in rollout_buffer.get_samples(self.mini_batch_size):
                obs = batch_data['observations']
                actions = batch_data['actions']
                old_log_probs = batch_data['old_log_probs']
                advantages = batch_data['advantages']
                returns = batch_data['returns']
                old_values = batch_data['old_values']
                
                # Normalize advantages
                if self.normalize_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Evaluate actions under current policy
                log_probs, values, entropy = self.network.evaluate_actions(obs, actions)
                
                # Compute ratio for PPO objective
                ratio = torch.exp(log_probs - old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Track clipping
                clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
                clip_fractions.append(clip_fraction.item())
                
                # Value loss
                if self.clip_value_loss:
                    # Clipped value loss
                    values_pred_clipped = old_values + torch.clamp(
                        values - old_values, -self.value_clip_epsilon, self.value_clip_epsilon
                    )
                    value_loss_clipped = (returns - values_pred_clipped).pow(2)
                    value_loss_unclipped = (returns - values).pow(2)
                    value_loss = torch.max(value_loss_clipped, value_loss_unclipped).mean()
                else:
                    # Simple MSE loss
                    value_loss = nn.functional.mse_loss(values, returns)
                
                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.value_loss_coef * value_loss + 
                       self.entropy_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.mean().item())
                
                # Compute approximate KL divergence
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    approx_kl_divs.append(approx_kl.item())
            
            # Check early stopping based on KL divergence
            mean_kl = np.mean(approx_kl_divs)
            kl_divs.append(mean_kl)
            
            if self.target_kl is not None and mean_kl > 1.5 * self.target_kl:
                print(f"Early stopping at epoch {epoch} due to KL divergence: {mean_kl:.4f}")
                break
        
        # Compute explained variance for value function
        with torch.no_grad():
            all_values = []
            all_returns = []
            
            for batch_data in rollout_buffer.get_samples(batch_size=None):
                values = self.network.get_value(batch_data['observations'])
                all_values.append(values)
                all_returns.append(batch_data['returns'])
            
            all_values = torch.cat(all_values)
            all_returns = torch.cat(all_returns)
            explained_var = explained_variance(all_values, all_returns).item()
        
        # Return statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses),
            'kl_div': np.mean(kl_divs),
            'clip_fraction': np.mean(clip_fractions),
            'explained_variance': explained_var,
            'epochs_used': epoch + 1
        }
        
        return stats
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def set_lr_scheduler(self, 
                        lr_scheduler_policy: Optional[Callable] = None,
                        lr_scheduler_value: Optional[Callable] = None):
        """Set learning rate schedulers."""
        self.lr_scheduler_policy = lr_scheduler_policy
        self.lr_scheduler_value = lr_scheduler_value