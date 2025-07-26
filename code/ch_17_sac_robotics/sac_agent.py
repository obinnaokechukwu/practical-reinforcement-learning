"""
Core SAC agent implementation.
Includes automatic temperature tuning and ensemble Q-functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import copy

from networks import SoftQNetwork, GaussianPolicy, init_weights
from replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic agent with automatic temperature tuning.
    """
    
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 # Network architecture
                 hidden_dims: List[int] = [256, 256],
                 # Learning rates
                 lr_q: float = 3e-4,
                 lr_policy: float = 3e-4,
                 lr_alpha: float = 3e-4,
                 # SAC parameters
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 target_entropy: Optional[float] = None,
                 # Training parameters
                 batch_size: int = 256,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # Networks
        self.q1 = SoftQNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q2 = SoftQNetwork(obs_dim, act_dim, hidden_dims).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dims).to(device)
        
        # Initialize weights
        self.q1.apply(lambda m: init_weights(m, gain=1))
        self.q2.apply(lambda m: init_weights(m, gain=1))
        self.policy.apply(lambda m: init_weights(m, gain=1))
        
        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr_q)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        
        # Automatic temperature tuning
        if target_entropy is None:
            # Heuristic: -dim(A)
            self.target_entropy = -act_dim
        else:
            self.target_entropy = target_entropy
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        
        # Training statistics
        self.training_steps = 0
    
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for environment interaction."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy.get_action(obs_tensor, deterministic)
            action = action_tensor.squeeze(0).cpu().numpy()
        return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update SAC networks using a batch of transitions.
        
        Returns:
            Dictionary of training metrics
        """
        obs = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_observations']
        dones = batch['dones']
        
        # Update Q-functions
        q1_loss, q2_loss = self._update_q_functions(
            obs, actions, rewards, next_obs, dones
        )
        
        # Update policy
        policy_loss, entropy = self._update_policy(obs)
        
        # Update temperature
        alpha_loss = self._update_temperature(entropy)
        
        # Update target networks
        self._soft_update_target_networks()
        
        self.training_steps += 1
        
        return {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'policy_loss': policy_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item(),
            'entropy': -entropy.mean().item()
        }
    
    def _update_q_functions(self, obs: torch.Tensor, actions: torch.Tensor,
                           rewards: torch.Tensor, next_obs: torch.Tensor,
                           dones: torch.Tensor) -> Tuple[float, float]:
        """Update Q-functions using Bellman backup."""
        with torch.no_grad():
            # Sample next actions and compute target Q-values
            next_actions, next_log_probs = self.policy.sample(next_obs)
            q1_next_target = self.q1_target(next_obs, next_actions)
            q2_next_target = self.q2_target(next_obs, next_actions)
            
            # Use minimum Q-value to address overestimation
            min_q_next_target = torch.min(q1_next_target, q2_next_target)
            
            # Compute soft value with entropy term
            next_v = min_q_next_target - self.alpha * next_log_probs
            
            # Compute target Q-values
            q_target = rewards + (1 - dones) * self.gamma * next_v
        
        # Current Q-values
        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        
        # Q-function losses
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        # Update Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Update Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        return q1_loss.item(), q2_loss.item()
    
    def _update_policy(self, obs: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Update policy to maximize Q-value and entropy."""
        # Sample actions from current policy
        actions, log_probs = self.policy.sample(obs)
        
        # Compute Q-values for sampled actions
        q1_pi = self.q1(obs, actions)
        q2_pi = self.q2(obs, actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        # Policy loss: maximize Q-value and entropy
        policy_loss = (self.alpha * log_probs - min_q_pi).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), log_probs
    
    def _update_temperature(self, log_probs: torch.Tensor) -> float:
        """Update temperature parameter alpha."""
        # Detach log_probs to avoid affecting policy gradient
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update alpha value
        self.alpha = self.log_alpha.exp().detach()
        
        return alpha_loss.item()
    
    def _soft_update_target_networks(self):
        """Soft update target networks using Polyak averaging."""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'log_alpha': self.log_alpha,
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'training_steps': self.training_steps
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().detach()
        
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.training_steps = checkpoint['training_steps']


class SACAgentWithEnsemble(SACAgent):
    """
    SAC agent with ensemble Q-functions for uncertainty estimation.
    """
    
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 num_q_networks: int = 5,
                 uncertainty_threshold: float = 0.1,
                 **kwargs):
        
        # Initialize base agent
        super().__init__(obs_dim, act_dim, **kwargs)
        
        self.num_q_networks = num_q_networks
        self.uncertainty_threshold = uncertainty_threshold
        
        # Replace Q-networks with ensemble
        hidden_dims = kwargs.get('hidden_dims', [256, 256])
        self.q_ensemble = nn.ModuleList([
            SoftQNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
            for _ in range(num_q_networks)
        ])
        self.q_ensemble_target = copy.deepcopy(self.q_ensemble)
        
        # Initialize ensemble
        for q_net in self.q_ensemble:
            q_net.apply(lambda m: init_weights(m, gain=1))
        
        # Ensemble optimizers
        lr_q = kwargs.get('lr_q', 3e-4)
        self.q_ensemble_optimizers = [
            optim.Adam(q_net.parameters(), lr=lr_q)
            for q_net in self.q_ensemble
        ]
    
    def _update_q_functions(self, obs: torch.Tensor, actions: torch.Tensor,
                           rewards: torch.Tensor, next_obs: torch.Tensor,
                           dones: torch.Tensor) -> Tuple[float, float]:
        """Update ensemble Q-functions."""
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_obs)
            
            # Compute target Q-values for all ensemble members
            q_next_targets = []
            for i in range(self.num_q_networks):
                q_next = self.q_ensemble_target[i](next_obs, next_actions)
                q_next_targets.append(q_next)
            
            # Use mean of two random Q-functions (like standard SAC)
            idx1, idx2 = np.random.choice(self.num_q_networks, 2, replace=False)
            min_q_next_target = torch.min(q_next_targets[idx1], q_next_targets[idx2])
            
            next_v = min_q_next_target - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * next_v
        
        # Update all Q-functions in ensemble
        q_losses = []
        for i in range(self.num_q_networks):
            q_pred = self.q_ensemble[i](obs, actions)
            q_loss = F.mse_loss(q_pred, q_target)
            
            self.q_ensemble_optimizers[i].zero_grad()
            q_loss.backward()
            self.q_ensemble_optimizers[i].step()
            
            q_losses.append(q_loss.item())
        
        return np.mean(q_losses), np.std(q_losses)
    
    def _update_policy(self, obs: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Update policy using ensemble Q-functions."""
        actions, log_probs = self.policy.sample(obs)
        
        # Use mean of random subset for policy update
        subset_size = min(2, self.num_q_networks)
        subset_idx = np.random.choice(self.num_q_networks, subset_size, replace=False)
        
        q_values = []
        for idx in subset_idx:
            q_values.append(self.q_ensemble[idx](obs, actions))
        
        min_q_pi = torch.stack(q_values).min(dim=0)[0]
        
        policy_loss = (self.alpha * log_probs - min_q_pi).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), log_probs
    
    def get_uncertainty(self, obs: np.ndarray, action: np.ndarray) -> float:
        """Estimate uncertainty using ensemble disagreement."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            q_values = []
            for q_net in self.q_ensemble:
                q_val = q_net(obs_tensor, action_tensor)
                q_values.append(q_val.item())
            
            uncertainty = np.std(q_values)
        
        return uncertainty