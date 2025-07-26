"""
Neural network architectures for SAC.
Includes Q-networks, policy networks, and utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


def mlp(input_dim: int, hidden_dims: list, output_dim: int, 
        activation: nn.Module = nn.ReLU, output_activation: Optional[nn.Module] = None):
    """Create a multi-layer perceptron."""
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    
    return nn.Sequential(*layers)


class SoftQNetwork(nn.Module):
    """Q-network for SAC."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.q_net = mlp(obs_dim + act_dim, hidden_dims, 1)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q(s,a)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q_net(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for SAC.
    Outputs squashed actions using tanh.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.shared_net = mlp(obs_dim, hidden_dims[:-1], hidden_dims[-1])
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log_std of Gaussian distribution."""
        features = self.shared_net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and compute log probability.
        Uses reparameterization trick for gradient flow.
        """
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        
        # Sample from Gaussian
        normal = Normal(mean, std)
        x = normal.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(x)
        
        # Compute log probability with Jacobian correction
        log_prob = normal.log_prob(x)
        # Correct for tanh squashing: log(1 - tanh^2(x))
        log_prob -= torch.log((1 - action.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for environment interaction."""
        if deterministic:
            mean, _ = self.forward(obs)
            return torch.tanh(mean)
        else:
            action, _ = self.sample(obs)
            return action
    
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given action."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        
        # Inverse of tanh
        eps = EPSILON
        atanh_action = 0.5 * torch.log((1 + action + eps) / (1 - action + eps))
        
        # Compute log probability
        normal = Normal(mean, std)
        log_prob = normal.log_prob(atanh_action)
        # Correct for tanh squashing
        log_prob -= torch.log((1 - action.pow(2)) + eps)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return log_prob


class DeterministicPolicy(nn.Module):
    """
    Deterministic policy network for SAC.
    Useful for evaluation or as a baseline.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.policy_net = mlp(obs_dim, hidden_dims, act_dim, output_activation=nn.Tanh)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute deterministic action."""
        return self.policy_net(obs)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Get action (always deterministic for this policy)."""
        return self.forward(obs)


class EnsembleQNetwork(nn.Module):
    """
    Ensemble of Q-networks for uncertainty estimation.
    Can be used for exploration or robustness.
    """
    
    def __init__(self, obs_dim: int, act_dim: int, num_networks: int = 5, 
                 hidden_dims: list = [256, 256]):
        super().__init__()
        self.num_networks = num_networks
        self.q_networks = nn.ModuleList([
            SoftQNetwork(obs_dim, act_dim, hidden_dims) 
            for _ in range(num_networks)
        ])
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor, 
                return_all: bool = False) -> torch.Tensor:
        """
        Compute Q-values from ensemble.
        
        Args:
            obs: Observations
            action: Actions
            return_all: If True, return all Q-values; else return mean
        """
        q_values = torch.stack([
            q_net(obs, action) for q_net in self.q_networks
        ], dim=0)
        
        if return_all:
            return q_values
        else:
            return q_values.mean(dim=0)
    
    def uncertainty(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty (std) of Q-value predictions."""
        q_values = self.forward(obs, action, return_all=True)
        return q_values.std(dim=0)


class CNNGaussianPolicy(nn.Module):
    """
    CNN-based Gaussian policy for image observations.
    """
    
    def __init__(self, obs_shape: tuple, act_dim: int, hidden_dim: int = 512):
        super().__init__()
        c, h, w = obs_shape
        
        # CNN encoder (similar to Nature DQN)
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output dimension
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1
        
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        linear_input_size = convh * convw * 64
        
        # Policy heads
        self.fc = nn.Linear(linear_input_size, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log_std from image observations."""
        cnn_features = self.cnn(obs)
        features = F.relu(self.fc(cnn_features))
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from CNN policy."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        
        log_prob = normal.log_prob(x)
        log_prob -= torch.log((1 - action.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for environment interaction."""
        if deterministic:
            mean, _ = self.forward(obs)
            return torch.tanh(mean)
        else:
            action, _ = self.sample(obs)
            return action


def init_weights(m: nn.Module, gain: float = 1.0):
    """Initialize network weights using orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)