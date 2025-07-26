"""
Neural network architectures for PPO.
Supports both continuous and discrete action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np


def init_weights(m):
    """Initialize network weights using orthogonal initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0)


class PPONetwork(nn.Module):
    """
    PPO network with shared encoder and separate policy/value heads.
    Supports continuous control with Gaussian policies.
    """
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), 
                 activation='tanh', log_std_init=-0.5):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Build shared encoder
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*layers)
        
        # Policy head (outputs mean of Gaussian)
        self.policy_mean = nn.Linear(prev_dim, act_dim)
        
        # Learnable log standard deviation
        self.policy_log_std = nn.Parameter(
            torch.ones(1, act_dim) * log_std_init
        )
        
        # Value head
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(init_weights)
        
        # Special initialization for policy output
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.constant_(self.policy_mean.bias, 0)
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass through the network.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            action: Sampled or deterministic action
            log_prob: Log probability of the action
            value: Estimated state value
            entropy: Entropy of the policy distribution
        """
        # Shared encoding
        features = self.shared_encoder(obs)
        
        # Policy outputs
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
        
        # Create distribution
        dist = Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        # Compute log probability and entropy
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        # Value estimate
        value = self.value_head(features)
        
        return action, log_prob, value, entropy
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate given actions under the current policy.
        Used during PPO updates.
        
        Args:
            obs: Observation tensor
            actions: Actions to evaluate
            
        Returns:
            log_prob: Log probability of the actions
            value: Estimated state values
            entropy: Entropy of the policy distribution
        """
        # Shared encoding
        features = self.shared_encoder(obs)
        
        # Policy outputs
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
        
        # Create distribution
        dist = Normal(action_mean, action_std)
        
        # Evaluate actions
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        # Value estimate
        value = self.value_head(features)
        
        return log_prob, value, entropy
    
    def get_value(self, obs):
        """Get value estimate for observations."""
        features = self.shared_encoder(obs)
        return self.value_head(features)


class DiscreteActionPPONetwork(nn.Module):
    """
    PPO network for discrete action spaces.
    Uses categorical distribution for action selection.
    """
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), 
                 activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Build shared encoder
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*layers)
        
        # Policy head (outputs logits)
        self.policy_head = nn.Linear(prev_dim, act_dim)
        
        # Value head
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(init_weights)
        
        # Special initialization for policy output
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0)
    
    def forward(self, obs, deterministic=False):
        """Forward pass for discrete actions."""
        # Shared encoding
        features = self.shared_encoder(obs)
        
        # Policy outputs
        action_logits = self.policy_head(features)
        
        # Create distribution
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        
        # Compute log probability and entropy
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        # Value estimate
        value = self.value_head(features)
        
        return action, log_prob, value, entropy
    
    def evaluate_actions(self, obs, actions):
        """Evaluate given actions for discrete action space."""
        # Shared encoding
        features = self.shared_encoder(obs)
        
        # Policy outputs
        action_logits = self.policy_head(features)
        
        # Create distribution
        dist = Categorical(logits=action_logits)
        
        # Evaluate actions
        log_prob = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        
        # Value estimate
        value = self.value_head(features)
        
        return log_prob, value, entropy
    
    def get_value(self, obs):
        """Get value estimate for observations."""
        features = self.shared_encoder(obs)
        return self.value_head(features)


class CNNPPONetwork(nn.Module):
    """
    CNN-based PPO network for image observations.
    Follows the architecture used in the original PPO paper.
    """
    def __init__(self, obs_shape, act_dim, hidden_dim=512):
        super().__init__()
        c, h, w = obs_shape
        
        # CNN encoder (same as Nature DQN)
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
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, hidden_dim),
            nn.ReLU()
        )
        
        # Policy and value heads
        self.policy_mean = nn.Linear(hidden_dim, act_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(1, act_dim))
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(init_weights)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.constant_(self.policy_mean.bias, 0)
    
    def forward(self, obs, deterministic=False):
        """Forward pass for CNN network."""
        # CNN encoding
        cnn_features = self.cnn(obs)
        features = self.fc(cnn_features)
        
        # Policy outputs
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
        
        # Create distribution
        dist = Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.sample()
        
        # Compute log probability and entropy
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        # Value estimate
        value = self.value_head(features)
        
        return action, log_prob, value, entropy
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions for CNN network."""
        # CNN encoding
        cnn_features = self.cnn(obs)
        features = self.fc(cnn_features)
        
        # Policy outputs
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
        
        # Create distribution
        dist = Normal(action_mean, action_std)
        
        # Evaluate actions
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        # Value estimate
        value = self.value_head(features)
        
        return log_prob, value, entropy
    
    def get_value(self, obs):
        """Get value estimate for CNN network."""
        cnn_features = self.cnn(obs)
        features = self.fc(cnn_features)
        return self.value_head(features)