"""
Policy and Value Network Architectures for REINFORCE

This module contains the neural network architectures used in policy gradient methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Neural network for policy function approximation."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through policy network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Categorical distribution over actions
        """
        logits = self.network(state)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """Neural network for value function approximation (baseline)."""
    
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through value network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Value estimates of shape (batch_size, 1)
        """
        return self.network(state)


class PolicyValueNetwork(nn.Module):
    """Combined policy and value network with shared backbone."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate output heads
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through combined network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            tuple: (policy_logits, value_estimates)
                - policy_logits: Action logits of shape (batch_size, action_dim)
                - value_estimates: Value estimates of shape (batch_size, 1)
        """
        # Shared feature extraction
        features = self.shared_layers(state)
        
        # Separate outputs
        policy_logits = self.policy_head(features)
        value_estimates = self.value_head(features)
        
        return policy_logits, value_estimates
    
    def get_action_and_value(self, state):
        """
        Get action distribution and value estimate for a single state.
        
        Args:
            state: State tensor of shape (state_dim,) or (1, state_dim)
            
        Returns:
            tuple: (action_dist, value)
                - action_dist: Categorical distribution over actions
                - value: Value estimate tensor
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        policy_logits, value = self.forward(state)
        action_dist = Categorical(logits=policy_logits)
        
        return action_dist, value


# Example usage
if __name__ == "__main__":
    # Test the networks
    state_dim = 8  # LunarLander state dimension
    action_dim = 4  # LunarLander action dimension
    batch_size = 32
    
    # Create test state batch
    states = torch.randn(batch_size, state_dim)
    
    print("Testing PolicyNetwork:")
    policy_net = PolicyNetwork(state_dim, action_dim)
    action_dist = policy_net(states)
    actions = action_dist.sample()
    log_probs = action_dist.log_prob(actions)
    print(f"  Action shape: {actions.shape}")
    print(f"  Log prob shape: {log_probs.shape}")
    
    print("\nTesting ValueNetwork:")
    value_net = ValueNetwork(state_dim)
    values = value_net(states)
    print(f"  Value shape: {values.shape}")
    
    print("\nTesting PolicyValueNetwork:")
    policy_value_net = PolicyValueNetwork(state_dim, action_dim)
    policy_logits, values = policy_value_net(states)
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Values shape: {values.shape}")
    
    # Test single state action selection
    single_state = torch.randn(state_dim)
    action_dist, value = policy_value_net.get_action_and_value(single_state)
    action = action_dist.sample()
    print(f"\nSingle state action: {action.item()}")
    print(f"Single state value: {value.item():.4f}")