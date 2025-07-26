"""
Deep Q-Network Implementation

This module contains the neural network architecture for DQN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    
    Architecture:
    - Two hidden layers with ReLU activations
    - Linear output layer (no activation)
    - Xavier initialization for stable training
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """
        Initialize the Q-network.
        
        Args:
            input_dim: Size of state space
            output_dim: Number of actions
            hidden_dim: Size of hidden layers
        """
        super(DQNNetwork, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Apply Xavier initialization to all layers."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: State tensor of shape (batch_size, input_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, output_dim)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation on output
    
    def get_q_values(self, state):
        """
        Get Q-values for a single state (no batch dimension).
        
        Args:
            state: State tensor of shape (input_dim,)
            
        Returns:
            Q-values tensor of shape (output_dim,)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.forward(state_tensor).squeeze(0)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    
    This architecture often learns more robust value functions by
    explicitly modeling V(s) and A(s,a) separately.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize all weights using Xavier initialization."""
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        self.apply(init_layer)
    
    def forward(self, x):
        """
        Forward pass computing Q-values using dueling architecture.
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        """
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine streams using the dueling formula
        # Subtract mean advantage for stability
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration via parameter noise.
    
    Instead of epsilon-greedy exploration, this layer adds
    learnable noise to network parameters.
    """
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Factorized noise parameters
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features ** 0.5)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features ** 0.5)
    
    def reset_noise(self):
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    @staticmethod
    def _scale_noise(size):
        """Generate scaled noise."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()


def create_network(network_type='dqn', input_dim=4, output_dim=2, hidden_dim=128):
    """
    Factory function to create different network architectures.
    
    Args:
        network_type: 'dqn' or 'dueling'
        input_dim: State space dimension
        output_dim: Action space dimension
        hidden_dim: Hidden layer size
        
    Returns:
        Neural network instance
    """
    if network_type == 'dqn':
        return DQNNetwork(input_dim, output_dim, hidden_dim)
    elif network_type == 'dueling':
        return DuelingDQN(input_dim, output_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


if __name__ == "__main__":
    # Test the networks
    print("Testing DQN Network:")
    net = DQNNetwork(4, 2)
    test_input = torch.randn(32, 4)  # Batch of 32 states
    output = net(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample Q-values: {output[0]}")
    
    print("\n\nTesting Dueling DQN:")
    dueling_net = DuelingDQN(4, 2)
    output = dueling_net(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Sample Q-values: {output[0]}")