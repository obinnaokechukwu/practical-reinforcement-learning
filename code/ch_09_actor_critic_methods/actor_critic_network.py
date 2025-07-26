"""
Actor-Critic Network Architectures

This module contains the neural network architectures for actor-critic methods
as demonstrated in Chapter 9.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCriticNetwork(nn.Module):
    """Shared network architecture for actor and critic."""
    
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for output layers
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
    
    def forward(self, state):
        """Forward pass returning both policy logits and value estimate."""
        shared_features = self.shared_layers(state)
        
        policy_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        
        return policy_logits, value
    
    def act(self, state):
        """Select action using current policy."""
        policy_logits, value = self.forward(state)
        
        # Create probability distribution
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        
        # Calculate log probability for the selected action
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob, value, dist.entropy()
    
    def evaluate(self, state):
        """Get value estimate without sampling action."""
        _, value = self.forward(state)
        return value


class LSTMActorCritic(nn.Module):
    """Actor-Critic with LSTM for partially observable environments."""
    
    def __init__(self, input_dim, action_dim, hidden_dim=256, lstm_dim=128):
        super(LSTMActorCritic, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm_dim = lstm_dim
        
        # Input processing
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # LSTM for memory
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)
        
        # Actor and Critic heads
        self.actor = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, state, hidden_state=None):
        """Forward pass with LSTM memory."""
        # Process input
        x = F.relu(self.input_layer(state))
        
        # Add sequence dimension if missing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        
        # Use last output
        features = lstm_out[:, -1, :]
        
        # Actor and critic outputs
        policy_logits = self.actor(features)
        value = self.critic(features)
        
        return policy_logits, value, new_hidden
    
    def act(self, state, hidden_state=None):
        """Select action with memory."""
        policy_logits, value, new_hidden = self.forward(state, hidden_state)
        
        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob, value, dist.entropy(), new_hidden


# Example usage and testing
if __name__ == "__main__":
    # Test basic ActorCriticNetwork
    print("Testing ActorCriticNetwork:")
    
    state_dim = 4  # CartPole
    action_dim = 2
    batch_size = 32
    
    net = ActorCriticNetwork(state_dim, action_dim)
    
    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    policy_logits, values = net(states)
    
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Values shape: {values.shape}")
    
    # Test action selection
    single_state = torch.randn(1, state_dim)
    action, log_prob, value, entropy = net.act(single_state)
    
    print(f"  Action: {action.item()}")
    print(f"  Log prob: {log_prob.item():.4f}")
    print(f"  Value: {value.item():.4f}")
    print(f"  Entropy: {entropy.item():.4f}")
    
    # Test LSTM version
    print("\nTesting LSTMActorCritic:")
    
    lstm_net = LSTMActorCritic(state_dim, action_dim)
    
    # Test with sequence
    seq_states = torch.randn(batch_size, 10, state_dim)  # 10 timesteps
    policy_logits, values, hidden = lstm_net(seq_states)
    
    print(f"  LSTM Policy logits shape: {policy_logits.shape}")
    print(f"  LSTM Values shape: {values.shape}")
    
    # Test single step with memory
    single_state = torch.randn(1, state_dim)
    action, log_prob, value, entropy, new_hidden = lstm_net.act(single_state)
    
    print(f"  LSTM Action: {action.item()}")
    print(f"  LSTM Value: {value.item():.4f}")
    
    print("All tests passed!")