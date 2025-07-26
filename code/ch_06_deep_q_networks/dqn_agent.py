"""
Deep Q-Network Agent Implementation

Complete DQN agent with all key components integrated.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from dqn_network import DQNNetwork, DuelingDQN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    
    Combines all the key innovations that make DQN work:
    - Neural network function approximation
    - Experience replay for stability
    - Target network for stable targets
    - Epsilon-greedy exploration
    """
    
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 buffer_size=10000,
                 batch_size=64,
                 target_update=10,
                 use_prioritized_replay=False,
                 use_dueling=False):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon value
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update: Target network update frequency
            use_prioritized_replay: Whether to use prioritized replay
            use_dueling: Whether to use dueling architecture
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Create networks
        if use_dueling:
            self.q_network = DuelingDQN(state_dim, action_dim)
            self.target_network = DuelingDQN(state_dim, action_dim)
        else:
            self.q_network = DQNNetwork(state_dim, action_dim)
            self.target_network = DQNNetwork(state_dim, action_dim)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = ReplayBuffer(buffer_size)
        
        # Metrics
        self.losses = []
        self.q_values = []
        self.td_errors = []
        self.update_count = 0
        
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Exploration during training
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        # Exploitation: choose best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            # Track Q-values for analysis
            self.q_values.append(q_values.mean().item())
            
            return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay buffer.
        
        For prioritized replay, calculates TD error for initial priority.
        """
        if self.use_prioritized_replay:
            # Calculate TD error for prioritization
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                
                current_q = self.q_network(state_tensor)[0, action]
                next_q = self.target_network(next_state_tensor).max(1)[0]
                target_q = reward + (1 - done) * self.gamma * next_q
                td_error = abs(target_q - current_q).item()
            
            self.memory.push(state, action, reward, next_state, done, td_error)
        else:
            self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training update.
        
        Samples from replay buffer and updates Q-network.
        """
        if not self.memory.is_ready(self.batch_size):
            return
        
        # Sample batch
        if self.use_prioritized_replay:
            batch_data, weights, indices = self.memory.sample(self.batch_size)
            if batch_data is None:
                return
            states, actions, rewards, next_states, dones = batch_data
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size)
            indices = None
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            # Double DQN: use online network to select actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate actions
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss with importance sampling weights
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors_np)
        
        # Record metrics
        self.losses.append(loss.item())
        self.td_errors.extend(td_errors.detach().cpu().numpy().flatten())
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from online to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def soft_update_target_network(self, tau=0.001):
        """
        Soft update of target network parameters.
        
        θ_target = τ*θ_online + (1-τ)*θ_target
        """
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_diagnostics(self):
        """
        Get diagnostic information about agent's learning.
        
        Returns:
            Dictionary of diagnostic metrics
        """
        diagnostics = {}
        
        if self.losses:
            diagnostics['loss_mean'] = np.mean(self.losses[-100:])
            diagnostics['loss_std'] = np.std(self.losses[-100:])
        
        if self.q_values:
            diagnostics['q_value_mean'] = np.mean(self.q_values[-100:])
            diagnostics['q_value_std'] = np.std(self.q_values[-100:])
            diagnostics['q_value_max'] = np.max(self.q_values[-100:])
        
        if self.td_errors:
            diagnostics['td_error_mean'] = np.mean(np.abs(self.td_errors[-1000:]))
            diagnostics['td_error_max'] = np.max(np.abs(self.td_errors[-1000:]))
        
        diagnostics['epsilon'] = self.epsilon
        diagnostics['buffer_size'] = len(self.memory)
        diagnostics['updates'] = self.update_count
        
        return diagnostics
    
    def save(self, filepath):
        """Save agent state."""
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, filepath)
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent that reduces overestimation bias.
    
    Uses online network to select actions and target network to evaluate them.
    """
    
    def train_step(self):
        """
        Training step with Double DQN update rule.
        
        Key difference: action selection and evaluation are decoupled.
        """
        if not self.memory.is_ready(self.batch_size):
            return
        
        # Sample batch
        if self.use_prioritized_replay:
            batch_data, weights, indices = self.memory.sample(self.batch_size)
            if batch_data is None:
                return
            states, actions, rewards, next_states, dones = batch_data
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size)
            indices = None
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN target
        with torch.no_grad():
            # Use online network to select best actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate those actions
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate loss
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities
        if self.use_prioritized_replay and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors_np)
        
        # Record metrics
        self.losses.append(loss.item())
        self.td_errors.extend(td_errors.detach().cpu().numpy().flatten())
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.update_target_network()


if __name__ == "__main__":
    # Test the agent
    print("Testing DQN Agent\n")
    
    # Create a simple agent
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    # Simulate some transitions
    for i in range(100):
        state = np.random.randn(4)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.random() < 0.1
        
        agent.store_transition(state, action, reward, next_state, done)
        
        # Train every 4 steps
        if i % 4 == 0:
            agent.train_step()
    
    # Get diagnostics
    print("Agent Diagnostics:")
    diagnostics = agent.get_diagnostics()
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Test Double DQN
    print("\n\nTesting Double DQN Agent")
    double_agent = DoubleDQNAgent(state_dim=4, action_dim=2, use_prioritized_replay=True)
    
    # Simulate training
    for i in range(100):
        state = np.random.randn(4)
        action = double_agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.random() < 0.1
        
        double_agent.store_transition(state, action, reward, next_state, done)
        
        if i % 4 == 0:
            double_agent.train_step()
    
    print("\nDouble DQN Diagnostics:")
    diagnostics = double_agent.get_diagnostics()
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")