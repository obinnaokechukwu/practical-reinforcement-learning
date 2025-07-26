"""
Experience Replay Buffer Implementations

This module contains different replay buffer strategies for DQN.
"""

import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """
    Basic experience replay buffer with uniform sampling.
    
    Stores transitions and samples them uniformly at random,
    breaking temporal correlations in the training data.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions uniformly at random.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} elements, "
                           f"cannot sample {batch_size}")
        
        batch = random.sample(self.buffer, batch_size)
        
        # Organize batch for efficient training
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples transitions with probability proportional to their TD error,
    focusing learning on surprising or difficult experiences.
    """
    
    def __init__(self, capacity=10000, alpha=0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, td_error=None):
        """
        Store transition with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            td_error: TD error for prioritization (None for new experiences)
        """
        transition = (state, action, reward, next_state, done)
        
        # Calculate priority
        if td_error is None:
            # New experiences get maximum priority
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            # Priority proportional to TD error magnitude
            priority = (abs(td_error) + 1e-6) ** self.alpha
        
        # Store with circular buffer logic
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample batch with importance sampling weights.
        
        Args:
            batch_size: Number of transitions to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full)
            
        Returns:
            Tuple of (batch, weights, indices)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} elements")
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        # w_i = (1/N * 1/P_i)^beta
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        # Extract batch
        batch = [self.buffer[idx] for idx in indices]
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        return (states, actions, rewards, next_states, dones), \
               torch.FloatTensor(weights), indices
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on new TD errors.
        
        Args:
            indices: Indices of transitions to update
            td_errors: New TD errors for these transitions
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples."""
        return len(self.buffer) >= batch_size


class NStepReplayBuffer:
    """
    N-step replay buffer for multi-step returns.
    
    Stores n-step trajectories and computes n-step returns,
    allowing faster propagation of rewards.
    """
    
    def __init__(self, capacity=10000, n_steps=3, gamma=0.99):
        """
        Initialize n-step replay buffer.
        
        Args:
            capacity: Maximum number of transitions
            n_steps: Number of steps for returns
            gamma: Discount factor
        """
        self.buffer = deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_steps)
        
    def push(self, state, action, reward, next_state, done):
        """Store transition, computing n-step returns when ready."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If episode ends, flush the buffer
        if done:
            while self.n_step_buffer:
                self._store_n_step_transition()
        # If buffer is full, store the n-step transition
        elif len(self.n_step_buffer) == self.n_steps:
            self._store_n_step_transition()
    
    def _store_n_step_transition(self):
        """Compute and store n-step transition."""
        if not self.n_step_buffer:
            return
            
        # Get first transition
        state, action, _, _, _ = self.n_step_buffer[0]
        
        # Compute n-step return
        n_step_return = 0
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * reward
            if done:
                # If episode ends, use this as final state
                _, _, _, next_state, _ = self.n_step_buffer[i]
                self.buffer.append((state, action, n_step_return, next_state, True))
                self.n_step_buffer.clear()
                return
        
        # Use the last state as next state
        _, _, _, next_state, done = self.n_step_buffer[-1]
        self.buffer.append((state, action, n_step_return, next_state, done))
        self.n_step_buffer.popleft()
    
    def sample(self, batch_size):
        """Sample batch of n-step transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer has only {len(self.buffer)} elements")
        
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


def test_replay_buffers():
    """Test different replay buffer implementations."""
    print("Testing Replay Buffers\n")
    
    # Test basic replay buffer
    print("1. Basic Replay Buffer:")
    buffer = ReplayBuffer(capacity=100)
    
    # Add some transitions
    for i in range(50):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = i % 10 == 9  # Episode ends every 10 steps
        buffer.push(state, action, reward, next_state, done)
    
    # Sample a batch
    if buffer.is_ready(32):
        states, actions, rewards, next_states, dones = buffer.sample(32)
        print(f"  Buffer size: {len(buffer)}")
        print(f"  Sampled batch shapes:")
        print(f"    States: {states.shape}")
        print(f"    Actions: {actions.shape}")
        print(f"    Rewards: {rewards.shape}")
    
    # Test prioritized replay buffer
    print("\n2. Prioritized Replay Buffer:")
    pri_buffer = PrioritizedReplayBuffer(capacity=100)
    
    # Add transitions with varying TD errors
    for i in range(50):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = i % 10 == 9
        td_error = np.random.exponential(1.0)  # Varying TD errors
        pri_buffer.push(state, action, reward, next_state, done, td_error)
    
    # Sample with importance weights
    if pri_buffer.is_ready(32):
        batch, weights, indices = pri_buffer.sample(32, beta=0.4)
        states, actions, rewards, next_states, dones = batch
        print(f"  Buffer size: {len(pri_buffer)}")
        print(f"  Importance weights: min={weights.min():.3f}, "
              f"max={weights.max():.3f}, mean={weights.mean():.3f}")
    
    # Test n-step buffer
    print("\n3. N-Step Replay Buffer:")
    n_buffer = NStepReplayBuffer(capacity=100, n_steps=3, gamma=0.99)
    
    # Add a full episode
    for i in range(20):
        state = np.random.randn(4)
        action = np.random.randint(2)
        reward = 1.0  # Constant reward for easy verification
        next_state = np.random.randn(4)
        done = i == 19
        n_buffer.push(state, action, reward, next_state, done)
    
    print(f"  Buffer size: {len(n_buffer)}")
    print(f"  N-steps: {n_buffer.n_steps}")
    print(f"  Expected n-step return: {sum(0.99**i for i in range(3)):.3f}")


if __name__ == "__main__":
    test_replay_buffers()