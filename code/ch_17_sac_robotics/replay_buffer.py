"""
Replay buffer implementation for SAC.
Includes prioritized replay and efficient sampling.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import random


class ReplayBuffer:
    """
    Standard replay buffer for off-policy algorithms.
    """
    
    def __init__(self, capacity: int, obs_shape: tuple, act_shape: tuple, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
        
        return batch
    
    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer for SAC.
    Prioritizes transitions with high TD error.
    """
    
    def __init__(self, capacity: int, obs_shape: tuple, act_shape: tuple, 
                 alpha: float = 0.6, beta: float = 0.4, device: str = 'cpu'):
        super().__init__(capacity, obs_shape, act_shape, device)
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.epsilon = 1e-6
        
        # Priority tree
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool):
        """Add transition with maximum priority."""
        # Set priority to max for new transitions
        self.priorities[self.ptr] = self.max_priority
        
        # Add transition
        super().add(obs, action, reward, next_obs, done)
    
    def sample(self, batch_size: int, beta: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Sample batch with prioritization."""
        if beta is None:
            beta = self.beta
        
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        
        # Get batch
        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
            'weights': weights,
            'indices': indices
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on TD errors."""
        priorities = priorities + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def anneal_beta(self, progress: float):
        """Anneal beta from initial value to 1.0."""
        self.beta = self.beta + (1.0 - self.beta) * progress


class HindsightReplayBuffer(ReplayBuffer):
    """
    Hindsight Experience Replay buffer for goal-conditioned SAC.
    """
    
    def __init__(self, capacity: int, obs_shape: tuple, act_shape: tuple, 
                 goal_shape: tuple, n_sampled_goal: int = 4, device: str = 'cpu'):
        # Store observations and goals separately
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.n_sampled_goal = n_sampled_goal
        
        # Pre-allocate memory
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.achieved_goals = np.zeros((capacity, *goal_shape), dtype=np.float32)
        self.desired_goals = np.zeros((capacity, *goal_shape), dtype=np.float32)
        
        # Episode information for HER
        self.episode_starts = []
        self.current_episode_start = 0
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool, achieved_goal: np.ndarray, 
            desired_goal: np.ndarray):
        """Add a transition with goal information."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.achieved_goals[self.ptr] = achieved_goal
        self.desired_goals[self.ptr] = desired_goal
        
        if done:
            # Store episode boundaries
            self.episode_starts.append((self.current_episode_start, self.ptr))
            self.current_episode_start = (self.ptr + 1) % self.capacity
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, reward_func=None) -> Dict[str, torch.Tensor]:
        """
        Sample batch with HER.
        
        Args:
            batch_size: Number of transitions to sample
            reward_func: Function to compute reward given achieved and desired goals
        """
        # Regular sampling
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Prepare batch
        observations = self.observations[indices].copy()
        actions = self.actions[indices].copy()
        next_observations = self.next_observations[indices].copy()
        dones = self.dones[indices].copy()
        
        # Mix of actual goals and hindsight goals
        achieved_goals = self.achieved_goals[indices].copy()
        desired_goals = self.desired_goals[indices].copy()
        
        # Apply HER to some transitions
        her_indices = np.where(np.random.uniform(size=batch_size) < 0.8)[0]
        
        for idx in her_indices:
            # Find the episode this transition belongs to
            episode_start = 0
            episode_end = self.size - 1
            
            for start, end in self.episode_starts:
                if start <= indices[idx] <= end:
                    episode_start = start
                    episode_end = end
                    break
            
            # Sample future achieved goal from the same episode
            if indices[idx] < episode_end:
                future_idx = np.random.randint(indices[idx] + 1, episode_end + 1)
                desired_goals[idx] = self.achieved_goals[future_idx].copy()
        
        # Recompute rewards if function provided
        if reward_func is not None:
            rewards = np.array([
                reward_func(achieved_goals[i], desired_goals[i]) 
                for i in range(batch_size)
            ]).reshape(-1, 1)
        else:
            rewards = self.rewards[indices].copy()
        
        # Convert to tensors
        batch = {
            'observations': torch.FloatTensor(observations).to(self.device),
            'actions': torch.FloatTensor(actions).to(self.device),
            'rewards': torch.FloatTensor(rewards).to(self.device),
            'next_observations': torch.FloatTensor(next_observations).to(self.device),
            'dones': torch.FloatTensor(dones).to(self.device),
            'achieved_goals': torch.FloatTensor(achieved_goals).to(self.device),
            'desired_goals': torch.FloatTensor(desired_goals).to(self.device)
        }
        
        return batch


class SegmentReplayBuffer:
    """
    Replay buffer that stores entire trajectory segments.
    Useful for algorithms that need temporal context.
    """
    
    def __init__(self, capacity: int, segment_length: int, 
                 obs_shape: tuple, act_shape: tuple, device: str = 'cpu'):
        self.capacity = capacity // segment_length
        self.segment_length = segment_length
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory for segments
        self.observations = np.zeros((self.capacity, segment_length + 1, *obs_shape), 
                                    dtype=np.float32)
        self.actions = np.zeros((self.capacity, segment_length, *act_shape), 
                               dtype=np.float32)
        self.rewards = np.zeros((self.capacity, segment_length, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, segment_length, 1), dtype=np.float32)
        
        # Temporary storage for current segment
        self.current_segment_obs = []
        self.current_segment_act = []
        self.current_segment_rew = []
        self.current_segment_done = []
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool):
        """Add transition to current segment."""
        if len(self.current_segment_obs) == 0:
            self.current_segment_obs.append(obs)
        
        self.current_segment_obs.append(next_obs)
        self.current_segment_act.append(action)
        self.current_segment_rew.append(reward)
        self.current_segment_done.append(done)
        
        # Check if segment is complete
        if len(self.current_segment_act) >= self.segment_length or done:
            self._store_segment()
    
    def _store_segment(self):
        """Store the current segment in the buffer."""
        if len(self.current_segment_act) == 0:
            return
        
        # Pad if necessary
        while len(self.current_segment_act) < self.segment_length:
            self.current_segment_obs.append(self.current_segment_obs[-1])
            self.current_segment_act.append(np.zeros_like(self.current_segment_act[-1]))
            self.current_segment_rew.append(0.0)
            self.current_segment_done.append(1.0)
        
        # Store segment
        self.observations[self.ptr] = np.array(self.current_segment_obs[:self.segment_length + 1])
        self.actions[self.ptr] = np.array(self.current_segment_act[:self.segment_length])
        self.rewards[self.ptr] = np.array(self.current_segment_rew[:self.segment_length]).reshape(-1, 1)
        self.dones[self.ptr] = np.array(self.current_segment_done[:self.segment_length]).reshape(-1, 1)
        
        # Clear current segment
        self.current_segment_obs = []
        self.current_segment_act = []
        self.current_segment_rew = []
        self.current_segment_done = []
        
        # Update pointers
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of segments."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }
        
        return batch
    
    def __len__(self):
        return self.size