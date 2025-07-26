"""
Rollout buffer for PPO.
Handles data collection and advantage computation.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple


class RolloutBuffer:
    """
    Rollout buffer for collecting trajectories and computing advantages.
    Supports both continuous and discrete action spaces.
    """
    def __init__(self, buffer_size, obs_shape, act_shape, num_envs, 
                 device='cpu', gae_lambda=0.95, gamma=0.99):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.num_envs = num_envs
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        # Initialize buffers
        self.observations = np.zeros((buffer_size, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        # Computed after rollout
        self.advantages = None
        self.returns = None
        
        # Tracking
        self.ptr = 0
        self.path_start_idx = np.zeros(num_envs, dtype=np.int32)
        self.full = False
    
    def add(self, obs, action, reward, done, value, log_prob):
        """Add a transition to the buffer."""
        assert self.ptr < self.buffer_size, "Buffer is full!"
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.cpu().numpy().squeeze()
        self.log_probs[self.ptr] = log_prob.cpu().numpy().squeeze()
        
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
    
    def compute_advantages_and_returns(self, last_values):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        Also computes the returns for value function training.
        
        Args:
            last_values: Value estimates for the last states in the rollout
        """
        # Initialize advantages
        self.advantages = np.zeros_like(self.rewards)
        
        # Convert last values to numpy
        last_values = last_values.cpu().numpy().squeeze()
        
        # Compute GAE for each environment
        for env_idx in range(self.num_envs):
            last_gae_lam = 0
            
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - self.dones[step, env_idx]
                    next_value = last_values[env_idx]
                else:
                    next_non_terminal = 1.0 - self.dones[step, env_idx]
                    next_value = self.values[step + 1, env_idx]
                
                # TD error
                delta = (self.rewards[step, env_idx] + 
                        self.gamma * next_value * next_non_terminal - 
                        self.values[step, env_idx])
                
                # GAE
                self.advantages[step, env_idx] = (
                    delta + self.gamma * self.gae_lambda * 
                    next_non_terminal * last_gae_lam
                )
                last_gae_lam = self.advantages[step, env_idx]
        
        # Compute returns (advantages + values)
        self.returns = self.advantages + self.values
    
    def get_samples(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Get all samples from the buffer for training.
        
        Args:
            batch_size: If specified, yield batches of this size
            
        Returns:
            Dictionary containing all buffer data
        """
        assert self.full or self.ptr > 0, "Buffer is empty!"
        assert self.advantages is not None, "Call compute_advantages_and_returns first!"
        
        # Get the actual size of collected data
        buffer_size = self.buffer_size if self.full else self.ptr
        
        # Flatten the batch dimension
        batch_size_total = buffer_size * self.num_envs
        
        # Create flattened views
        obs_flat = self.observations[:buffer_size].reshape(batch_size_total, *self.obs_shape)
        actions_flat = self.actions[:buffer_size].reshape(batch_size_total, *self.act_shape)
        log_probs_flat = self.log_probs[:buffer_size].reshape(batch_size_total, 1)
        advantages_flat = self.advantages[:buffer_size].reshape(batch_size_total, 1)
        returns_flat = self.returns[:buffer_size].reshape(batch_size_total, 1)
        values_flat = self.values[:buffer_size].reshape(batch_size_total, 1)
        
        # Convert to tensors
        data = {
            'observations': torch.FloatTensor(obs_flat).to(self.device),
            'actions': torch.FloatTensor(actions_flat).to(self.device),
            'old_log_probs': torch.FloatTensor(log_probs_flat).to(self.device),
            'advantages': torch.FloatTensor(advantages_flat).to(self.device),
            'returns': torch.FloatTensor(returns_flat).to(self.device),
            'old_values': torch.FloatTensor(values_flat).to(self.device)
        }
        
        if batch_size is None:
            yield data
        else:
            # Generate random batches
            indices = np.arange(batch_size_total)
            np.random.shuffle(indices)
            
            start_idx = 0
            while start_idx < batch_size_total:
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_data = {
                    key: value[batch_indices] for key, value in data.items()
                }
                yield batch_data
                start_idx += batch_size
    
    def reset(self):
        """Reset the buffer."""
        self.ptr = 0
        self.full = False
        self.advantages = None
        self.returns = None
        self.path_start_idx.fill(0)


class DictRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer for dictionary observations (e.g., goal-conditioned tasks).
    """
    def __init__(self, buffer_size, obs_space, act_shape, num_envs, 
                 device='cpu', gae_lambda=0.95, gamma=0.99):
        # Don't call parent __init__ since we need different observation handling
        self.buffer_size = buffer_size
        self.obs_space = obs_space
        self.act_shape = act_shape
        self.num_envs = num_envs
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        
        # Initialize observation buffers for each key
        self.observations = {}
        for key, space in obs_space.spaces.items():
            self.observations[key] = np.zeros(
                (buffer_size, num_envs, *space.shape), 
                dtype=np.float32
            )
        
        # Initialize other buffers
        self.actions = np.zeros((buffer_size, num_envs, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        # Computed after rollout
        self.advantages = None
        self.returns = None
        
        # Tracking
        self.ptr = 0
        self.full = False
    
    def add(self, obs, action, reward, done, value, log_prob):
        """Add a transition to the buffer with dict observations."""
        assert self.ptr < self.buffer_size, "Buffer is full!"
        
        # Store each observation component
        for key in self.observations.keys():
            self.observations[key][self.ptr] = obs[key]
        
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value.cpu().numpy().squeeze()
        self.log_probs[self.ptr] = log_prob.cpu().numpy().squeeze()
        
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
    
    def get_samples(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Get samples with dictionary observations."""
        assert self.full or self.ptr > 0, "Buffer is empty!"
        assert self.advantages is not None, "Call compute_advantages_and_returns first!"
        
        # Get the actual size of collected data
        buffer_size = self.buffer_size if self.full else self.ptr
        batch_size_total = buffer_size * self.num_envs
        
        # Flatten observations
        obs_flat = {}
        for key, obs_array in self.observations.items():
            obs_shape = obs_array.shape[2:]  # Remove buffer_size and num_envs dims
            obs_flat[key] = obs_array[:buffer_size].reshape(batch_size_total, *obs_shape)
        
        # Flatten other data
        actions_flat = self.actions[:buffer_size].reshape(batch_size_total, *self.act_shape)
        log_probs_flat = self.log_probs[:buffer_size].reshape(batch_size_total, 1)
        advantages_flat = self.advantages[:buffer_size].reshape(batch_size_total, 1)
        returns_flat = self.returns[:buffer_size].reshape(batch_size_total, 1)
        values_flat = self.values[:buffer_size].reshape(batch_size_total, 1)
        
        # Convert to tensors
        obs_tensors = {
            key: torch.FloatTensor(obs).to(self.device) 
            for key, obs in obs_flat.items()
        }
        
        data = {
            'observations': obs_tensors,
            'actions': torch.FloatTensor(actions_flat).to(self.device),
            'old_log_probs': torch.FloatTensor(log_probs_flat).to(self.device),
            'advantages': torch.FloatTensor(advantages_flat).to(self.device),
            'returns': torch.FloatTensor(returns_flat).to(self.device),
            'old_values': torch.FloatTensor(values_flat).to(self.device)
        }
        
        if batch_size is None:
            yield data
        else:
            # Generate random batches
            indices = np.arange(batch_size_total)
            np.random.shuffle(indices)
            
            start_idx = 0
            while start_idx < batch_size_total:
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Create batch data
                batch_obs = {
                    key: obs_tensor[batch_indices] 
                    for key, obs_tensor in obs_tensors.items()
                }
                
                batch_data = {
                    'observations': batch_obs,
                    'actions': data['actions'][batch_indices],
                    'old_log_probs': data['old_log_probs'][batch_indices],
                    'advantages': data['advantages'][batch_indices],
                    'returns': data['returns'][batch_indices],
                    'old_values': data['old_values'][batch_indices]
                }
                
                yield batch_data
                start_idx += batch_size