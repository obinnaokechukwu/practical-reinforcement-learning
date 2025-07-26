"""
Utility functions for PPO implementation.
Includes normalization, logging, and environment wrappers.
"""

import numpy as np
import torch
import gym
from gym import spaces
from collections import deque
import threading
from typing import Optional, Callable, List, Dict
import yaml
import os


class RunningMeanStd:
    """
    Tracks the running mean and standard deviation of a data stream.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.lock = threading.Lock()

    def update(self, x):
        """Update statistics with new data batch."""
        with self.lock:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from the moments of a batch."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        """Normalize the input using running statistics."""
        with self.lock:
            return (x - self.mean) / np.sqrt(self.var + 1e-8)
    
    def denormalize(self, x):
        """Denormalize the input."""
        with self.lock:
            return x * np.sqrt(self.var + 1e-8) + self.mean


class NormalizedEnv(gym.Wrapper):
    """
    Wrapper that normalizes observations and optionally rewards.
    """
    def __init__(self, env, obs_norm=True, reward_norm=True, 
                 clip_obs=10., clip_reward=10., gamma=0.99):
        super().__init__(env)
        self.obs_norm = obs_norm
        self.reward_norm = reward_norm
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        
        # Initialize normalizers
        if self.obs_norm:
            self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        if self.reward_norm:
            self.reward_rms = RunningMeanStd(shape=())
            self.returns = np.zeros(getattr(env, 'num_envs', 1))
    
    def step(self, action):
        """Step with normalization."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Normalize observation
        if self.obs_norm:
            obs = self._normalize_obs(obs)
        
        # Normalize reward
        if self.reward_norm:
            reward = self._normalize_reward(reward)
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """Reset with observation normalization."""
        obs, info = self.env.reset(**kwargs)
        
        if self.obs_norm:
            obs = self._normalize_obs(obs)
        
        if self.reward_norm:
            self.returns = np.zeros(getattr(self.env, 'num_envs', 1))
        
        return obs, info
    
    def _normalize_obs(self, obs):
        """Normalize observations."""
        self.obs_rms.update(obs)
        normalized_obs = self.obs_rms.normalize(obs)
        return np.clip(normalized_obs, -self.clip_obs, self.clip_obs)
    
    def _normalize_reward(self, reward):
        """Normalize rewards using running estimate of return."""
        # Update returns
        self.returns = self.returns * self.gamma + reward
        self.reward_rms.update(self.returns)
        
        # Normalize
        normalized_reward = reward / np.sqrt(self.reward_rms.var + 1e-8)
        return np.clip(normalized_reward, -self.clip_reward, self.clip_reward)


class VecEnv:
    """
    Base class for vectorized environments.
    Manages multiple environment instances for parallel data collection.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
    
    def reset(self):
        """Reset all environments."""
        raise NotImplementedError
    
    def step(self, actions):
        """Step all environments with given actions."""
        raise NotImplementedError
    
    def close(self):
        """Close all environments."""
        raise NotImplementedError


class DummyVecEnv(VecEnv):
    """
    Vectorized environment that runs environments sequentially.
    Simple but slower than SubprocVecEnv.
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)
        
        # Track episode statistics
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_lengths = [0] * self.num_envs
    
    def reset(self):
        obs_list = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            obs_list.append(obs)
            infos.append(info)
            self.episode_rewards[i] = 0.0
            self.episode_lengths[i] = 0
        
        return np.array(obs_list), infos
    
    def step(self, actions):
        obs_list, reward_list, done_list, truncated_list, info_list = [], [], [], [], []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, truncated, info = env.step(action)
            
            # Track episode statistics
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            # Handle episode termination
            if done or truncated:
                info['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.episode_lengths[i]
                }
                obs, _ = env.reset()
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
            
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (np.array(obs_list), np.array(reward_list), 
                np.array(done_list), np.array(truncated_list), info_list)
    
    def close(self):
        for env in self.envs:
            env.close()


class RewardLogger:
    """
    Logger for tracking training statistics.
    """
    def __init__(self, window_size=100):
        self.episode_rewards = deque(maxlen=window_size * 10)
        self.episode_lengths = deque(maxlen=window_size * 10)
        self.episode_count = 0
        self.total_steps = 0
        
        # PPO specific metrics
        self.policy_losses = deque(maxlen=window_size)
        self.value_losses = deque(maxlen=window_size)
        self.entropies = deque(maxlen=window_size)
        self.kl_divs = deque(maxlen=window_size)
        self.clip_fractions = deque(maxlen=window_size)
    
    def log_episode(self, reward, length):
        """Log episode statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_count += 1
        self.total_steps += length
    
    def log_training(self, policy_loss, value_loss, entropy, kl_div, clip_fraction):
        """Log training statistics."""
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.kl_divs.append(kl_div)
        self.clip_fractions.append(clip_fraction)
    
    def get_stats(self):
        """Get current statistics."""
        if len(self.episode_rewards) == 0:
            return {}
        
        stats = {
            'episode_reward_mean': np.mean(self.episode_rewards),
            'episode_reward_std': np.std(self.episode_rewards),
            'episode_reward_min': np.min(self.episode_rewards),
            'episode_reward_max': np.max(self.episode_rewards),
            'episode_length_mean': np.mean(self.episode_lengths),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }
        
        if len(self.policy_losses) > 0:
            stats.update({
                'policy_loss': np.mean(self.policy_losses),
                'value_loss': np.mean(self.value_losses),
                'entropy': np.mean(self.entropies),
                'kl_div': np.mean(self.kl_divs),
                'clip_fraction': np.mean(self.clip_fractions)
            })
        
        return stats
    
    def print_stats(self, step):
        """Print formatted statistics."""
        stats = self.get_stats()
        if not stats:
            return
        
        print(f"\n{'='*60}")
        print(f"Step: {step:,}")
        print(f"Episodes: {stats['episode_count']:,}")
        print(f"Total Steps: {stats['total_steps']:,}")
        print(f"Episode Reward: {stats['episode_reward_mean']:.2f} "
              f"± {stats['episode_reward_std']:.2f} "
              f"[{stats['episode_reward_min']:.2f}, {stats['episode_reward_max']:.2f}]")
        print(f"Episode Length: {stats['episode_length_mean']:.1f}")
        
        if 'policy_loss' in stats:
            print(f"Policy Loss: {stats['policy_loss']:.4f}")
            print(f"Value Loss: {stats['value_loss']:.4f}")
            print(f"Entropy: {stats['entropy']:.4f}")
            print(f"KL Divergence: {stats['kl_div']:.4f}")
            print(f"Clip Fraction: {stats['clip_fraction']:.3f}")
        print(f"{'='*60}\n")


def explained_variance(y_pred, y_true):
    """
    Compute fraction of variance that ypred explains about y_true.
    Returns 1 - Var[y_true - y_pred] / Var[y_true]
    """
    assert y_pred.shape == y_true.shape
    var_y = torch.var(y_true)
    return 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial learning rate
        
    Returns:
        Schedule function that takes progress (0 to 1) and returns lr
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def cosine_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    Cosine learning rate schedule.
    """
    def func(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        return final_value + (initial_value - final_value) * 0.5 * (1 + np.cos(np.pi * progress))
    return func


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, step, path):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        print(f"Checkpoint loaded from {path} (step {step})")
        return step
    else:
        print(f"No checkpoint found at {path}")
        return 0