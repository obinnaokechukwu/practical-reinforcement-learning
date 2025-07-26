"""
Training script for PPO on continuous control tasks.
Supports various MuJoCo environments and custom configurations.
"""

import argparse
import os
import time
import numpy as np
import torch
import gymnasium as gym
from typing import Optional

from ppo_agent import PPOAgent
from buffer import RolloutBuffer
from utils import (
    DummyVecEnv, NormalizedEnv, RewardLogger, 
    linear_schedule, cosine_schedule, save_checkpoint, load_checkpoint
)


def make_env(env_id: str, seed: int, normalize: bool = True):
    """Create a single environment instance."""
    def _init():
        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        
        if normalize:
            env = NormalizedEnv(env, obs_norm=True, reward_norm=True)
        
        return env
    
    return _init


def train_ppo(
    env_id: str = "HalfCheetah-v4",
    # Training parameters
    total_timesteps: int = 1_000_000,
    num_envs: int = 8,
    num_steps: int = 2048,
    # PPO parameters
    learning_rate: float = 3e-4,
    clip_epsilon: float = 0.2,
    epochs: int = 10,
    mini_batch_size: int = 64,
    # GAE parameters
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    # Other parameters
    normalize: bool = True,
    lr_schedule: str = "constant",  # constant, linear, cosine
    save_freq: int = 100_000,
    eval_freq: int = 10_000,
    checkpoint_path: Optional[str] = None,
    seed: int = 42,
    device: str = "auto"
):
    """
    Train PPO on a continuous control task.
    
    Args:
        env_id: Gymnasium environment ID
        total_timesteps: Total number of environment steps
        num_envs: Number of parallel environments
        num_steps: Steps per environment before update
        learning_rate: Initial learning rate
        clip_epsilon: PPO clipping parameter
        epochs: Number of epochs per update
        mini_batch_size: Mini-batch size for updates
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        normalize: Whether to normalize observations and rewards
        lr_schedule: Learning rate schedule type
        save_freq: Frequency of model checkpoints
        eval_freq: Frequency of evaluation
        checkpoint_path: Path to load checkpoint from
        seed: Random seed
        device: Device to use (auto, cpu, cuda)
    """
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create environments
    env_fns = [make_env(env_id, seed + i, normalize) for i in range(num_envs)]
    envs = DummyVecEnv(env_fns)
    
    # Create evaluation environment
    eval_env = gym.make(env_id)
    eval_env.reset(seed=seed + 1000)
    
    # Get dimensions
    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    continuous = isinstance(envs.action_space, gym.spaces.Box)
    
    print(f"Environment: {env_id}")
    print(f"Observation space: {envs.observation_space}")
    print(f"Action space: {envs.action_space}")
    print(f"Number of environments: {num_envs}")
    
    # Create PPO agent
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        continuous=continuous,
        lr_policy=learning_rate,
        lr_value=learning_rate * 3,  # Higher LR for value function
        clip_epsilon=clip_epsilon,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device
    )
    
    # Set learning rate schedule
    if lr_schedule == "linear":
        agent.set_lr_scheduler(
            lr_scheduler_policy=linear_schedule(learning_rate),
            lr_scheduler_value=linear_schedule(learning_rate * 3)
        )
    elif lr_schedule == "cosine":
        agent.set_lr_scheduler(
            lr_scheduler_policy=cosine_schedule(learning_rate),
            lr_scheduler_value=cosine_schedule(learning_rate * 3)
        )
    
    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=num_steps,
        obs_shape=envs.observation_space.shape,
        act_shape=envs.action_space.shape,
        num_envs=num_envs,
        device=device,
        gae_lambda=gae_lambda,
        gamma=gamma
    )
    
    # Create logger
    logger = RewardLogger()
    
    # Load checkpoint if provided
    start_timestep = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        # Extract timestep from checkpoint name if possible
        try:
            start_timestep = int(checkpoint_path.split('_')[-1].split('.')[0])
        except:
            pass
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Initialize environments
    obs, _ = envs.reset()
    
    # Training loop
    num_updates = (total_timesteps - start_timestep) // (num_steps * num_envs)
    
    print(f"\nStarting training...")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Updates to perform: {num_updates:,}")
    
    start_time = time.time()
    
    for update in range(num_updates):
        # Collect rollout
        for step in range(num_steps):
            # Select action
            actions, log_probs, values = agent.act(obs, deterministic=False)
            
            # Step environment
            next_obs, rewards, dones, truncated, infos = envs.step(actions)
            
            # Store transition
            buffer.add(obs, actions, rewards, dones, values, log_probs)
            
            # Update observation
            obs = next_obs
            
            # Log episode statistics
            for info in infos:
                if 'episode' in info:
                    logger.log_episode(info['episode']['r'], info['episode']['l'])
        
        # Compute advantages and returns
        last_values = agent.evaluate_value(obs)
        buffer.compute_advantages_and_returns(last_values)
        
        # Update policy and value function
        current_timestep = start_timestep + (update + 1) * num_steps * num_envs
        lr_progress = 1.0 - (current_timestep / total_timesteps)
        
        update_stats = agent.update(buffer, lr_progress if lr_schedule != "constant" else None)
        
        # Log training statistics
        logger.log_training(
            update_stats['policy_loss'],
            update_stats['value_loss'],
            update_stats['entropy'],
            update_stats['kl_div'],
            update_stats['clip_fraction']
        )
        
        # Clear buffer
        buffer.reset()
        
        # Print statistics
        if (update + 1) % 10 == 0:
            logger.print_stats(current_timestep)
            
            # Timing information
            elapsed_time = time.time() - start_time
            fps = current_timestep / elapsed_time
            eta = (total_timesteps - current_timestep) / fps / 3600  # Hours
            
            print(f"FPS: {fps:.0f}")
            print(f"ETA: {eta:.1f} hours")
            print(f"Explained Variance: {update_stats['explained_variance']:.3f}")
            print(f"Epochs Used: {update_stats['epochs_used']}/{epochs}")
        
        # Evaluate
        if eval_freq > 0 and (current_timestep % eval_freq) < (num_steps * num_envs):
            eval_reward = evaluate_policy(agent, eval_env, n_episodes=10)
            print(f"\nEvaluation at {current_timestep:,} steps: {eval_reward:.2f}")
        
        # Save checkpoint
        if save_freq > 0 and (current_timestep % save_freq) < (num_steps * num_envs):
            checkpoint_name = f"ppo_{env_id}_{current_timestep}.pt"
            agent.save(checkpoint_name)
            print(f"Checkpoint saved: {checkpoint_name}")
    
    # Final save
    agent.save(f"ppo_{env_id}_final.pt")
    
    # Close environments
    envs.close()
    eval_env.close()
    
    print("\nTraining completed!")
    
    return agent


def evaluate_policy(agent: PPOAgent, env: gym.Env, n_episodes: int = 10) -> float:
    """
    Evaluate the policy on the environment.
    
    Args:
        agent: PPO agent
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        
    Returns:
        Mean episode reward
    """
    episode_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Select action deterministically
            action, _, _ = agent.act(np.array([obs]), deterministic=True)
            action = action[0]  # Remove batch dimension
            
            # Step environment
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on continuous control tasks")
    
    # Environment
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Gymnasium environment ID")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total number of environment steps")
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="Steps per environment before update")
    
    # PPO parameters
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clipping parameter")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per update")
    parser.add_argument("--mini-batch-size", type=int, default=64,
                        help="Mini-batch size for updates")
    
    # GAE parameters
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    
    # Other parameters
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable observation and reward normalization")
    parser.add_argument("--lr-schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Learning rate schedule")
    parser.add_argument("--save-freq", type=int, default=100_000,
                        help="Frequency of model checkpoints")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Frequency of evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Train
    train_ppo(
        env_id=args.env,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        epochs=args.epochs,
        mini_batch_size=args.mini_batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        normalize=not args.no_normalize,
        lr_schedule=args.lr_schedule,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        device=args.device
    )