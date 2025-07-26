"""
Training script for SAC on robotics environments.
Supports various continuous control tasks.
"""

import argparse
import os
import time
import numpy as np
import torch
import gymnasium as gym
from collections import deque
import yaml
from typing import Optional, Dict

from sac_agent import SACAgent, SACAgentWithEnsemble
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def make_env(env_id: str, seed: Optional[int] = None):
    """Create and configure environment."""
    env = gym.make(env_id)
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    return env


def evaluate_policy(agent: SACAgent, env: gym.Env, 
                   n_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
    """
    Evaluate agent's policy.
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action = agent.act(obs, deterministic=deterministic)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths)
    }


def train_sac(
    # Environment
    env_id: str = "Ant-v4",
    # SAC parameters
    hidden_dims: list = [256, 256],
    lr_q: float = 3e-4,
    lr_policy: float = 3e-4,
    lr_alpha: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    auto_alpha: bool = True,
    target_entropy: Optional[float] = None,
    # Training parameters
    total_timesteps: int = 1_000_000,
    batch_size: int = 256,
    learning_starts: int = 10_000,
    update_frequency: int = 1,
    gradient_steps: int = 1,
    # Buffer parameters
    buffer_size: int = 1_000_000,
    prioritized_replay: bool = False,
    # Evaluation parameters
    eval_frequency: int = 10_000,
    n_eval_episodes: int = 10,
    # Other parameters
    save_frequency: int = 50_000,
    log_frequency: int = 1000,
    save_dir: str = "models",
    seed: int = 42,
    device: str = "auto",
    wandb_project: Optional[str] = None,
    config_path: Optional[str] = None
):
    """
    Train SAC agent on continuous control task.
    """
    # Load config if provided
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Override parameters with config values
        locals().update(config)
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Initialize wandb if requested
    if wandb_project:
        import wandb
        wandb.init(
            project=wandb_project,
            config={
                "env_id": env_id,
                "algorithm": "SAC",
                "total_timesteps": total_timesteps,
                "batch_size": batch_size,
                "lr_q": lr_q,
                "lr_policy": lr_policy,
                "gamma": gamma,
                "tau": tau
            }
        )
    
    # Create environment
    env = make_env(env_id, seed)
    eval_env = make_env(env_id, seed + 1000)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Environment: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create SAC agent
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=hidden_dims,
        lr_q=lr_q,
        lr_policy=lr_policy,
        lr_alpha=lr_alpha if auto_alpha else 0,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        target_entropy=target_entropy,
        batch_size=batch_size,
        device=device
    )
    
    # Create replay buffer
    if prioritized_replay:
        buffer = PrioritizedReplayBuffer(
            capacity=buffer_size,
            obs_shape=env.observation_space.shape,
            act_shape=env.action_space.shape,
            device=device
        )
    else:
        buffer = ReplayBuffer(
            capacity=buffer_size,
            obs_shape=env.observation_space.shape,
            act_shape=env.action_space.shape,
            device=device
        )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    training_metrics = deque(maxlen=100)
    
    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    start_time = time.time()
    
    print("Starting training...")
    
    for timestep in range(total_timesteps):
        # Select action
        if timestep < learning_starts:
            # Random action during warm-up
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Store transition
        buffer.add(obs, action, reward, next_obs, done or truncated)
        
        # Update counters
        episode_reward += reward
        episode_length += 1
        obs = next_obs
        
        # Handle episode termination
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1
            
            # Reset episode
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Log episode metrics
            if wandb_project and len(episode_rewards) > 0:
                wandb.log({
                    "episode_reward": episode_rewards[-1],
                    "episode_length": episode_lengths[-1],
                    "timestep": timestep
                })
        
        # Train agent
        if timestep >= learning_starts and timestep % update_frequency == 0:
            for _ in range(gradient_steps):
                batch = buffer.sample(batch_size)
                metrics = agent.update(batch)
                training_metrics.append(metrics)
        
        # Evaluate
        if timestep % eval_frequency == 0 and timestep > 0:
            eval_metrics = evaluate_policy(
                agent, eval_env, n_eval_episodes, deterministic=True
            )
            
            print(f"\nTimestep: {timestep:,}")
            print(f"Episode: {episode_count}")
            print(f"Eval reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
            
            if len(episode_rewards) > 0:
                print(f"Train reward (100 ep): {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
            
            if len(training_metrics) > 0:
                recent_metrics = list(training_metrics)[-100:]
                print(f"Q1 loss: {np.mean([m['q1_loss'] for m in recent_metrics]):.4f}")
                print(f"Q2 loss: {np.mean([m['q2_loss'] for m in recent_metrics]):.4f}")
                print(f"Policy loss: {np.mean([m['policy_loss'] for m in recent_metrics]):.4f}")
                print(f"Alpha: {recent_metrics[-1]['alpha']:.4f}")
                print(f"Entropy: {recent_metrics[-1]['entropy']:.4f}")
            
            # Time estimate
            elapsed = time.time() - start_time
            fps = timestep / elapsed
            eta = (total_timesteps - timestep) / fps / 3600
            print(f"FPS: {fps:.0f}, ETA: {eta:.1f} hours")
            
            if wandb_project:
                wandb.log({
                    "eval_reward_mean": eval_metrics['mean_reward'],
                    "eval_reward_std": eval_metrics['std_reward'],
                    "timestep": timestep
                })
        
        # Log training metrics
        if timestep % log_frequency == 0 and len(training_metrics) > 0:
            recent_metrics = list(training_metrics)[-100:]
            log_dict = {
                'q1_loss': np.mean([m['q1_loss'] for m in recent_metrics]),
                'q2_loss': np.mean([m['q2_loss'] for m in recent_metrics]),
                'policy_loss': np.mean([m['policy_loss'] for m in recent_metrics]),
                'alpha_loss': np.mean([m['alpha_loss'] for m in recent_metrics]),
                'alpha': recent_metrics[-1]['alpha'],
                'entropy': recent_metrics[-1]['entropy'],
                'timestep': timestep
            }
            
            if len(episode_rewards) > 0:
                log_dict['train_reward_mean'] = np.mean(episode_rewards)
                log_dict['train_reward_std'] = np.std(episode_rewards)
                log_dict['train_length_mean'] = np.mean(episode_lengths)
            
            if wandb_project:
                wandb.log(log_dict)
        
        # Save model
        if timestep % save_frequency == 0 and timestep > 0:
            save_path = os.path.join(save_dir, f"sac_{env_id}_{timestep}.pt")
            agent.save(save_path)
            print(f"\nModel saved to {save_path}")
    
    # Final save
    final_path = os.path.join(save_dir, f"sac_{env_id}_final.pt")
    agent.save(final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")
    
    # Final evaluation
    final_eval = evaluate_policy(agent, eval_env, n_eval_episodes=30, deterministic=True)
    print(f"\nFinal evaluation: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    if wandb_project:
        wandb.finish()
    
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on continuous control tasks")
    
    # Environment
    parser.add_argument("--env", type=str, default="Ant-v4",
                        help="Gymnasium environment ID")
    
    # SAC parameters
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256],
                        help="Hidden layer dimensions")
    parser.add_argument("--lr-q", type=float, default=3e-4,
                        help="Q-function learning rate")
    parser.add_argument("--lr-policy", type=float, default=3e-4,
                        help="Policy learning rate")
    parser.add_argument("--lr-alpha", type=float, default=3e-4,
                        help="Temperature learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Target network update rate")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Initial temperature")
    parser.add_argument("--no-auto-alpha", action="store_true",
                        help="Disable automatic temperature tuning")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--learning-starts", type=int, default=10_000,
                        help="Timesteps before training starts")
    parser.add_argument("--update-frequency", type=int, default=1,
                        help="Update frequency")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Gradient steps per update")
    
    # Buffer parameters
    parser.add_argument("--buffer-size", type=int, default=1_000_000,
                        help="Replay buffer size")
    parser.add_argument("--prioritized-replay", action="store_true",
                        help="Use prioritized experience replay")
    
    # Evaluation parameters
    parser.add_argument("--eval-frequency", type=int, default=10_000,
                        help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    
    # Other parameters
    parser.add_argument("--save-frequency", type=int, default=50_000,
                        help="Model save frequency")
    parser.add_argument("--log-frequency", type=int, default=1000,
                        help="Logging frequency")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Weights & Biases project name")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    
    args = parser.parse_args()
    
    # Train
    train_sac(
        env_id=args.env,
        hidden_dims=args.hidden_dims,
        lr_q=args.lr_q,
        lr_policy=args.lr_policy,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=not args.no_auto_alpha,
        total_timesteps=args.total_timesteps,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        update_frequency=args.update_frequency,
        gradient_steps=args.gradient_steps,
        buffer_size=args.buffer_size,
        prioritized_replay=args.prioritized_replay,
        eval_frequency=args.eval_frequency,
        n_eval_episodes=args.n_eval_episodes,
        save_frequency=args.save_frequency,
        log_frequency=args.log_frequency,
        save_dir=args.save_dir,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        config_path=args.config
    )