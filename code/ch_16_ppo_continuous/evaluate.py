"""
Evaluation script for trained PPO agents.
Supports rendering and saving videos of agent performance.
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
from typing import Optional

from ppo_agent import PPOAgent
from utils import NormalizedEnv


def evaluate_agent(
    agent_path: str,
    env_id: str,
    n_episodes: int = 10,
    render: bool = False,
    record_video: bool = False,
    video_folder: str = "videos",
    deterministic: bool = True,
    normalize: bool = True,
    seed: Optional[int] = None
):
    """
    Evaluate a trained PPO agent.
    
    Args:
        agent_path: Path to saved agent checkpoint
        env_id: Gymnasium environment ID
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        record_video: Whether to record videos
        video_folder: Folder to save videos
        deterministic: Whether to use deterministic actions
        normalize: Whether to normalize observations
        seed: Random seed
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create environment
    if render:
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id)
    
    # Add video recording wrapper if requested
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env, 
            video_folder,
            episode_trigger=lambda x: True,  # Record all episodes
            name_prefix=f"ppo_{env_id}"
        )
    
    # Add normalization wrapper if needed
    if normalize:
        env = NormalizedEnv(env, obs_norm=True, reward_norm=False)
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    
    # Create agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        continuous=continuous,
        device=device
    )
    
    # Load trained model
    agent.load(agent_path)
    print(f"Loaded agent from: {agent_path}")
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nEvaluating for {n_episodes} episodes...")
    print("-" * 50)
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            # Select action
            action, _, _ = agent.act(np.array([obs]), deterministic=deterministic)
            action = action[0]  # Remove batch dimension
            
            # Step environment
            obs, reward, done, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Print summary statistics
    print("-" * 50)
    print(f"\nEvaluation Summary:")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f}")
    
    # Close environment
    env.close()
    
    return episode_rewards, episode_lengths


def visualize_value_function(
    agent_path: str,
    env_id: str,
    n_samples: int = 1000,
    seed: Optional[int] = None
):
    """
    Visualize the learned value function by sampling random states.
    
    Args:
        agent_path: Path to saved agent checkpoint
        env_id: Gymnasium environment ID
        n_samples: Number of states to sample
        seed: Random seed
    """
    import matplotlib.pyplot as plt
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create environment
    env = gym.make(env_id)
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    
    # Create and load agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        continuous=continuous,
        device=device
    )
    agent.load(agent_path)
    
    # Collect states and values
    states = []
    values = []
    actual_returns = []
    
    print(f"Collecting {n_samples} state samples...")
    
    while len(states) < n_samples:
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_states = [obs]
        episode_rewards = []
        
        # Collect episode
        while not (done or truncated) and len(states) + len(episode_states) < n_samples:
            action, _, _ = agent.act(np.array([obs]), deterministic=False)
            action = action[0]
            
            obs, reward, done, truncated, _ = env.step(action)
            
            episode_states.append(obs)
            episode_rewards.append(reward)
        
        # Compute actual returns for this episode
        episode_returns = []
        running_return = 0
        for reward in reversed(episode_rewards):
            running_return = reward + agent.gamma * running_return
            episode_returns.insert(0, running_return)
        
        # Add to collections
        n_to_add = min(len(episode_states) - 1, n_samples - len(states))
        states.extend(episode_states[:n_to_add])
        actual_returns.extend(episode_returns[:n_to_add])
    
    # Compute value estimates
    print("Computing value estimates...")
    for state in states:
        value = agent.evaluate_value(np.array([state]))
        values.append(value.item())
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_returns, values, alpha=0.5, s=10)
    plt.plot([min(actual_returns), max(actual_returns)], 
             [min(actual_returns), max(actual_returns)], 
             'r--', label='Perfect prediction')
    plt.xlabel('Actual Return')
    plt.ylabel('Value Estimate')
    plt.title(f'Value Function Accuracy - {env_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compute and display R²
    actual_returns_np = np.array(actual_returns)
    values_np = np.array(values)
    correlation = np.corrcoef(actual_returns_np, values_np)[0, 1]
    r_squared = correlation ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'value_function_{env_id}.png', dpi=150)
    plt.show()
    
    print(f"\nValue function R² score: {r_squared:.3f}")
    
    env.close()


def analyze_policy(
    agent_path: str,
    env_id: str,
    n_states: int = 100,
    seed: Optional[int] = None
):
    """
    Analyze the learned policy by examining action distributions.
    
    Args:
        agent_path: Path to saved agent checkpoint
        env_id: Gymnasium environment ID
        n_states: Number of states to analyze
        seed: Random seed
    """
    import matplotlib.pyplot as plt
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create environment
    env = gym.make(env_id)
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    continuous = isinstance(env.action_space, gym.spaces.Box)
    
    if not continuous:
        print("Policy analysis is currently only supported for continuous action spaces.")
        return
    
    # Create and load agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        continuous=continuous,
        device=device
    )
    agent.load(agent_path)
    
    # Collect states
    states = []
    print(f"Collecting {n_states} states...")
    
    while len(states) < n_states:
        obs, _ = env.reset()
        done = False
        truncated = False
        
        while not (done or truncated) and len(states) < n_states:
            states.append(obs)
            
            action, _, _ = agent.act(np.array([obs]), deterministic=True)
            action = action[0]
            
            obs, _, done, truncated, _ = env.step(action)
    
    # Analyze action distributions
    print("Analyzing action distributions...")
    
    # Get action statistics for all states
    all_means = []
    all_stds = []
    
    with torch.no_grad():
        for state in states:
            obs_tensor = torch.FloatTensor([state]).to(device)
            features = agent.network.shared_encoder(obs_tensor)
            action_mean = agent.network.policy_mean(features).cpu().numpy()[0]
            action_std = torch.exp(agent.network.policy_log_std).cpu().numpy()[0]
            
            all_means.append(action_mean)
            all_stds.append(action_std)
    
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    
    # Create visualization
    fig, axes = plt.subplots(2, act_dim, figsize=(5 * act_dim, 8))
    if act_dim == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(act_dim):
        # Plot action means histogram
        axes[0, i].hist(all_means[:, i], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[0, i].set_title(f'Action {i} - Mean Distribution')
        axes[0, i].set_xlabel('Mean')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
        
        # Plot action stds histogram
        axes[1, i].hist(all_stds[:, i], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, i].set_title(f'Action {i} - Std Distribution')
        axes[1, i].set_xlabel('Standard Deviation')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Policy Analysis - {env_id}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'policy_analysis_{env_id}.png', dpi=150)
    plt.show()
    
    # Print statistics
    print(f"\nPolicy Statistics:")
    for i in range(act_dim):
        print(f"\nAction {i}:")
        print(f"  Mean - Average: {np.mean(all_means[:, i]):.3f}, Std: {np.std(all_means[:, i]):.3f}")
        print(f"  Std  - Average: {np.mean(all_stds[:, i]):.3f}, Std: {np.std(all_stds[:, i]):.3f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agents")
    
    parser.add_argument("agent_path", type=str,
                        help="Path to saved agent checkpoint")
    parser.add_argument("--env", type=str, required=True,
                        help="Gymnasium environment ID")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("--record", action="store_true",
                        help="Record videos of evaluation")
    parser.add_argument("--video-folder", type=str, default="videos",
                        help="Folder to save videos")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (default: deterministic)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable observation normalization")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--analyze-value", action="store_true",
                        help="Analyze the value function")
    parser.add_argument("--analyze-policy", action="store_true",
                        help="Analyze the policy")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_agent(
        agent_path=args.agent_path,
        env_id=args.env,
        n_episodes=args.episodes,
        render=args.render,
        record_video=args.record,
        video_folder=args.video_folder,
        deterministic=not args.stochastic,
        normalize=not args.no_normalize,
        seed=args.seed
    )
    
    # Additional analyses if requested
    if args.analyze_value:
        visualize_value_function(
            agent_path=args.agent_path,
            env_id=args.env,
            seed=args.seed
        )
    
    if args.analyze_policy:
        analyze_policy(
            agent_path=args.agent_path,
            env_id=args.env,
            seed=args.seed
        )