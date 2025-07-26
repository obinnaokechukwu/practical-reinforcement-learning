"""
Evaluation and analysis tools for SAC agents.
Includes visualization of learned behaviors.
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import os

from sac_agent import SACAgent, SACAgentWithEnsemble
from train import make_env


def evaluate_agent(
    agent_path: str,
    env_id: str,
    n_episodes: int = 10,
    render: bool = False,
    record_video: bool = False,
    video_dir: str = "videos",
    deterministic: bool = True,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Evaluate a trained SAC agent.
    
    Returns:
        Dictionary with evaluation results
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create environment
    if render:
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id)
    
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            env, video_dir,
            episode_trigger=lambda x: True,
            name_prefix=f"sac_{env_id}"
        )
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Create and load agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    agent.load(agent_path)
    
    print(f"Loaded agent from: {agent_path}")
    print(f"Evaluating on: {env_id}")
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    episode_data = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        states = []
        actions = []
        rewards = []
        
        while not (done or truncated):
            action = agent.act(obs, deterministic=deterministic)
            
            states.append(obs)
            actions.append(action)
            
            obs, reward, done, truncated, _ = env.step(action)
            
            rewards.append(reward)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_data.append({
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards)
        })
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")
    
    env.close()
    
    return {
        'episode_rewards': np.array(episode_rewards),
        'episode_lengths': np.array(episode_lengths),
        'episode_data': episode_data,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }


def analyze_q_values(
    agent_path: str,
    env_id: str,
    n_samples: int = 1000,
    seed: Optional[int] = None
):
    """
    Analyze Q-value estimates across state-action space.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create environment
    env = make_env(env_id, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Load agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    agent.load(agent_path)
    
    # Collect state-action-value samples
    states = []
    actions = []
    q1_values = []
    q2_values = []
    actual_returns = []
    
    print(f"Collecting {n_samples} samples...")
    
    while len(states) < n_samples:
        obs, _ = env.reset()
        done = False
        truncated = False
        
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        while not (done or truncated) and len(states) + len(episode_states) < n_samples:
            action = agent.act(obs, deterministic=False)
            
            # Compute Q-values
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
                
                q1 = agent.q1(obs_tensor, action_tensor).item()
                q2 = agent.q2(obs_tensor, action_tensor).item()
            
            episode_states.append(obs)
            episode_actions.append(action)
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_rewards.append(reward)
        
        # Compute actual returns
        episode_returns = []
        running_return = 0
        for reward in reversed(episode_rewards):
            running_return = reward + agent.gamma * running_return
            episode_returns.insert(0, running_return)
        
        # Add to collections
        n_to_add = min(len(episode_states), n_samples - len(states))
        states.extend(episode_states[:n_to_add])
        actions.extend(episode_actions[:n_to_add])
        actual_returns.extend(episode_returns[:n_to_add])
        
        # Compute Q-values for collected states
        for i in range(n_to_add):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(episode_states[i]).unsqueeze(0).to(device)
                action_tensor = torch.FloatTensor(episode_actions[i]).unsqueeze(0).to(device)
                
                q1 = agent.q1(obs_tensor, action_tensor).item()
                q2 = agent.q2(obs_tensor, action_tensor).item()
                
                q1_values.append(q1)
                q2_values.append(q2)
    
    # Convert to arrays
    q1_values = np.array(q1_values)
    q2_values = np.array(q2_values)
    actual_returns = np.array(actual_returns)
    min_q_values = np.minimum(q1_values, q2_values)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Q1 vs actual returns
    axes[0, 0].scatter(actual_returns, q1_values, alpha=0.5, s=10)
    axes[0, 0].plot([actual_returns.min(), actual_returns.max()],
                    [actual_returns.min(), actual_returns.max()],
                    'r--', label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual Return')
    axes[0, 0].set_ylabel('Q1 Estimate')
    axes[0, 0].set_title('Q1 Network Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q2 vs actual returns
    axes[0, 1].scatter(actual_returns, q2_values, alpha=0.5, s=10)
    axes[0, 1].plot([actual_returns.min(), actual_returns.max()],
                    [actual_returns.min(), actual_returns.max()],
                    'r--', label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Return')
    axes[0, 1].set_ylabel('Q2 Estimate')
    axes[0, 1].set_title('Q2 Network Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min Q vs actual returns
    axes[1, 0].scatter(actual_returns, min_q_values, alpha=0.5, s=10)
    axes[1, 0].plot([actual_returns.min(), actual_returns.max()],
                    [actual_returns.min(), actual_returns.max()],
                    'r--', label='Perfect prediction')
    axes[1, 0].set_xlabel('Actual Return')
    axes[1, 0].set_ylabel('Min(Q1, Q2) Estimate')
    axes[1, 0].set_title('Conservative Q Estimate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q1 vs Q2
    axes[1, 1].scatter(q1_values, q2_values, alpha=0.5, s=10)
    axes[1, 1].plot([min(q1_values.min(), q2_values.min()),
                     max(q1_values.max(), q2_values.max())],
                    [min(q1_values.min(), q2_values.min()),
                     max(q1_values.max(), q2_values.max())],
                    'r--', label='Q1 = Q2')
    axes[1, 1].set_xlabel('Q1 Estimate')
    axes[1, 1].set_ylabel('Q2 Estimate')
    axes[1, 1].set_title('Q-Network Agreement')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Q-Value Analysis - {env_id}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'q_value_analysis_{env_id}.png', dpi=150)
    plt.show()
    
    # Compute statistics
    q1_error = np.mean(np.abs(q1_values - actual_returns))
    q2_error = np.mean(np.abs(q2_values - actual_returns))
    min_q_error = np.mean(np.abs(min_q_values - actual_returns))
    
    q1_correlation = np.corrcoef(q1_values, actual_returns)[0, 1]
    q2_correlation = np.corrcoef(q2_values, actual_returns)[0, 1]
    min_q_correlation = np.corrcoef(min_q_values, actual_returns)[0, 1]
    
    print(f"\nQ-Value Statistics:")
    print(f"Q1 MAE: {q1_error:.3f}, Correlation: {q1_correlation:.3f}")
    print(f"Q2 MAE: {q2_error:.3f}, Correlation: {q2_correlation:.3f}")
    print(f"Min Q MAE: {min_q_error:.3f}, Correlation: {min_q_correlation:.3f}")
    print(f"Q1-Q2 Correlation: {np.corrcoef(q1_values, q2_values)[0, 1]:.3f}")
    
    env.close()


def analyze_policy_entropy(
    agent_path: str,
    env_id: str,
    n_episodes: int = 5,
    seed: Optional[int] = None
):
    """
    Analyze policy entropy over episodes.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create environment
    env = make_env(env_id, seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Load agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    agent.load(agent_path)
    
    print(f"Analyzing policy entropy for {n_episodes} episodes...")
    
    all_entropies = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        episode_entropies = []
        episode_states = []
        
        while not (done or truncated):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                _, log_prob = agent.policy.sample(obs_tensor)
                entropy = -log_prob.item()
            
            episode_entropies.append(entropy)
            episode_states.append(obs)
            
            action = agent.act(obs, deterministic=False)
            obs, _, done, truncated, _ = env.step(action)
        
        all_entropies.append(episode_entropies)
        print(f"Episode {episode + 1}: Mean entropy = {np.mean(episode_entropies):.3f}")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    
    for i, entropies in enumerate(all_entropies):
        plt.plot(entropies, alpha=0.7, label=f'Episode {i+1}')
    
    plt.xlabel('Timestep')
    plt.ylabel('Policy Entropy')
    plt.title(f'Policy Entropy During Episodes - {env_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'policy_entropy_{env_id}.png', dpi=150)
    plt.show()
    
    # Overall statistics
    all_entropies_flat = [e for ep in all_entropies for e in ep]
    print(f"\nOverall entropy statistics:")
    print(f"Mean: {np.mean(all_entropies_flat):.3f}")
    print(f"Std: {np.std(all_entropies_flat):.3f}")
    print(f"Min: {np.min(all_entropies_flat):.3f}")
    print(f"Max: {np.max(all_entropies_flat):.3f}")
    
    env.close()


def compare_deterministic_stochastic(
    agent_path: str,
    env_id: str,
    n_episodes: int = 20,
    seed: Optional[int] = None
):
    """
    Compare deterministic and stochastic policy performance.
    """
    print("Evaluating deterministic policy...")
    det_results = evaluate_agent(
        agent_path, env_id, n_episodes, 
        deterministic=True, seed=seed
    )
    
    print("\nEvaluating stochastic policy...")
    stoch_results = evaluate_agent(
        agent_path, env_id, n_episodes, 
        deterministic=False, seed=seed
    )
    
    # Statistical comparison
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        det_results['episode_rewards'],
        stoch_results['episode_rewards']
    )
    
    print("\n" + "=" * 50)
    print("Performance Comparison:")
    print(f"Deterministic: {det_results['mean_reward']:.2f} ± {det_results['std_reward']:.2f}")
    print(f"Stochastic: {stoch_results['mean_reward']:.2f} ± {stoch_results['std_reward']:.2f}")
    print(f"T-test p-value: {p_value:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    plt.boxplot([det_results['episode_rewards'], stoch_results['episode_rewards']],
                labels=['Deterministic', 'Stochastic'])
    plt.ylabel('Episode Reward')
    plt.title(f'Policy Comparison - {env_id}')
    plt.grid(True, alpha=0.3)
    
    # Add mean lines
    plt.axhline(y=det_results['mean_reward'], color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=stoch_results['mean_reward'], color='orange', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'policy_comparison_{env_id}.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAC agents")
    
    parser.add_argument("agent_path", type=str,
                        help="Path to saved agent")
    parser.add_argument("--env", type=str, required=True,
                        help="Environment ID")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    parser.add_argument("--record", action="store_true",
                        help="Record videos")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory for videos")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--analyze-q", action="store_true",
                        help="Analyze Q-values")
    parser.add_argument("--analyze-entropy", action="store_true",
                        help="Analyze policy entropy")
    parser.add_argument("--compare-policies", action="store_true",
                        help="Compare deterministic vs stochastic")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_agent(
        args.agent_path,
        args.env,
        args.episodes,
        args.render,
        args.record,
        args.video_dir,
        not args.stochastic,
        args.seed
    )
    
    # Additional analyses
    if args.analyze_q:
        analyze_q_values(args.agent_path, args.env, seed=args.seed)
    
    if args.analyze_entropy:
        analyze_policy_entropy(args.agent_path, args.env, seed=args.seed)
    
    if args.compare_policies:
        compare_deterministic_stochastic(args.agent_path, args.env, seed=args.seed)