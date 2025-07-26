"""
Q-Learning Experiments

Comprehensive experiments to demonstrate Q-learning capabilities and behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

from treasure_hunt_env import TreasureHuntEnv
from q_learning_agent import TreasureHuntQLearning, BasicQLearningAgent, evaluate_policy


def experiment_basic_learning():
    """
    Observe how Q-learning progressively improves its policy.
    """
    print("=== Experiment 1: Basic Learning Dynamics ===\n")
    
    # Create environment and agent
    env = TreasureHuntEnv(width=6, height=6, num_treasures=2, num_traps=3)
    agent = TreasureHuntQLearning(env)
    
    # Track learning at different stages
    stages = [10, 50, 100, 500]
    stage_results = {}
    
    for num_episodes in stages:
        # Train for additional episodes
        agent.train(num_episodes - (stages[stages.index(num_episodes)-1] 
                                   if stages.index(num_episodes) > 0 else 0),
                   verbose=False)
        
        # Evaluate current policy
        test_rewards = []
        for _ in range(20):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.get_action(state, training=False)
                state, reward, done = env.step(action)
                episode_reward += reward
            
            test_rewards.append(episode_reward)
        
        stage_results[num_episodes] = {
            'mean_reward': np.mean(test_rewards),
            'std_reward': np.std(test_rewards),
            'q_values_learned': len(agent.Q)
        }
        
        print(f"After {num_episodes} episodes:")
        print(f"  Mean reward: {stage_results[num_episodes]['mean_reward']:.2f}")
        print(f"  Std reward: {stage_results[num_episodes]['std_reward']:.2f}")
        print(f"  States discovered: {stage_results[num_episodes]['q_values_learned']}")
    
    # Visualize progression
    episodes = list(stage_results.keys())
    mean_rewards = [stage_results[e]['mean_reward'] for e in episodes]
    std_rewards = [stage_results[e]['std_reward'] for e in episodes]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(episodes, mean_rewards, yerr=std_rewards, 
                 marker='o', markersize=10, linewidth=2, capsize=10)
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Test Reward', fontsize=12)
    plt.title('Q-Learning Performance Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('learning_progression.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return stage_results


def experiment_exploration_impact():
    """
    Demonstrate how exploration affects learning and final performance.
    """
    print("\n=== Experiment 2: Impact of Exploration ===\n")
    
    # Test different exploration strategies
    exploration_configs = [
        {'epsilon': 0.01, 'decay': 1.0, 'name': 'Minimal (ε=0.01)'},
        {'epsilon': 0.1, 'decay': 1.0, 'name': 'Moderate (ε=0.1)'},
        {'epsilon': 0.5, 'decay': 1.0, 'name': 'High (ε=0.5)'},
        {'epsilon': 1.0, 'decay': 0.995, 'name': 'Decaying (ε=1→0.01)'}
    ]
    
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, config in enumerate(exploration_configs):
        # Create fresh environment and agent
        env = TreasureHuntEnv(width=8, height=8, num_treasures=3, num_traps=4)
        agent = TreasureHuntQLearning(env, epsilon=config['epsilon'])
        agent.epsilon_decay = config['decay']
        
        # Train
        agent.train(num_episodes=500, verbose=False)
        
        # Plot learning curve
        ax = axes[idx]
        ax.plot(agent.episode_rewards, alpha=0.3, color='blue')
        
        # Add smoothed curve
        if len(agent.episode_rewards) > 50:
            smoothed = np.convolve(agent.episode_rewards, 
                                 np.ones(50)/50, mode='valid')
            ax.plot(smoothed, linewidth=2, color='red', 
                   label='50-episode average')
        
        ax.set_title(config['name'], fontsize=12)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-60, 100)
        
        # Store results
        results[config['name']] = {
            'final_performance': np.mean(agent.episode_rewards[-50:]),
            'states_explored': len(agent.Q),
            'convergence_episode': find_convergence_episode(agent.episode_rewards)
        }
    
    plt.suptitle('Exploration Strategy Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('exploration_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary table
    print("\nExploration Strategy Results:")
    print("-" * 70)
    print(f"{'Strategy':<20} {'Final Reward':<15} {'States Found':<15} {'Convergence':<15}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<20} {res['final_performance']:<15.2f} "
              f"{res['states_explored']:<15} {res['convergence_episode']:<15}")
    
    return results


def experiment_q_value_analysis():
    """
    Visualize learned Q-values to understand the agent's policy.
    """
    print("\n=== Experiment 3: Q-Value Analysis ===\n")
    
    # Train an agent
    env = TreasureHuntEnv(width=5, height=5, num_treasures=1, num_traps=2)
    agent = TreasureHuntQLearning(env)
    agent.train(num_episodes=500, verbose=False)
    
    # Create Q-value heatmap for the starting state
    # This shows which actions the agent prefers in different positions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for action in range(4):
        ax = axes[action // 2, action % 2]
        
        # Create grid of Q-values for this action
        q_grid = np.full((env.height, env.width), np.nan)
        
        # Fill in Q-values for positions where agent has been
        for state, action_values in agent.Q.items():
            if len(state) == 4:  # Full state representation
                pos = state[0]  # Extract position
                # Use simplified state for visualization
                simple_state = (pos, frozenset(), frozenset(), state[3])
                if simple_state in agent.Q:
                    q_grid[pos] = agent.Q[simple_state][action]
        
        # Plot heatmap
        masked_grid = np.ma.masked_invalid(q_grid)
        im = ax.imshow(masked_grid, cmap='RdYlGn', aspect='equal')
        ax.set_title(f'Q-values for {action_names[action]}', fontsize=12)
        
        # Mark special positions
        ax.plot(0, 0, 'bo', markersize=10)  # Start
        
        # Mark treasures
        for pos in env.treasures:
            ax.plot(pos[1], pos[0], 'y*', markersize=15)
        
        # Mark traps
        for pos in env.traps:
            ax.plot(pos[1], pos[0], 'rx', markersize=12)
        
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(env.height - 0.5, -0.5)
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Q-Value Heatmaps by Action', fontsize=16)
    plt.tight_layout()
    plt.savefig('q_value_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Demonstrate optimal path
    print("\nOptimal path from start:")
    state = env.reset()
    path = [state[0]]  # Track positions
    
    for step in range(20):
        action = agent.get_action(state, training=False)
        state, reward, done = env.step(action)
        path.append(state[0])
        
        if done:
            print(f"  Path: {' -> '.join(str(p) for p in path)}")
            print(f"  Result: {'Success!' if reward > 0 else 'Failed!'}")
            break


def experiment_learning_rate_comparison():
    """
    Compare different learning rates to understand their impact.
    """
    print("\n=== Experiment 4: Learning Rate Analysis ===\n")
    
    alphas = [0.01, 0.1, 0.3, 0.5]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, alpha in enumerate(alphas):
        env = TreasureHuntEnv(width=8, height=8, num_treasures=3, num_traps=4)
        agent = TreasureHuntQLearning(env, alpha=alpha)
        
        agent.train(num_episodes=500, verbose=False)
        
        axes[idx].plot(agent.episode_rewards, alpha=0.3)
        if len(agent.episode_rewards) > 50:
            smoothed = np.convolve(agent.episode_rewards,
                                  np.ones(50)/50, mode='valid')
            axes[idx].plot(smoothed, linewidth=2)
        
        axes[idx].set_title(f'α = {alpha}', fontsize=14)
        axes[idx].set_xlabel('Episode')
        axes[idx].set_ylabel('Reward')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-100, 100)
        
        # Add text with final performance
        final_perf = np.mean(agent.episode_rewards[-50:])
        axes[idx].text(0.05, 0.95, f'Final: {final_perf:.1f}', 
                      transform=axes[idx].transAxes,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Learning Rate Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def experiment_initialization_impact():
    """
    Test how Q-value initialization affects exploration and learning.
    """
    print("\n=== Experiment 5: Q-Value Initialization ===\n")
    
    init_values = [-10, 0, 5, 10]
    init_results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, init_val in enumerate(init_values):
        env = TreasureHuntEnv(width=8, height=8, num_treasures=3, num_traps=4)
        agent = TreasureHuntQLearning(env)
        
        # Override initialization
        agent.Q = defaultdict(lambda: defaultdict(lambda: init_val))
        
        agent.train(num_episodes=300, verbose=False)
        
        # Plot learning curve
        ax = axes[idx]
        ax.plot(agent.episode_rewards, alpha=0.5)
        if len(agent.episode_rewards) > 30:
            smoothed = np.convolve(agent.episode_rewards,
                                  np.ones(30)/30, mode='valid')
            ax.plot(smoothed, linewidth=2)
        
        ax.set_title(f'Q_init = {init_val}', fontsize=14)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-100, 100)
        
        # Measure early exploration
        early_rewards = agent.episode_rewards[:50]
        late_rewards = agent.episode_rewards[-50:]
        
        init_results[init_val] = {
            'early_mean': np.mean(early_rewards),
            'late_mean': np.mean(late_rewards),
            'convergence_episode': find_convergence_episode(agent.episode_rewards),
            'states_explored': len(agent.Q)
        }
    
    plt.suptitle('Q-Value Initialization Impact', fontsize=16)
    plt.tight_layout()
    plt.savefig('initialization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nInitialization Results:")
    print("-" * 80)
    print(f"{'Q_init':<10} {'Early Perf':<15} {'Late Perf':<15} {'Convergence':<15} {'States':<10}")
    print("-" * 80)
    for init_val, results in init_results.items():
        print(f"{init_val:<10} {results['early_mean']:<15.2f} "
              f"{results['late_mean']:<15.2f} "
              f"{results['convergence_episode']:<15} "
              f"{results['states_explored']:<10}")


def find_convergence_episode(rewards, window=50, threshold=5):
    """Find episode where performance stabilizes."""
    if len(rewards) < window * 2:
        return len(rewards)
    
    for i in range(window, len(rewards) - window):
        recent_std = np.std(rewards[i:i+window])
        if recent_std < threshold:
            return i
    
    return len(rewards)


def run_all_experiments():
    """
    Run all experiments to demonstrate Q-learning capabilities.
    """
    print("=== Q-Learning Comprehensive Experiments ===\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run experiments
    learning_results = experiment_basic_learning()
    exploration_results = experiment_exploration_impact()
    experiment_q_value_analysis()
    experiment_learning_rate_comparison()
    experiment_initialization_impact()
    
    # Summary insights
    print("\n=== Key Insights ===")
    print("1. Q-learning learns optimal policies through trial and error")
    print("2. Exploration is crucial for discovering good policies")
    print("3. The algorithm is robust to different exploration strategies")
    print("4. Q-values directly encode the learned policy")
    print("5. Off-policy learning enables learning from any experience")
    print("6. Learning rate affects stability vs speed tradeoff")
    print("7. Optimistic initialization encourages exploration")


if __name__ == "__main__":
    run_all_experiments()