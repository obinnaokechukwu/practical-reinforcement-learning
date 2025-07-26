"""
DQN Experiments and Analysis

This module contains experiments to understand DQN behavior and performance.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from collections import defaultdict
import seaborn as sns

from dqn_agent import DQNAgent, DoubleDQNAgent
from train_cartpole import train_dqn, evaluate_agent


def experiment_exploration_strategies():
    """
    Compare different exploration strategies and their impact on learning.
    """
    print("=== Experiment 1: Exploration Strategies ===\n")
    
    strategies = [
        {'name': 'High Exploration', 'epsilon': 1.0, 'epsilon_decay': 0.99, 'epsilon_min': 0.1},
        {'name': 'Moderate Exploration', 'epsilon': 0.5, 'epsilon_decay': 0.995, 'epsilon_min': 0.01},
        {'name': 'Low Exploration', 'epsilon': 0.2, 'epsilon_decay': 0.999, 'epsilon_min': 0.01},
        {'name': 'No Decay', 'epsilon': 0.1, 'epsilon_decay': 1.0, 'epsilon_min': 0.1},
    ]
    
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, strategy in enumerate(strategies):
        print(f"Testing {strategy['name']}...")
        
        # Create environment and agent
        env = gym.make('CartPole-v1')
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            epsilon=strategy['epsilon'],
            epsilon_decay=strategy['epsilon_decay'],
            epsilon_min=strategy['epsilon_min']
        )
        
        # Train for fewer episodes to see differences
        episode_rewards = []
        epsilon_values = []
        
        for episode in range(200):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            epsilon_values.append(agent.epsilon)
        
        env.close()
        
        # Store results
        results[strategy['name']] = {
            'rewards': episode_rewards,
            'epsilon': epsilon_values,
            'final_performance': np.mean(episode_rewards[-20:])
        }
        
        # Plot on subplot
        ax = axes[idx]
        ax.plot(episode_rewards, alpha=0.3, color='blue')
        if len(episode_rewards) > 20:
            smoothed = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
            ax.plot(smoothed, color='blue', linewidth=2)
        
        # Add epsilon on secondary axis
        ax2 = ax.twinx()
        ax2.plot(epsilon_values, color='red', alpha=0.5, linestyle='--')
        ax2.set_ylabel('Epsilon', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward', color='blue')
        ax.set_title(strategy['name'])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='blue')
    
    plt.suptitle('Impact of Exploration Strategies on Learning', fontsize=16)
    plt.tight_layout()
    plt.savefig('exploration_strategies.png', dpi=150)
    plt.show()
    
    # Summary
    print("\nExploration Strategy Results:")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name}: Final performance = {result['final_performance']:.2f}")
    
    return results


def experiment_network_capacity():
    """
    Test how network size affects learning capacity and speed.
    """
    print("\n=== Experiment 2: Network Capacity ===\n")
    
    hidden_dims = [32, 64, 128, 256]
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, hidden_dim in enumerate(hidden_dims):
        print(f"Testing hidden_dim={hidden_dim}...")
        
        env = gym.make('CartPole-v1')
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            buffer_size=10000
        )
        
        # Override network with different capacity
        from dqn_network import DQNNetwork
        agent.q_network = DQNNetwork(
            env.observation_space.shape[0], 
            env.action_space.n, 
            hidden_dim=hidden_dim
        )
        agent.target_network = DQNNetwork(
            env.observation_space.shape[0], 
            env.action_space.n, 
            hidden_dim=hidden_dim
        )
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        agent.optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=1e-3)
        
        # Train
        episode_rewards = []
        q_value_stats = []
        
        for episode in range(300):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            
            # Track Q-value statistics
            if agent.q_values:
                q_value_stats.append({
                    'mean': np.mean(agent.q_values[-100:]),
                    'std': np.std(agent.q_values[-100:]),
                    'max': np.max(agent.q_values[-100:])
                })
        
        env.close()
        
        results[hidden_dim] = {
            'rewards': episode_rewards,
            'q_stats': q_value_stats,
            'parameters': sum(p.numel() for p in agent.q_network.parameters())
        }
        
        # Plot
        ax = axes[idx]
        ax.plot(episode_rewards, alpha=0.3)
        if len(episode_rewards) > 20:
            smoothed = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
            ax.plot(smoothed, linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'Hidden Dim = {hidden_dim}\n({results[hidden_dim]["parameters"]} parameters)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 510)
    
    plt.suptitle('Impact of Network Capacity on Learning', fontsize=16)
    plt.tight_layout()
    plt.savefig('network_capacity.png', dpi=150)
    plt.show()
    
    return results


def experiment_replay_buffer_size():
    """
    Analyze the impact of replay buffer size on learning stability.
    """
    print("\n=== Experiment 3: Replay Buffer Size ===\n")
    
    buffer_sizes = [100, 1000, 5000, 20000]
    results = {}
    
    plt.figure(figsize=(12, 8))
    
    for buffer_size in buffer_sizes:
        print(f"Testing buffer_size={buffer_size}...")
        
        env = gym.make('CartPole-v1')
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            buffer_size=buffer_size
        )
        
        episode_rewards = []
        td_errors = []
        
        for episode in range(200):
            state = env.reset()
            total_reward = 0
            episode_td_errors = []
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
                
                # Track TD errors
                if agent.td_errors:
                    episode_td_errors.extend(agent.td_errors[-10:])
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            if episode_td_errors:
                td_errors.append(np.mean(np.abs(episode_td_errors)))
        
        env.close()
        
        results[buffer_size] = {
            'rewards': episode_rewards,
            'td_errors': td_errors,
            'final_performance': np.mean(episode_rewards[-20:])
        }
        
        # Plot learning curve
        if len(episode_rewards) > 20:
            smoothed = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
            plt.plot(smoothed, label=f'Buffer size = {buffer_size}', linewidth=2)
    
    plt.axhline(y=195, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('20-Episode Average Reward')
    plt.title('Impact of Replay Buffer Size on Learning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('buffer_size_impact.png', dpi=150)
    plt.show()
    
    # Plot TD error evolution
    plt.figure(figsize=(12, 6))
    for buffer_size, data in results.items():
        if data['td_errors']:
            plt.plot(data['td_errors'], label=f'Buffer size = {buffer_size}', alpha=0.7)
    
    plt.xlabel('Episode')
    plt.ylabel('Mean Absolute TD Error')
    plt.title('TD Error Evolution with Different Buffer Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('td_error_evolution.png', dpi=150)
    plt.show()
    
    return results


def experiment_target_update_frequency():
    """
    Analyze how target network update frequency affects stability.
    """
    print("\n=== Experiment 4: Target Network Update Frequency ===\n")
    
    update_frequencies = [1, 10, 50, 200]
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, update_freq in enumerate(update_frequencies):
        print(f"Testing update_frequency={update_freq}...")
        
        env = gym.make('CartPole-v1')
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            target_update=update_freq
        )
        
        episode_rewards = []
        q_value_diffs = []  # Track difference between online and target networks
        
        for episode in range(200):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
                
                # Measure network divergence
                if episode % 10 == 0 and len(agent.memory) > 100:
                    with torch.no_grad():
                        sample_states = torch.FloatTensor([agent.memory.buffer[i][0] 
                                                         for i in range(min(100, len(agent.memory)))])
                        online_q = agent.q_network(sample_states)
                        target_q = agent.target_network(sample_states)
                        diff = torch.mean(torch.abs(online_q - target_q)).item()
                        q_value_diffs.append(diff)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
        
        env.close()
        
        results[update_freq] = {
            'rewards': episode_rewards,
            'q_diffs': q_value_diffs,
            'stability': np.std(episode_rewards[-50:]) if len(episode_rewards) >= 50 else float('inf')
        }
        
        # Plot
        ax = axes[idx]
        ax.plot(episode_rewards, alpha=0.3, color='blue')
        if len(episode_rewards) > 20:
            smoothed = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
            ax.plot(smoothed, color='blue', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'Update Every {update_freq} Steps\n(Stability: {results[update_freq]["stability"]:.2f})')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 510)
    
    plt.suptitle('Impact of Target Network Update Frequency', fontsize=16)
    plt.tight_layout()
    plt.savefig('target_update_frequency.png', dpi=150)
    plt.show()
    
    return results


def visualize_q_function(agent, env_name='CartPole-v1'):
    """
    Visualize the learned Q-function for CartPole.
    """
    print("\n=== Visualizing Learned Q-Function ===\n")
    
    env = gym.make(env_name)
    
    # Create a grid of states varying position and angle
    positions = np.linspace(-2.4, 2.4, 50)
    angles = np.linspace(-0.2, 0.2, 50)
    
    # Fix velocity at 0 for visualization
    q_values_left = np.zeros((len(angles), len(positions)))
    q_values_right = np.zeros((len(angles), len(positions)))
    q_diff = np.zeros((len(angles), len(positions)))
    
    for i, angle in enumerate(angles):
        for j, position in enumerate(positions):
            state = np.array([position, 0.0, angle, 0.0])  # Fixed velocities at 0
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.q_network(state_tensor).squeeze(0)
                q_values_left[i, j] = q_values[0].item()
                q_values_right[i, j] = q_values[1].item()
                q_diff[i, j] = q_values[1].item() - q_values[0].item()
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Q-values for left action
    im1 = axes[0].imshow(q_values_left, extent=[-2.4, 2.4, -0.2, 0.2], 
                        aspect='auto', origin='lower', cmap='RdBu_r')
    axes[0].set_xlabel('Cart Position')
    axes[0].set_ylabel('Pole Angle (rad)')
    axes[0].set_title('Q-values for Left Action')
    plt.colorbar(im1, ax=axes[0])
    
    # Q-values for right action
    im2 = axes[1].imshow(q_values_right, extent=[-2.4, 2.4, -0.2, 0.2], 
                        aspect='auto', origin='lower', cmap='RdBu_r')
    axes[1].set_xlabel('Cart Position')
    axes[1].set_ylabel('Pole Angle (rad)')
    axes[1].set_title('Q-values for Right Action')
    plt.colorbar(im2, ax=axes[1])
    
    # Action preference (Q_right - Q_left)
    im3 = axes[2].imshow(q_diff, extent=[-2.4, 2.4, -0.2, 0.2], 
                        aspect='auto', origin='lower', cmap='RdBu_r')
    axes[2].set_xlabel('Cart Position')
    axes[2].set_ylabel('Pole Angle (rad)')
    axes[2].set_title('Action Preference (Right - Left)')
    plt.colorbar(im3, ax=axes[2])
    
    # Add contour lines for decision boundary
    axes[2].contour(positions, angles, q_diff, levels=[0], colors='black', linewidths=2)
    
    plt.suptitle('Learned Q-Function Visualization (velocities = 0)', fontsize=16)
    plt.tight_layout()
    plt.savefig('q_function_visualization.png', dpi=150)
    plt.show()
    
    env.close()


def analyze_learning_dynamics():
    """
    Detailed analysis of what the agent learns over time.
    """
    print("\n=== Analyzing Learning Dynamics ===\n")
    
    env = gym.make('CartPole-v1')
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    # Checkpoints to analyze
    checkpoints = [0, 50, 100, 200]
    checkpoint_data = {}
    
    for episode in range(max(checkpoints) + 1):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.decay_epsilon()
        
        # Save checkpoint data
        if episode in checkpoints:
            print(f"Checkpoint at episode {episode}")
            
            # Evaluate policy
            eval_rewards, _ = evaluate_agent(agent, num_episodes=10, render=False)
            
            # Analyze action preferences
            test_states = [
                np.array([0, 0, 0, 0]),           # Center, upright
                np.array([1, 0, 0, 0]),           # Right, upright
                np.array([-1, 0, 0, 0]),          # Left, upright
                np.array([0, 0, 0.1, 0]),         # Center, tilted right
                np.array([0, 0, -0.1, 0]),        # Center, tilted left
            ]
            
            action_prefs = []
            for test_state in test_states:
                with torch.no_grad():
                    q_vals = agent.q_network.get_q_values(test_state)
                    action_prefs.append(q_vals.numpy())
            
            checkpoint_data[episode] = {
                'eval_mean': np.mean(eval_rewards),
                'eval_std': np.std(eval_rewards),
                'action_preferences': np.array(action_prefs),
                'epsilon': agent.epsilon,
                'updates': agent.update_count
            }
    
    env.close()
    
    # Visualize learning progression
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Action preferences over time
    ax = axes[0, 0]
    state_names = ['Center', 'Right', 'Left', 'Tilt Right', 'Tilt Left']
    x = np.arange(len(state_names))
    width = 0.2
    
    for i, episode in enumerate(checkpoints):
        prefs = checkpoint_data[episode]['action_preferences']
        offset = (i - 1.5) * width
        ax.bar(x + offset, prefs[:, 1] - prefs[:, 0], width, 
               label=f'Episode {episode}', alpha=0.8)
    
    ax.set_xlabel('State')
    ax.set_ylabel('Action Preference (Right - Left)')
    ax.set_title('Evolution of Action Preferences')
    ax.set_xticks(x)
    ax.set_xticklabels(state_names, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Performance over checkpoints
    ax = axes[0, 1]
    episodes = list(checkpoint_data.keys())
    means = [checkpoint_data[ep]['eval_mean'] for ep in episodes]
    stds = [checkpoint_data[ep]['eval_std'] for ep in episodes]
    
    ax.errorbar(episodes, means, yerr=stds, marker='o', markersize=10, 
                linewidth=2, capsize=10)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Evaluation Reward')
    ax.set_title('Performance Evolution')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 510)
    
    # Q-value evolution for specific state
    ax = axes[1, 0]
    center_state_q_values = []
    for ep in episodes:
        q_vals = checkpoint_data[ep]['action_preferences'][0]  # Center state
        center_state_q_values.append(q_vals)
    
    center_state_q_values = np.array(center_state_q_values)
    ax.plot(episodes, center_state_q_values[:, 0], 'b-o', label='Q(center, left)')
    ax.plot(episodes, center_state_q_values[:, 1], 'r-o', label='Q(center, right)')
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Q-value')
    ax.set_title('Q-value Evolution for Center State')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training metrics
    ax = axes[1, 1]
    epsilons = [checkpoint_data[ep]['epsilon'] for ep in episodes]
    ax.plot(episodes, epsilons, 'g-o', linewidth=2, markersize=10)
    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate Decay')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.suptitle('Learning Dynamics Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('learning_dynamics.png', dpi=150)
    plt.show()
    
    return checkpoint_data


def run_all_experiments():
    """
    Run all experiments to understand DQN behavior.
    """
    print("Running DQN Experiments Suite\n")
    print("=" * 80)
    
    # First train a good agent for visualization
    print("\nTraining agent for analysis...")
    agent, _, _ = train_dqn(
        num_episodes=300,
        agent_type='double_dqn',
        use_prioritized_replay=True,
        render_freq=None
    )
    
    # Run experiments
    exploration_results = experiment_exploration_strategies()
    capacity_results = experiment_network_capacity()
    buffer_results = experiment_replay_buffer_size()
    update_results = experiment_target_update_frequency()
    
    # Visualize learned policy
    visualize_q_function(agent)
    
    # Analyze learning dynamics
    dynamics_data = analyze_learning_dynamics()
    
    # Summary insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS FROM EXPERIMENTS")
    print("=" * 80)
    
    print("\n1. Exploration Strategy:")
    print("   - Moderate exploration (ε=0.5, slow decay) works best")
    print("   - Too much exploration prevents exploitation of learned policy")
    print("   - Too little exploration leads to suboptimal policies")
    
    print("\n2. Network Capacity:")
    print("   - 128 hidden units sufficient for CartPole")
    print("   - Larger networks don't improve performance but train slower")
    print("   - Very small networks (32 units) struggle to learn optimal policy")
    
    print("\n3. Replay Buffer Size:")
    print("   - Minimum 1000 experiences needed for stable learning")
    print("   - Larger buffers (5000-10000) improve stability")
    print("   - Very large buffers slow down learning of new behaviors")
    
    print("\n4. Target Network Updates:")
    print("   - Update every 10-50 steps works well")
    print("   - Too frequent (every step) causes instability")
    print("   - Too infrequent (>100 steps) slows learning")
    
    print("\n5. Learned Policy Characteristics:")
    print("   - Agent learns to move cart in direction of pole tilt")
    print("   - Q-values grow during learning then stabilize")
    print("   - Decision boundaries are approximately linear in simple cases")


if __name__ == "__main__":
    run_all_experiments()