"""
Training Script for CartPole-v1

Complete training pipeline for DQN on CartPole environment.
"""

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import os

from dqn_agent import DQNAgent, DoubleDQNAgent


def train_dqn(env_name='CartPole-v1',
              num_episodes=500,
              agent_type='dqn',
              use_prioritized_replay=False,
              use_dueling=False,
              render_freq=None,
              save_freq=50,
              checkpoint_dir='checkpoints'):
    """
    Train DQN on specified environment.
    
    Args:
        env_name: Gym environment name
        num_episodes: Number of episodes to train
        agent_type: 'dqn' or 'double_dqn'
        use_prioritized_replay: Whether to use prioritized experience replay
        use_dueling: Whether to use dueling architecture
        render_freq: Render every N episodes (None to disable)
        save_freq: Save checkpoint every N episodes
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        Trained agent, episode rewards, episode lengths
    """
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Training {agent_type.upper()} on {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Prioritized replay: {use_prioritized_replay}")
    print(f"Dueling architecture: {use_dueling}\n")
    
    # Create agent
    agent_class = DoubleDQNAgent if agent_type == 'double_dqn' else DQNAgent
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        use_prioritized_replay=use_prioritized_replay,
        use_dueling=use_dueling
    )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    solved = False
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Episode loop
        while True:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            agent.train_step()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render if requested
            if render_freq and episode % render_freq == 0:
                env.render()
            
            if done:
                break
        
        # Post-episode updates
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Progress report
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            diagnostics = agent.get_diagnostics()
            
            print(f"\nEpisode {episode}:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Length: {avg_length:.2f}")
            print(f"  Epsilon: {diagnostics['epsilon']:.3f}")
            print(f"  Loss: {diagnostics.get('loss_mean', 0):.4f}")
            print(f"  Q-values: {diagnostics.get('q_value_mean', 0):.2f}")
        
        # Save checkpoint
        if episode % save_freq == 0 and episode > 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"{agent_type}_episode_{episode}.pth"
            )
            agent.save(checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
        
        # Check if solved
        if len(episode_rewards) >= 100:
            if np.mean(episode_rewards[-100:]) >= 195.0:
                print(f"\n🎉 Environment solved in {episode} episodes!")
                solved = True
                break
    
    env.close()
    
    # Save final model and training history
    final_checkpoint = os.path.join(checkpoint_dir, f"{agent_type}_final.pth")
    agent.save(final_checkpoint)
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'solved': solved,
        'final_episode': episode,
        'config': {
            'env_name': env_name,
            'agent_type': agent_type,
            'use_prioritized_replay': use_prioritized_replay,
            'use_dueling': use_dueling
        }
    }
    
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return agent, episode_rewards, episode_lengths


def visualize_training(episode_rewards, episode_lengths, agent, save_path='training_plots.png'):
    """
    Create comprehensive visualization of training progress.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Episode rewards with success threshold
    ax = axes[0, 0]
    ax.plot(episode_rewards, alpha=0.3, color='blue', label='Raw rewards')
    
    # Smoothed rewards
    if len(episode_rewards) > 20:
        window = min(50, len(episode_rewards) // 4)
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        x_smooth = np.arange(window//2, len(episode_rewards) - window//2 + 1)
        ax.plot(x_smooth, smoothed, color='blue', linewidth=2, label=f'{window}-ep average')
    
    # Success threshold
    ax.axhline(y=195, color='red', linestyle='--', label='Success threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Learning Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode lengths
    ax = axes[0, 1]
    ax.plot(episode_lengths, alpha=0.3, color='green')
    
    if len(episode_lengths) > 20:
        window = min(50, len(episode_lengths) // 4)
        smoothed = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        x_smooth = np.arange(window//2, len(episode_lengths) - window//2 + 1)
        ax.plot(x_smooth, smoothed, color='green', linewidth=2)
    
    ax.axhline(y=500, color='red', linestyle='--', label='Max possible')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Loss evolution
    ax = axes[1, 0]
    if agent.losses:
        # Sample losses for plotting (too many points otherwise)
        losses = agent.losses
        if len(losses) > 1000:
            indices = np.linspace(0, len(losses)-1, 1000, dtype=int)
            losses = [losses[i] for i in indices]
            x_vals = indices
        else:
            x_vals = range(len(losses))
        
        ax.plot(x_vals, losses, alpha=0.7, color='red')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Evolution')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    # 4. Q-value statistics
    ax = axes[1, 1]
    if agent.q_values:
        q_values = agent.q_values
        if len(q_values) > 1000:
            indices = np.linspace(0, len(q_values)-1, 1000, dtype=int)
            q_values = [q_values[i] for i in indices]
            x_vals = indices
        else:
            x_vals = range(len(q_values))
        
        ax.plot(x_vals, q_values, alpha=0.7, color='purple')
        ax.set_xlabel('Action Selection Step')
        ax.set_ylabel('Average Q-value')
        ax.set_title('Q-value Evolution')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('DQN Training Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nTraining Summary:")
    print(f"  Total episodes: {len(episode_rewards)}")
    print(f"  Final average reward (100 ep): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"  Best episode reward: {max(episode_rewards):.2f}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  Total updates: {agent.update_count}")


def evaluate_agent(agent, env_name='CartPole-v1', num_episodes=10, render=True):
    """
    Evaluate trained agent performance.
    """
    env = gym.make(env_name)
    rewards = []
    lengths = []
    
    print(f"\nEvaluating agent on {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            if render:
                env.render()
            
            # Always exploit during evaluation
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {episode + 1}: Reward = {total_reward}, Length = {steps}")
    
    env.close()
    
    print(f"\nEvaluation Results:")
    print(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Average Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"  Success Rate: {sum(r >= 195 for r in rewards) / len(rewards):.1%}")
    
    return rewards, lengths


def compare_algorithms():
    """
    Compare different DQN variants on CartPole.
    """
    configurations = [
        {'name': 'Vanilla DQN', 'agent_type': 'dqn', 
         'use_prioritized_replay': False, 'use_dueling': False},
        {'name': 'DQN + PER', 'agent_type': 'dqn', 
         'use_prioritized_replay': True, 'use_dueling': False},
        {'name': 'Double DQN', 'agent_type': 'double_dqn', 
         'use_prioritized_replay': False, 'use_dueling': False},
        {'name': 'Dueling DQN', 'agent_type': 'dqn', 
         'use_prioritized_replay': False, 'use_dueling': True},
        {'name': 'Rainbow Lite', 'agent_type': 'double_dqn', 
         'use_prioritized_replay': True, 'use_dueling': True},
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Training {config['name']}")
        print(f"{'='*60}")
        
        # Train agent
        agent, rewards, lengths = train_dqn(
            num_episodes=300,
            render_freq=None,
            checkpoint_dir=f"checkpoints/{config['name'].replace(' ', '_')}",
            **{k: v for k, v in config.items() if k != 'name'}
        )
        
        # Evaluate
        eval_rewards, _ = evaluate_agent(agent, num_episodes=20, render=False)
        
        results[config['name']] = {
            'training_rewards': rewards,
            'eval_mean': np.mean(eval_rewards),
            'eval_std': np.std(eval_rewards),
            'episodes_to_solve': len(rewards) if np.mean(rewards[-100:]) >= 195 else None
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        rewards = result['training_rewards']
        if len(rewards) > 20:
            smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
            plt.plot(smoothed, label=name, linewidth=2)
    
    plt.axhline(y=195, color='red', linestyle='--', label='Success threshold')
    plt.xlabel('Episode')
    plt.ylabel('20-Episode Average Reward')
    plt.title('DQN Variants Comparison on CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print comparison table
    print("\n\nAlgorithm Comparison Results:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Eval Score':<15} {'Episodes to Solve':<20} {'Success Rate':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        eval_score = f"{result['eval_mean']:.1f} ± {result['eval_std']:.1f}"
        episodes = result['episodes_to_solve'] or "Not solved"
        success_rate = f"{result['eval_mean'] / 500:.1%}"
        print(f"{name:<20} {eval_score:<15} {str(episodes):<20} {success_rate:<15}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN on CartPole-v1')
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--agent', choices=['dqn', 'double_dqn'], default='dqn')
    parser.add_argument('--prioritized', action='store_true', help='Use prioritized replay')
    parser.add_argument('--dueling', action='store_true', help='Use dueling architecture')
    parser.add_argument('--render', action='store_true', help='Render during training')
    parser.add_argument('--compare', action='store_true', help='Compare all algorithms')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_algorithms()
    else:
        # Train single agent
        agent, rewards, lengths = train_dqn(
            num_episodes=args.episodes,
            agent_type=args.agent,
            use_prioritized_replay=args.prioritized,
            use_dueling=args.dueling,
            render_freq=10 if args.render else None
        )
        
        # Visualize results
        visualize_training(rewards, lengths, agent)
        
        # Final evaluation
        evaluate_agent(agent, num_episodes=10, render=True)