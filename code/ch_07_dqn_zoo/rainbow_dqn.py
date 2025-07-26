"""
Rainbow DQN: Combining All Improvements

Demonstrates how Double DQN, Dueling architecture, and Prioritized Replay
can be combined into a single powerful agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gym
from dueling_dqn import DuelingNetwork
from prioritized_replay import PrioritizedReplayBuffer
from double_dqn import DoubleDQN


class RainbowDQN:
    """
    Rainbow DQN combining multiple improvements:
    - Double Q-learning (reduced overestimation)
    - Dueling architecture (value-advantage decomposition)
    - Prioritized experience replay (focused learning)
    
    Not included (but in full Rainbow):
    - Multi-step returns
    - Distributional RL (C51)
    - Noisy networks
    """
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, alpha=0.6, beta_start=0.4,
                 n_step=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_step = n_step
        
        # Dueling networks
        self.q_network = DuelingNetwork(state_dim, action_dim)
        self.target_network = DuelingNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Prioritized replay
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha, beta_start)
        
        # N-step buffer
        self.n_step_buffer = []
        
        # Metrics
        self.losses = []
        self.q_values = []
        self.values = []
        self.advantages = []
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            # Track decomposition
            value, advantages = self.q_network.get_value_advantage(state_tensor)
            self.values.append(value.item())
            self.advantages.append(advantages.squeeze().numpy())
            self.q_values.append(q_values.max().item())
            
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition with n-step returns."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If episode ends or buffer is full
        if done or len(self.n_step_buffer) == self.n_step:
            # Calculate n-step return
            n_step_return = 0
            for i, (s, a, r, s_, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma ** i) * r
                if d:
                    # Store n-step transition
                    self.memory.push((self.n_step_buffer[0][0], 
                                    self.n_step_buffer[0][1],
                                    n_step_return, s_, d))
                    self.n_step_buffer = []
                    return
            
            # Store n-step transition
            if self.n_step_buffer:
                self.memory.push((self.n_step_buffer[0][0], 
                                self.n_step_buffer[0][1],
                                n_step_return, 
                                self.n_step_buffer[-1][3],
                                self.n_step_buffer[-1][4]))
                self.n_step_buffer.pop(0)
    
    def train_step(self, batch_size=64):
        """
        Training with all improvements:
        - Double Q-learning targets
        - Dueling architecture
        - Prioritized sampling
        - N-step returns
        """
        if self.memory.tree.n_entries < batch_size:
            return None
        
        # Prioritized sampling
        batch, idxs, weights = self.memory.sample(batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])  # n-step returns
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        # Current Q-values (dueling architecture)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN targets with dueling
        with torch.no_grad():
            # Online network selects actions
            next_actions = self.q_network(next_states).argmax(dim=1)
            # Target network evaluates (dueling architecture)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            # N-step targets
            targets = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
        
        # TD errors for priorities
        td_errors = targets - current_q_values.squeeze()
        
        # Weighted loss (prioritized replay)
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(idxs, td_errors.detach().numpy())
        
        self.losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def compare_all_variants(num_episodes=500):
    """Compare vanilla DQN, individual improvements, and Rainbow."""
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Different configurations to test
    configs = [
        ('Vanilla DQN', {
            'double': False,
            'dueling': False, 
            'prioritized': False,
            'n_step': 1
        }),
        ('Double DQN', {
            'double': True,
            'dueling': False,
            'prioritized': False,
            'n_step': 1
        }),
        ('Dueling DQN', {
            'double': False,
            'dueling': True,
            'prioritized': False,
            'n_step': 1
        }),
        ('Prioritized DQN', {
            'double': False,
            'dueling': False,
            'prioritized': True,
            'n_step': 1
        }),
        ('Rainbow DQN', {
            'double': True,
            'dueling': True,
            'prioritized': True,
            'n_step': 3
        })
    ]
    
    results = {}
    
    for name, config in configs:
        print(f"\nTraining {name}...")
        
        if name == 'Rainbow DQN':
            agent = RainbowDQN(state_dim, action_dim, n_step=config['n_step'])
        else:
            # Use appropriate agent based on config
            if config['prioritized']:
                from prioritized_replay import PrioritizedDQN
                agent = PrioritizedDQN(state_dim, action_dim)
            elif config['dueling']:
                from dueling_dqn import DuelingDQN
                agent = DuelingDQN(state_dim, action_dim)
            else:
                agent = DoubleDQN(state_dim, action_dim)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train
                if hasattr(agent, 'memory'):
                    if hasattr(agent.memory, 'tree') and agent.memory.tree.n_entries >= 64:
                        agent.train_step()
                    elif hasattr(agent.memory, '__len__') and len(agent.memory) >= 64:
                        if name == 'Double DQN' or name == 'Vanilla DQN':
                            agent.train_step(use_double=config['double'])
                        else:
                            agent.train_step()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Update target network
            if episode % 10 == 0:
                agent.update_target_network()
                if episode % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    print(f"  Episode {episode}: Avg Reward = {avg_reward:.2f}")
        
        results[name] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'final_performance': np.mean(episode_rewards[-100:])
        }
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning curves
    ax = axes[0, 0]
    for name, data in results.items():
        if len(data['rewards']) > 50:
            smooth = np.convolve(data['rewards'], np.ones(50)/50, mode='valid')
            ax.plot(smooth, label=name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('50-Episode Average Reward')
    ax.set_title('Learning Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=195, color='red', linestyle='--', alpha=0.5, label='Solved')
    
    # Time to solve
    ax = axes[0, 1]
    solve_times = []
    names = []
    for name, data in results.items():
        # Find first episode where 100-episode average >= 195
        rewards = data['rewards']
        solved = False
        for i in range(100, len(rewards)):
            if np.mean(rewards[i-100:i]) >= 195:
                solve_times.append(i)
                solved = True
                break
        if not solved:
            solve_times.append(len(rewards))
        names.append(name)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), solve_times, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Episodes to Solve')
    ax.set_title('Sample Efficiency Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, time in zip(bars, solve_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(time)}', ha='center', va='bottom')
    
    # Final performance
    ax = axes[1, 0]
    final_perfs = [results[name]['final_performance'] for name in results.keys()]
    bars = ax.bar(range(len(names)), final_perfs, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Average Reward (last 100 episodes)')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=195, color='red', linestyle='--', alpha=0.5)
    
    # Improvement over vanilla
    ax = axes[1, 1]
    vanilla_perf = results['Vanilla DQN']['final_performance']
    improvements = [(results[name]['final_performance'] / vanilla_perf - 1) * 100 
                   for name in results.keys()]
    bars = ax.bar(range(len(names)), improvements, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Improvement over Vanilla DQN (%)')
    ax.set_title('Relative Performance Improvement')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=1)
    
    # Add values on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.1f}%', ha='center', 
                va='bottom' if imp > 0 else 'top')
    
    plt.suptitle('DQN Zoo: Comprehensive Algorithm Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('rainbow_comparison.png', dpi=150)
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Algorithm':<20} {'Final Reward':<15} {'Episodes to Solve':<20} {'Improvement':<15}")
    print("-"*80)
    
    for i, name in enumerate(names):
        final_reward = final_perfs[i]
        solve_time = solve_times[i]
        improvement = improvements[i]
        print(f"{name:<20} {final_reward:<15.2f} {solve_time:<20} {improvement:>+14.1f}%")
    
    env.close()
    return results


def visualize_rainbow_components(agent, env_name='CartPole-v1'):
    """Visualize the different components of Rainbow DQN."""
    
    if not hasattr(agent, 'values') or not agent.values:
        print("No data to visualize. Train the agent first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Value evolution
    ax = axes[0, 0]
    ax.plot(agent.values, alpha=0.7, color='blue')
    ax.set_xlabel('Steps')
    ax.set_ylabel('State Value V(s)')
    ax.set_title('State Value Evolution')
    ax.grid(True, alpha=0.3)
    
    # Advantage distribution
    ax = axes[0, 1]
    if agent.advantages:
        all_advantages = np.concatenate(agent.advantages)
        ax.hist(all_advantages, bins=50, alpha=0.7, color='green', density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Advantage Value')
        ax.set_ylabel('Density')
        ax.set_title('Advantage Distribution (Dueling)')
        ax.grid(True, alpha=0.3)
    
    # Q-value evolution
    ax = axes[1, 0]
    ax.plot(agent.q_values, alpha=0.7, color='purple')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Max Q-value')
    ax.set_title('Q-value Evolution')
    ax.grid(True, alpha=0.3)
    
    # Loss evolution
    ax = axes[1, 1]
    if agent.losses:
        ax.plot(agent.losses, alpha=0.5, color='red')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (with Priority Weighting)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Rainbow DQN Component Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('rainbow_components.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("=== Rainbow DQN: Combining All Improvements ===\n")
    
    # Compare all variants
    results = compare_all_variants(num_episodes=500)
    
    # Train and analyze Rainbow specifically
    print("\n\nTraining Rainbow DQN for component analysis...")
    env = gym.make('CartPole-v1')
    rainbow_agent = RainbowDQN(
        env.observation_space.shape[0],
        env.action_space.n
    )
    
    for episode in range(200):
        state = env.reset()
        while True:
            action = rainbow_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            rainbow_agent.store_transition(state, action, reward, next_state, done)
            
            if rainbow_agent.memory.tree.n_entries >= 64:
                rainbow_agent.train_step()
            
            state = next_state
            if done:
                break
        
        rainbow_agent.decay_epsilon()
        if episode % 10 == 0:
            rainbow_agent.update_target_network()
    
    env.close()
    
    # Visualize components
    print("\nVisualizing Rainbow components...")
    visualize_rainbow_components(rainbow_agent)