"""
Double DQN Implementation

Demonstrates how Double DQN reduces overestimation bias by decoupling
action selection from action evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt


class DQNNetwork(nn.Module):
    """Standard feedforward network for Q-value approximation."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def compute_dqn_targets(q_network, target_network, next_states, rewards, dones, gamma=0.99):
    """
    Standard DQN target computation.
    Uses the same network (target) for both action selection and evaluation.
    """
    with torch.no_grad():
        # Target network selects AND evaluates
        next_q_values = target_network(next_states)
        max_next_q = next_q_values.max(dim=1)[0]
        targets = rewards + (1 - dones) * gamma * max_next_q
    return targets


def compute_double_dqn_targets(q_network, target_network, next_states, rewards, dones, gamma=0.99):
    """
    Double DQN target computation.
    Decouples action selection (online network) from evaluation (target network).
    """
    with torch.no_grad():
        # Step 1: Online network selects best actions
        next_q_values_online = q_network(next_states)
        best_actions = next_q_values_online.argmax(dim=1)
        
        # Step 2: Target network evaluates those actions
        next_q_values_target = target_network(next_states)
        max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze()
        
        targets = rewards + (1 - dones) * gamma * max_next_q
    return targets


class DoubleDQN:
    """Double DQN implementation with clear separation of standard vs double Q-learning."""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Create networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
        # Metrics for comparison
        self.q_values_history = []
        self.losses = []
        
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            self.q_values_history.append(q_values.max().item())
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=64, use_double=True):
        """
        Training step comparing standard vs double Q-learning.
        
        The key difference is in how we compute the target Q-values.
        """
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        # Current Q-values for actions taken
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute targets - this is where Double DQN differs!
        if use_double:
            targets = compute_double_dqn_targets(
                self.q_network, self.target_network, next_states, rewards, dones, self.gamma
            )
        else:
            targets = compute_dqn_targets(
                self.q_network, self.target_network, next_states, rewards, dones, self.gamma
            )
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)  # Gradient clipping
        self.optimizer.step()
        
        self.losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from online to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def demonstrate_overestimation(num_episodes=300):
    """
    Compare standard DQN vs Double DQN to show overestimation differences.
    We'll use a simple environment where we can track true Q-values.
    """
    import gym
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Train both variants
    results = {}
    
    for use_double in [False, True]:
        print(f"\nTraining {'Double' if use_double else 'Standard'} DQN...")
        agent = DoubleDQN(state_dim, action_dim)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Select action
                action = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, _ = env.step(action)
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train
                if len(agent.memory) >= 64:
                    agent.train_step(batch_size=64, use_double=use_double)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Post-episode updates
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Update target network periodically
            if episode % 10 == 0:
                agent.update_target_network()
                if episode % 50 == 0:
                    print(f"  Episode {episode}: Avg Reward = {np.mean(episode_rewards[-50:]):.2f}, "
                          f"Avg Q-value = {np.mean(agent.q_values_history[-1000:]):.2f}")
        
        variant_name = "Double DQN" if use_double else "Standard DQN"
        results[variant_name] = {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'q_values': agent.q_values_history,
            'losses': agent.losses
        }
    
    # Create comprehensive comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Learning curves
    ax = axes[0, 0]
    for variant, data in results.items():
        rewards_smooth = np.convolve(data['rewards'], np.ones(20)/20, mode='valid')
        ax.plot(rewards_smooth, label=variant, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('20-Episode Average Reward')
    ax.set_title('Learning Performance Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Q-value evolution (showing overestimation)
    ax = axes[0, 1]
    for variant, data in results.items():
        # Sample Q-values for cleaner plot
        q_vals = data['q_values']
        if len(q_vals) > 1000:
            indices = np.linspace(0, len(q_vals)-1, 1000, dtype=int)
            q_vals = [q_vals[i] for i in indices]
        ax.plot(q_vals, label=variant, alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Average Q-value')
    ax.set_title('Q-value Evolution (Overestimation Analysis)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Direct overestimation comparison
    ax = axes[1, 0]
    # Calculate overestimation as difference from optimal
    # For CartPole, theoretical max is ~500
    for variant, data in results.items():
        q_vals = data['q_values']
        overestimation = [max(0, q - 500) for q in q_vals]
        if len(overestimation) > 1000:
            indices = np.linspace(0, len(overestimation)-1, 1000, dtype=int)
            overestimation = [overestimation[i] for i in indices]
        ax.plot(overestimation, label=variant, alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Overestimation (Q - Theoretical Max)')
    ax.set_title('Overestimation Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # 4. Loss comparison
    ax = axes[1, 1]
    for variant, data in results.items():
        losses = data['losses']
        if len(losses) > 1000:
            indices = np.linspace(0, len(losses)-1, 1000, dtype=int)
            losses = [losses[i] for i in indices]
        ax.plot(losses, label=variant, alpha=0.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.suptitle('Double DQN vs Standard DQN: Comprehensive Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('double_dqn_comparison.png', dpi=150)
    plt.show()
    
    env.close()
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("Final Performance Statistics (last 50 episodes):")
    print("="*60)
    for variant, data in results.items():
        final_reward = np.mean(data['rewards'][-50:])
        final_q = np.mean(data['q_values'][-1000:]) if data['q_values'] else 0
        max_q = np.max(data['q_values']) if data['q_values'] else 0
        print(f"\n{variant}:")
        print(f"  Average Reward: {final_reward:.2f}")
        print(f"  Average Q-value: {final_q:.2f}")
        print(f"  Maximum Q-value reached: {max_q:.2f}")
        print(f"  Overestimation: {max(0, max_q - 500):.2f}")
    
    return results


def visualize_target_computation():
    """
    Visualize the difference between standard and double DQN target computation.
    """
    # Create example data
    batch_size = 5
    action_dim = 4
    
    # Random Q-values for visualization
    torch.manual_seed(42)
    online_q = torch.randn(batch_size, action_dim) * 2 + 5
    target_q = online_q + torch.randn(batch_size, action_dim) * 0.5  # Slightly different
    
    # Standard DQN computation
    standard_actions = target_q.argmax(dim=1)
    standard_values = target_q.max(dim=1)[0]
    
    # Double DQN computation
    double_actions = online_q.argmax(dim=1)
    double_values = target_q.gather(1, double_actions.unsqueeze(1)).squeeze()
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Standard DQN
    x = np.arange(batch_size)
    width = 0.2
    
    for a in range(action_dim):
        values = target_q[:, a].numpy()
        ax1.bar(x + a*width, values, width, label=f'Action {a}', alpha=0.7)
    
    # Mark selected actions
    for i, a in enumerate(standard_actions):
        ax1.plot(i + a*width + width/2, standard_values[i], 'r*', markersize=15)
    
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Q-value')
    ax1.set_title('Standard DQN: Target Network Selects and Evaluates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Double DQN
    for a in range(action_dim):
        online_vals = online_q[:, a].numpy()
        target_vals = target_q[:, a].numpy()
        ax2.bar(x + a*width - width/2, online_vals, width/2, 
               label=f'Online Q({a})', alpha=0.5)
        ax2.bar(x + a*width, target_vals, width/2, 
               label=f'Target Q({a})', alpha=0.5)
    
    # Mark selections
    for i, a in enumerate(double_actions):
        ax2.plot(i + a*width - width/4, online_q[i, a], 'g^', markersize=10)  # Selection
        ax2.plot(i + a*width + width/4, double_values[i], 'r*', markersize=15)  # Evaluation
    
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Q-value')
    ax2.set_title('Double DQN: Online Selects (green), Target Evaluates (red)')
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('double_dqn_mechanism.png', dpi=150)
    plt.show()
    
    # Print comparison
    print("Target Value Comparison:")
    print("-" * 40)
    print(f"{'Sample':<10} {'Standard':<15} {'Double':<15} {'Difference':<15}")
    print("-" * 40)
    for i in range(batch_size):
        diff = standard_values[i].item() - double_values[i].item()
        print(f"{i:<10} {standard_values[i].item():<15.3f} "
              f"{double_values[i].item():<15.3f} {diff:<15.3f}")
    
    print(f"\nAverage overestimation: {(standard_values - double_values).mean().item():.3f}")


if __name__ == "__main__":
    print("=== Double DQN Demonstration ===\n")
    
    # First show the mechanism
    print("1. Visualizing Double DQN Mechanism:")
    visualize_target_computation()
    
    # Then demonstrate on CartPole
    print("\n2. Comparing Standard vs Double DQN on CartPole:")
    results = demonstrate_overestimation(num_episodes=300)