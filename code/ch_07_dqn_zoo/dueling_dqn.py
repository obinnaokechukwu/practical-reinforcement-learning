"""
Dueling DQN Implementation

Demonstrates the dueling architecture that separates state value V(s)
from action advantages A(s,a).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import gym


class DuelingNetwork(nn.Module):
    """
    Dueling network architecture that separates state value and action advantages.
    
    Key insight: Q(s,a) = V(s) + A(s,a)
    where V(s) is the value of being in state s, and A(s,a) is the 
    advantage of taking action a over other actions.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingNetwork, self).__init__()
        
        # Shared feature extraction layers
        # These layers process the state into useful features
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream: estimates V(s)
        # This represents how good it is to be in this state
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value output
        )
        
        # Advantage stream: estimates A(s,a) for each action
        # This represents the relative advantage of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # One output per action
        )
        
    def forward(self, x):
        """Forward pass combining value and advantages into Q-values."""
        # Extract features
        features = self.feature_layer(x)
        
        # Compute value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine using the dueling formula
        # We subtract the mean advantage to ensure identifiability
        # This forces the advantages to be relative to the average
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_value_advantage(self, x):
        """Get separate value and advantage components for analysis."""
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value, advantages


class DuelingDQN:
    """Complete Dueling DQN implementation."""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Use dueling architecture for both networks
        self.q_network = DuelingNetwork(state_dim, action_dim)
        self.target_network = DuelingNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
        # Metrics
        self.value_history = []
        self.advantage_history = []
        self.losses = []
    
    def select_action(self, state, training=True):
        """Select action and track value/advantage decomposition."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            # Also track the decomposition
            value, advantages = self.q_network.get_value_advantage(state_tensor)
            self.value_history.append(value.item())
            self.advantage_history.append(advantages.squeeze().numpy())
            
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=64):
        """Training step for Dueling DQN."""
        if len(self.memory) < batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN target computation
        with torch.no_grad():
            # Online network selects actions
            next_actions = self.q_network(next_states).argmax(dim=1)
            # Target network evaluates them
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss and optimization
        loss = nn.MSELoss()(current_q_values.squeeze(), targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        return loss.item()
    
    def update_target_network(self):
        """Update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def analyze_decomposition(self, states):
        """Analyze value/advantage decomposition for given states."""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            values, advantages = self.q_network.get_value_advantage(states_tensor)
            q_values = self.q_network(states_tensor)
            
        return {
            'values': values.numpy(),
            'advantages': advantages.numpy(),
            'q_values': q_values.numpy()
        }


def visualize_dueling_decomposition(env_name='CartPole-v1'):
    """Visualize how Dueling DQN decomposes Q into V and A."""
    # Train a Dueling DQN
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DuelingDQN(state_dim, action_dim)
    
    # Quick training
    print("Training Dueling DQN...")
    for episode in range(200):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.memory) > 64:
                agent.train_step()
            
            state = next_state
            if done:
                break
        
        agent.decay_epsilon()
        if episode % 10 == 0:
            agent.update_target_network()
            if episode % 50 == 0:
                print(f"  Episode {episode}")
    
    # Collect states for analysis
    print("\nCollecting states for analysis...")
    states = []
    state = env.reset()
    for _ in range(200):
        states.append(state)
        action = agent.select_action(state, training=False)
        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()
    
    # Analyze decomposition
    analysis = agent.analyze_decomposition(states)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # State values over time
    axes[0, 0].plot(analysis['values'], label='V(s)', linewidth=2, color='blue')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('State Value')
    axes[0, 0].set_title('State Values During Episode')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Advantages for each action
    for a in range(action_dim):
        axes[0, 1].plot(analysis['advantages'][:, a], 
                       label=f'A(s,{a})', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Advantage')
    axes[0, 1].set_title('Action Advantages')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Q-values decomposition at specific time points
    time_points = [50, 100, 150]
    x = np.arange(action_dim)
    width = 0.25
    
    for i, t in enumerate(time_points):
        if t < len(states):
            v = analysis['values'][t, 0]
            a = analysis['advantages'][t]
            q = analysis['q_values'][t]
            
            # Stack bar chart showing decomposition
            axes[1, 0].bar(x + i*width, [v]*action_dim, width, 
                          label=f't={t} V(s)', alpha=0.7)
            axes[1, 0].bar(x + i*width, a, width, bottom=[v]*action_dim,
                          label=f't={t} A(s,a)', alpha=0.7)
    
    axes[1, 0].set_xlabel('Action')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Q = V + A Decomposition')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels([f'Action {i}' for i in range(action_dim)])
    
    # Advantage distribution
    all_advantages = analysis['advantages'].flatten()
    axes[1, 1].hist(all_advantages, bins=50, alpha=0.7, density=True, color='green')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Advantage Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Advantage Distribution (should center around 0)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add text showing centering
    mean_adv = np.mean(all_advantages)
    axes[1, 1].text(0.05, 0.95, f'Mean: {mean_adv:.6f}', 
                   transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Dueling DQN: Value-Advantage Decomposition Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('dueling_decomposition.png', dpi=150)
    plt.show()
    
    env.close()
    
    # Print statistics
    print("\nDecomposition Statistics:")
    print(f"Average V(s): {analysis['values'].mean():.3f}")
    print(f"Average |A(s,a)|: {np.abs(analysis['advantages']).mean():.3f}")
    print(f"A(s,a) centering: {analysis['advantages'].mean():.6f} (should be ≈0)")
    
    return analysis


def compare_architectures(num_episodes=300):
    """Compare standard DQN vs Dueling DQN architecture."""
    import gym
    from double_dqn import DoubleDQN
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    results = {}
    
    # Train standard DQN
    print("Training Standard DQN...")
    standard_agent = DoubleDQN(state_dim, action_dim)
    standard_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = standard_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            standard_agent.store_transition(state, action, reward, next_state, done)
            
            if len(standard_agent.memory) >= 64:
                standard_agent.train_step(use_double=True)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        standard_agent.decay_epsilon()
        standard_rewards.append(total_reward)
        
        if episode % 10 == 0:
            standard_agent.update_target_network()
    
    results['Standard DQN'] = {
        'rewards': standard_rewards,
        'final_q_values': standard_agent.q_values_history[-1000:] if standard_agent.q_values_history else []
    }
    
    # Train Dueling DQN
    print("\nTraining Dueling DQN...")
    dueling_agent = DuelingDQN(state_dim, action_dim)
    dueling_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = dueling_agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            dueling_agent.store_transition(state, action, reward, next_state, done)
            
            if len(dueling_agent.memory) >= 64:
                dueling_agent.train_step()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        dueling_agent.decay_epsilon()
        dueling_rewards.append(total_reward)
        
        if episode % 10 == 0:
            dueling_agent.update_target_network()
    
    results['Dueling DQN'] = {
        'rewards': dueling_rewards,
        'values': dueling_agent.value_history[-1000:] if dueling_agent.value_history else [],
        'advantages': dueling_agent.advantage_history[-1000:] if dueling_agent.advantage_history else []
    }
    
    # Comparison plot
    plt.figure(figsize=(12, 5))
    
    # Learning curves
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        rewards_smooth = np.convolve(data['rewards'], np.ones(20)/20, mode='valid')
        plt.plot(rewards_smooth, label=name, linewidth=2)
    
    plt.xlabel('Episode')
    plt.ylabel('20-Episode Average Reward')
    plt.title('Architecture Comparison: Learning Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Value statistics (Dueling only)
    plt.subplot(1, 2, 2)
    if 'values' in results['Dueling DQN']:
        values = results['Dueling DQN']['values']
        plt.hist(values, bins=50, alpha=0.7, label='State Values V(s)', color='blue')
        plt.axvline(np.mean(values), color='blue', linestyle='--', linewidth=2)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Dueling DQN: State Value Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('architecture_comparison.png', dpi=150)
    plt.show()
    
    env.close()
    
    # Print comparison
    print("\nArchitecture Comparison Results:")
    print("-" * 50)
    for name, data in results.items():
        final_perf = np.mean(data['rewards'][-50:])
        print(f"{name}: Final performance = {final_perf:.2f}")
    
    return results


if __name__ == "__main__":
    print("=== Dueling DQN Demonstration ===\n")
    
    # Visualize the decomposition
    print("1. Analyzing Value-Advantage Decomposition:")
    analysis = visualize_dueling_decomposition()
    
    # Compare architectures
    print("\n2. Comparing Standard vs Dueling Architecture:")
    results = compare_architectures(num_episodes=300)