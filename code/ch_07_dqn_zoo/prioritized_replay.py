"""
Prioritized Experience Replay Implementation

Demonstrates how to sample experiences based on their TD error magnitude,
focusing learning on surprising or difficult transitions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import gym


class SumTree:
    """
    Binary tree data structure where parent nodes are the sum of children.
    
    This allows us to efficiently sample based on priority values.
    Each leaf contains a priority value and its associated data.
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree structure: parent nodes are sum of children
        # Size is 2*capacity - 1 to store both leaves and internal nodes
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0  # Position for next write
        
    def _propagate(self, idx, change):
        """Propagate priority changes up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """Find leaf index for given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1
        
        # If we're at a leaf
        if left >= len(self.tree):
            return idx
        
        # Navigate down the tree based on cumulative sum
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """Get total priority sum."""
        return self.tree[0]
    
    def add(self, priority, data):
        """Add new data with given priority."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """Update priority of existing node."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """Get data for given cumulative sum."""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Experience replay buffer that samples based on TD error priorities.
    
    Key concepts:
    - Priority = |TD error|^α (α controls prioritization strength)
    - Importance sampling weights correct for bias: w = (N·P)^(-β)
    - β anneals from β_start to 1 over training
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha  # How much to prioritize (0=uniform, 1=full priority)
        self.beta_start = beta_start  # Initial importance sampling correction
        self.beta_frames = beta_frames  # Frames to anneal beta to 1
        self.frame = 1  # Current frame for beta annealing
        
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_priority = 1.0  # Running max priority for new experiences
        
    def beta(self):
        """Anneal beta from beta_start to 1.0 over training."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, transition):
        """Store transition with maximum priority (optimistic initialization)."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size):
        """
        Sample batch based on priorities.
        
        Returns:
            batch: Sampled transitions
            idxs: Tree indices (for updating priorities later)
            weights: Importance sampling weights
        """
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size  # Divide priority sum into segments
        priorities = []
        
        self.frame += 1
        
        # Sample one transition from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # Sample uniformly within segment
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta())
        is_weight /= is_weight.max()  # Normalize for stability
        
        return batch, idxs, torch.FloatTensor(is_weight)
    
    def update_priorities(self, idxs, td_errors):
        """
        Update priorities based on new TD errors.
        
        Priority = (|TD error| + ε)^α
        where ε is a small constant for stability.
        """
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class PrioritizedDQN:
    """
    DQN agent with prioritized experience replay.
    
    Key differences from standard DQN:
    - Samples experiences based on TD error magnitude
    - Uses importance sampling weights in loss calculation
    - Updates priorities after each training step
    """
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, alpha=0.6, beta_start=0.4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Networks
        self.q_network = self._build_network(state_dim, action_dim)
        self.target_network = self._build_network(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha, beta_start)
        
        # Metrics
        self.losses = []
        self.td_errors = []
        self.priorities = []
    
    def _build_network(self, input_dim, output_dim, hidden_dim=128):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in prioritized buffer."""
        self.memory.push((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=64):
        """
        Training step with prioritized replay.
        
        Key differences:
        1. Sample based on priorities
        2. Use importance sampling weights in loss
        3. Update priorities with new TD errors
        """
        if self.memory.tree.n_entries < batch_size:
            return None
        
        # Sample with priorities
        batch, idxs, weights = self.memory.sample(batch_size)
        
        # Unpack batch
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN targets
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Calculate TD errors for priority updates
        td_errors = targets - current_q_values.squeeze()
        
        # Weighted loss (importance sampling)
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(idxs, td_errors.detach().numpy())
        
        # Track metrics
        self.losses.append(loss.item())
        self.td_errors.extend(td_errors.detach().numpy())
        self.priorities.append(np.mean([self.memory.tree.tree[idx] for idx in idxs]))
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def visualize_priority_distribution(agent):
    """Visualize the distribution of priorities in the replay buffer."""
    # Get all priorities
    priorities = []
    for i in range(agent.memory.tree.n_entries):
        idx = i + agent.memory.capacity - 1
        priorities.append(agent.memory.tree.tree[idx])
    
    priorities = np.array(priorities)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Priority distribution
    ax1.hist(priorities, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Priority Value')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Experience Priorities')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_priorities = np.sort(priorities)[::-1]
    cumsum = np.cumsum(sorted_priorities)
    cumsum = cumsum / cumsum[-1]
    
    ax2.plot(np.arange(len(cumsum)) / len(cumsum) * 100, cumsum, linewidth=2)
    ax2.set_xlabel('Top X% of Experiences')
    ax2.set_ylabel('Cumulative Priority')
    ax2.set_title('Cumulative Priority Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add lines showing concentration
    top_10_idx = int(0.1 * len(cumsum))
    ax2.axvline(10, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(cumsum[top_10_idx], color='red', linestyle='--', alpha=0.5)
    ax2.text(15, cumsum[top_10_idx] - 0.05, 
             f'Top 10% have {cumsum[top_10_idx]:.1%} of total priority',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('priority_distribution.png', dpi=150)
    plt.show()
    
    return priorities


def compare_replay_strategies():
    """
    Compare uniform vs prioritized replay on a task with rare important events.
    """
    # Create a simple environment with rare rewards
    class RareRewardEnv:
        """
        Environment where most transitions give small negative rewards,
        but rare transitions give large positive rewards.
        """
        def __init__(self):
            self.state = 0
            self.max_steps = 100
            self.steps = 0
            self.rare_state = 50  # State with rare high reward
            
        def reset(self):
            self.state = 0
            self.steps = 0
            return np.array([self.state / 100.0])  # Normalize state
        
        def step(self, action):
            self.steps += 1
            
            # Rare high-reward transition
            if self.state == self.rare_state and action == 1:
                reward = 10.0
                self.state = 0
            # Common transitions
            elif action == 0:
                self.state = min(self.state + 1, 100)
                reward = -0.1
            else:
                self.state = max(self.state - 1, 0)
                reward = -0.1
            
            done = self.steps >= self.max_steps
            return np.array([self.state / 100.0]), reward, done, {}
    
    # Compare both replay strategies
    results = {}
    
    for use_prioritized in [False, True]:
        print(f"\nTraining with {'Prioritized' if use_prioritized else 'Uniform'} Replay")
        
        env = RareRewardEnv()
        
        if use_prioritized:
            from double_dqn import DoubleDQN
            agent = PrioritizedDQN(state_dim=1, action_dim=2)
        else:
            # Standard DQN for comparison
            from double_dqn import DoubleDQN
            agent = DoubleDQN(state_dim=1, action_dim=2)
        
        episode_rewards = []
        rare_rewards_found = []  # Track when rare reward is discovered
        td_error_stats = []
        
        for episode in range(200):
            state = env.reset()
            total_reward = 0
            found_rare = False
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                if reward == 10.0:
                    found_rare = True
                
                agent.store_transition(state, action, reward, next_state, done)
                
                if hasattr(agent, 'memory') and ((use_prioritized and agent.memory.tree.n_entries >= 64) or 
                                                (not use_prioritized and len(agent.memory) >= 64)):
                    agent.train_step()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            agent.decay_epsilon()
            episode_rewards.append(total_reward)
            rare_rewards_found.append(found_rare)
            
            if episode % 10 == 0:
                agent.update_target_network()
                if episode % 50 == 0:
                    print(f"  Episode {episode}: Avg Reward = {np.mean(episode_rewards[-50:]):.2f}")
        
        replay_type = "Prioritized" if use_prioritized else "Uniform"
        results[replay_type] = {
            'rewards': episode_rewards,
            'found_rare': rare_rewards_found,
            'agent': agent if use_prioritized else None
        }
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning curves
    ax = axes[0, 0]
    for replay_type, data in results.items():
        smooth_rewards = np.convolve(data['rewards'], np.ones(10)/10, mode='valid')
        ax.plot(smooth_rewards, label=replay_type, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('10-Episode Average Reward')
    ax.set_title('Learning Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative rare transitions found
    ax = axes[0, 1]
    for replay_type, data in results.items():
        cumulative_found = np.cumsum(data['found_rare'])
        ax.plot(cumulative_found, label=replay_type, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Rare Rewards Found')
    ax.set_title('Discovery of High-Reward Transitions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # TD error evolution (prioritized only)
    if results['Prioritized']['agent'] is not None:
        agent = results['Prioritized']['agent']
        if agent.td_errors:
            ax = axes[1, 0]
            td_errors_abs = np.abs(agent.td_errors)
            # Sample for plotting
            if len(td_errors_abs) > 1000:
                indices = np.linspace(0, len(td_errors_abs)-1, 1000, dtype=int)
                td_errors_plot = [td_errors_abs[i] for i in indices]
            else:
                td_errors_plot = td_errors_abs
            
            ax.plot(td_errors_plot, alpha=0.5, color='red')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('|TD Error|')
            ax.set_title('TD Error Magnitude Over Training (Prioritized)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    # Priority evolution
    if results['Prioritized']['agent'] is not None:
        agent = results['Prioritized']['agent']
        if agent.priorities:
            ax = axes[1, 1]
            ax.plot(agent.priorities, alpha=0.7, color='green')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Average Priority')
            ax.set_title('Average Sampled Priority Over Time')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Uniform vs Prioritized Experience Replay Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('replay_comparison.png', dpi=150)
    plt.show()
    
    # Statistics
    print("\nFinal Statistics:")
    for replay_type, data in results.items():
        total_rare = sum(data['found_rare'])
        avg_reward = np.mean(data['rewards'][-20:])
        print(f"{replay_type}: Found rare transition {total_rare} times, "
              f"Final avg reward: {avg_reward:.2f}")
    
    # Show priority distribution for prioritized agent
    if results['Prioritized']['agent'] is not None:
        print("\nAnalyzing priority distribution...")
        visualize_priority_distribution(results['Prioritized']['agent'])
    
    return results


if __name__ == "__main__":
    print("=== Prioritized Experience Replay Demonstration ===\n")
    
    # Compare replay strategies
    results = compare_replay_strategies()