"""
Q-Learning Agent Implementation

This module implements various Q-learning agents, from basic to advanced,
demonstrating the progression of features that make Q-learning practical.
"""

import numpy as np
from collections import defaultdict, deque
import random


class BasicQLearningAgent:
    """
    A minimal Q-learning implementation to demonstrate the core algorithm.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: maps (state, action) pairs to values
        self.Q = defaultdict(lambda: defaultdict(float))
        
    def get_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # Choose best action, breaking ties randomly
            q_values = [self.Q[state][a] for a in range(self.env.action_space.n)]
            max_q = max(q_values) if q_values else 0
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values based on experience."""
        # Current Q-value
        current_q = self.Q[state][action]
        
        # Best Q-value for next state
        if done:
            max_next_q = 0  # Terminal state has no future
        else:
            next_q_values = [self.Q[next_state][a] 
                           for a in range(self.env.action_space.n)]
            max_next_q = max(next_q_values) if next_q_values else 0
        
        # Q-learning update
        target = reward + self.gamma * max_next_q
        self.Q[state][action] += self.alpha * (target - current_q)
    
    def train_episode(self):
        """Run one episode of training."""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action
            action = self.get_action(state)
            
            # Take action
            next_state, reward, done = self.env.step(action)
            
            # Learn from experience
            self.learn(state, action, reward, next_state, done)
            
            # Update tracking
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
                
        return total_reward, steps


class TreasureHuntQLearning:
    """
    An advanced Q-learning agent with features for practical learning.
    
    Enhancements over basic agent:
    - Optimistic initialization for exploration
    - Epsilon decay for exploration scheduling
    - State visit tracking for analysis
    - Experience replay for sample efficiency
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Optimistic initialization encourages exploration
        # By starting with high values, the agent will try each action
        # at least once before settling on exploitation
        self.Q = defaultdict(lambda: defaultdict(lambda: 5.0))
        
        # Tracking for analysis
        self.state_visits = defaultdict(int)
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Experience replay buffer
        self.memory = deque(maxlen=5000)
        
    def get_action(self, state, training=True):
        """ε-greedy action selection with decay."""
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # Break ties randomly for equal Q-values
            q_values = [self.Q[state][a] for a in range(self.env.action_space.n)]
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Experience replay to improve sample efficiency."""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * max(self.Q[next_state].values())
            
            # Smaller learning rate for replay
            replay_alpha = self.alpha * 0.5
            self.Q[state][action] += replay_alpha * (target - self.Q[state][action])
    
    def train_episode(self):
        """Train for one episode."""
        state = self.env.reset()
        done = False
        
        total_reward = 0
        steps = 0
        
        while not done:
            # Choose and execute action
            action = self.get_action(state)
            next_state, reward, done = self.env.step(action)
            
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.gamma * max(self.Q[next_state].values())
            
            td_error = target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error
            
            # Store experience
            self.remember(state, action, reward, next_state, done)
            
            # Update statistics
            self.state_visits[state] += 1
            total_reward += reward
            steps += 1
            
            state = next_state
        
        # Experience replay
        self.replay()
        
        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_reward, steps
    
    def train(self, num_episodes, verbose=True):
        """Full training loop."""
        for episode in range(num_episodes):
            reward, steps = self.train_episode()
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(steps)
            
            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                print(f"Episode {episode}:")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Length: {avg_length:.1f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                print(f"  States Discovered: {len(self.Q)}")
    
    def visualize_learning(self):
        """Plot learning curves and statistics."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards over time
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) > 100:
            smoothed = np.convolve(self.episode_rewards, 
                                  np.ones(100)/100, mode='valid')
            axes[0, 0].plot(smoothed, label='100-Episode Average', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Learning Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.3, label='Episode Length')
        if len(self.episode_lengths) > 100:
            smoothed = np.convolve(self.episode_lengths,
                                  np.ones(100)/100, mode='valid')
            axes[0, 1].plot(smoothed, label='100-Episode Average', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].set_title('Episode Length Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # State visit distribution
        visit_counts = list(self.state_visits.values())
        axes[1, 0].hist(visit_counts, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Visit Count')
        axes[1, 0].set_ylabel('Number of States')
        axes[1, 0].set_title('State Visit Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-value distribution
        all_q_values = []
        for state_q in self.Q.values():
            all_q_values.extend(state_q.values())
        
        axes[1, 1].hist(all_q_values, bins=50, alpha=0.7, color='blue')
        axes[1, 1].set_xlabel('Q-value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Q-value Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def demonstrate_policy(self, max_steps=100, render=True):
        """Show the learned policy in action."""
        state = self.env.reset()
        done = False
        steps = 0
        
        if render:
            self.env.render(self.Q)
            import matplotlib.pyplot as plt
            plt.show()
        
        while not done and steps < max_steps:
            action = self.get_action(state, training=False)
            state, reward, done = self.env.step(action)
            steps += 1
            
            if render:
                plt.clf()
                self.env.render(self.Q)
                plt.pause(0.5)
        
        return self.env.total_reward, steps


def debug_q_learning(agent, env, num_episodes=10):
    """Diagnostic tool for Q-learning issues."""
    
    print("=== Q-Learning Diagnostics ===")
    
    # 1. Check exploration
    actions_taken = defaultdict(int)
    for _ in range(100):
        state = env.reset()
        action = agent.get_action(state)
        actions_taken[action] += 1
    
    print(f"\nAction distribution: {dict(actions_taken)}")
    if min(actions_taken.values()) == 0:
        print("WARNING: Some actions never explored!")
    
    # 2. Check Q-value magnitudes
    all_q_values = []
    for state_q in agent.Q.values():
        all_q_values.extend(state_q.values())
    
    if all_q_values:
        print(f"\nQ-value statistics:")
        print(f"  Mean: {np.mean(all_q_values):.3f}")
        print(f"  Std:  {np.std(all_q_values):.3f}")
        print(f"  Min:  {np.min(all_q_values):.3f}")
        print(f"  Max:  {np.max(all_q_values):.3f}")
        
        if np.std(all_q_values) < 0.01:
            print("WARNING: Q-values not differentiating!")
    
    # 3. Check learning progress
    initial_performance = evaluate_policy(agent, env, episodes=10)
    agent.train(100, verbose=False)
    final_performance = evaluate_policy(agent, env, episodes=10)
    
    print(f"\nPerformance improvement: {initial_performance:.2f} -> {final_performance:.2f}")
    if final_performance <= initial_performance:
        print("WARNING: No improvement detected!")
    
    # 4. Visualize Q-values for specific states
    if hasattr(env, 'get_test_states'):
        test_states = env.get_test_states()
        print("\nQ-values for test states:")
        for state in test_states:
            q_vals = [agent.Q[state][a] for a in range(env.action_space.n)]
            print(f"  State {state[0]}: {[f'{q:.2f}' for q in q_vals]}")
            print(f"    Best action: {np.argmax(q_vals)}")


def evaluate_policy(agent, env, episodes=10):
    """Evaluate an agent's policy over multiple episodes."""
    total_rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(state, training=False)
            state, reward, done = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


if __name__ == "__main__":
    # Example usage
    from treasure_hunt_env import TreasureHuntEnv
    
    # Create environment
    env = TreasureHuntEnv(width=6, height=6, num_treasures=2, num_traps=3)
    
    # Test basic agent
    print("Testing Basic Q-Learning Agent:")
    basic_agent = BasicQLearningAgent(env)
    
    # Train for a few episodes
    for i in range(10):
        reward, steps = basic_agent.train_episode()
        print(f"Episode {i}: Reward = {reward:.2f}, Steps = {steps}")
    
    # Test advanced agent
    print("\n\nTesting Advanced Q-Learning Agent:")
    advanced_agent = TreasureHuntQLearning(env)
    advanced_agent.train(num_episodes=500)
    
    # Visualize results
    advanced_agent.visualize_learning()
    
    # Run diagnostics
    debug_q_learning(advanced_agent, env)