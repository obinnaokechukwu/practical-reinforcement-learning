import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dqn import DQNAgent, DQNNetwork, train_dqn
import gym
from collections import deque
import random


class UnstableQLearning:
    """Demonstrates instability in naive neural Q-learning."""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=32):
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.gamma = 0.99
        
    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Current Q-value
        current_q_values = self.q_network(state_tensor)
        current_q = current_q_values[0, action]
        
        # Target Q-value (SOURCE OF INSTABILITY)
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            max_next_q = next_q_values.max(1)[0]
            target_q = reward + (1 - done) * self.gamma * max_next_q
        
        # Loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return current_q.item(), target_q.item()


def demonstrate_instability():
    """Show Q-value divergence without target networks."""
    print("=== Experiment: Neural Q-Learning Instability ===")
    
    agent = UnstableQLearning()
    
    q_values = []
    target_values = []
    
    # Correlated sequence: same state-action repeatedly
    state = np.array([1.0, 0.0, 0.0, 0.0])
    action = 0
    
    for step in range(1000):
        # Small positive reward
        reward = 0.1
        next_state = state + np.random.normal(0, 0.01, 4)  # Slight variation
        
        q, target = agent.update(state, action, reward, next_state, False)
        q_values.append(q)
        target_values.append(target)
        
        state = next_state
    
    # Plot divergence
    plt.figure(figsize=(10, 6))
    plt.plot(q_values, label='Q-values', alpha=0.7)
    plt.plot(target_values, label='Target values', alpha=0.7)
    plt.xlabel('Update Step')
    plt.ylabel('Value')
    plt.title('Q-Value Divergence in Naive Neural Q-Learning')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('q_value_divergence.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Final Q-value: {q_values[-1]:.2e}")
    print(f"Q-value explosion factor: {q_values[-1] / q_values[0]:.2e}")


def compare_replay_strategies():
    """Compare uniform vs prioritized experience replay."""
    print("\n=== Experiment: Experience Replay Comparison ===")
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Train with uniform replay
    print("\nTraining with Uniform Experience Replay...")
    agent_uniform = DQNAgent(state_dim, action_dim, use_prioritized_replay=False)
    uniform_rewards = []
    
    for episode in range(200):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent_uniform.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent_uniform.store_transition(state, action, reward, next_state, done)
            
            if len(agent_uniform.memory) > agent_uniform.batch_size:
                agent_uniform.train_step()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent_uniform.decay_epsilon()
        uniform_rewards.append(total_reward)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(uniform_rewards[-50:]):.2f}")
    
    # Train with prioritized replay
    print("\nTraining with Prioritized Experience Replay...")
    agent_prioritized = DQNAgent(state_dim, action_dim, use_prioritized_replay=True)
    prioritized_rewards = []
    
    for episode in range(200):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent_prioritized.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent_prioritized.store_transition(state, action, reward, next_state, done)
            
            if len(agent_prioritized.memory) > agent_prioritized.batch_size:
                agent_prioritized.train_step()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent_prioritized.decay_epsilon()
        prioritized_rewards.append(total_reward)
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(prioritized_rewards[-50:]):.2f}")
    
    env.close()
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Smooth rewards
    window = 20
    uniform_smooth = np.convolve(uniform_rewards, np.ones(window)/window, mode='valid')
    prioritized_smooth = np.convolve(prioritized_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(uniform_smooth, label='Uniform Replay', linewidth=2, alpha=0.8)
    plt.plot(prioritized_smooth, label='Prioritized Replay', linewidth=2, alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Uniform vs Prioritized Experience Replay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('replay_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def ablation_study():
    """Ablation study: impact of each DQN component."""
    print("\n=== Experiment: DQN Component Ablation ===")
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    configurations = {
        'Full DQN': {'use_replay': True, 'use_target': True},
        'No Target Network': {'use_replay': True, 'use_target': False},
        'No Experience Replay': {'use_replay': False, 'use_target': True},
        'Vanilla Neural Q': {'use_replay': False, 'use_target': False}
    }
    
    results = {}
    
    for config_name, config in configurations.items():
        print(f"\nTesting: {config_name}")
        
        # Modified agent for ablation
        class AblationAgent(DQNAgent):
            def __init__(self, state_dim, action_dim, use_replay=True, use_target=True):
                super().__init__(state_dim, action_dim, use_prioritized_replay=False)
                self.use_replay = use_replay
                self.use_target = use_target
                
                if not use_replay:
                    self.memory = deque(maxlen=1)  # Only store latest transition
                
            def train_step(self):
                if len(self.memory) < 1:
                    return
                
                # Get batch
                if self.use_replay and len(self.memory) >= self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                else:
                    batch = list(self.memory)
                
                states = torch.FloatTensor([t[0] for t in batch])
                actions = torch.LongTensor([t[1] for t in batch])
                rewards = torch.FloatTensor([t[2] for t in batch])
                next_states = torch.FloatTensor([t[3] for t in batch])
                dones = torch.FloatTensor([t[4] for t in batch])
                
                # Current Q values
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                
                # Target Q values
                with torch.no_grad():
                    if self.use_target:
                        next_q_values = self.target_network(next_states).max(1)[0]
                    else:
                        next_q_values = self.q_network(next_states).max(1)[0]
                    
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
                
                # Loss
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
                self.optimizer.step()
                
                self.losses.append(loss.item())
                
                # Update target network
                if self.use_target:
                    self.update_count += 1
                    if self.update_count % self.target_update == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Train agent
        agent = AblationAgent(state_dim, action_dim, **config)
        rewards = []
        
        for episode in range(150):
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
            rewards.append(total_reward)
        
        results[config_name] = rewards
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    for config_name, rewards in results.items():
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed, label=config_name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('DQN Component Ablation Study')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ablation_study.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print final performance
    print("\nFinal Performance (last 20 episodes):")
    for config_name, rewards in results.items():
        final_avg = np.mean(rewards[-20:])
        print(f"{config_name}: {final_avg:.2f}")


def target_network_update_frequency():
    """Study impact of target network update frequency."""
    print("\n=== Experiment: Target Network Update Frequency ===")
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    update_frequencies = [1, 5, 10, 50, 100]
    results = {}
    
    for freq in update_frequencies:
        print(f"\nTraining with target update frequency: {freq}")
        
        agent = DQNAgent(state_dim, action_dim, target_update=freq)
        rewards = []
        
        for episode in range(200):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                
                if len(agent.memory) > agent.batch_size:
                    agent.train_step()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            agent.decay_epsilon()
            rewards.append(total_reward)
        
        results[freq] = rewards
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    for freq, rewards in results.items():
        smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
        plt.plot(smoothed, label=f'Update every {freq} steps', linewidth=2, alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Impact of Target Network Update Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('target_update_frequency.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_q_function():
    """Visualize learned Q-function for CartPole."""
    print("\n=== Visualization: Learned Q-Function ===")
    
    # Load trained agent
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    # Train briefly for visualization
    print("Training agent for visualization...")
    for episode in range(100):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            if len(agent.memory) > agent.batch_size:
                agent.train_step()
            
            state = next_state
            if done:
                break
        agent.decay_epsilon()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Fix cart position and velocity, vary pole angle and angular velocity
    cart_pos = 0.0
    cart_vel = 0.0
    
    angles = np.linspace(-0.2, 0.2, 50)
    angular_vels = np.linspace(-2, 2, 50)
    
    for action_idx in range(2):
        Q_values = np.zeros((len(angular_vels), len(angles)))
        
        for i, angular_vel in enumerate(angular_vels):
            for j, angle in enumerate(angles):
                state = np.array([cart_pos, cart_vel, angle, angular_vel])
                with torch.no_grad():
                    q_vals = agent.q_network(torch.FloatTensor(state).unsqueeze(0))
                Q_values[i, j] = q_vals[0, action_idx].item()
        
        # Plot heatmap
        im = axes[0, action_idx].imshow(Q_values, extent=[angles[0], angles[-1], 
                                                          angular_vels[0], angular_vels[-1]],
                                       aspect='auto', origin='lower', cmap='viridis')
        axes[0, action_idx].set_xlabel('Pole Angle (rad)')
        axes[0, action_idx].set_ylabel('Angular Velocity (rad/s)')
        axes[0, action_idx].set_title(f'Q-values for Action {action_idx}')
        plt.colorbar(im, ax=axes[0, action_idx])
    
    # Plot Q-value difference (action preference)
    with torch.no_grad():
        Q_diff = np.zeros((len(angular_vels), len(angles)))
        for i, angular_vel in enumerate(angular_vels):
            for j, angle in enumerate(angles):
                state = np.array([cart_pos, cart_vel, angle, angular_vel])
                q_vals = agent.q_network(torch.FloatTensor(state).unsqueeze(0))
                Q_diff[i, j] = q_vals[0, 1].item() - q_vals[0, 0].item()
    
    im = axes[1, 0].imshow(Q_diff, extent=[angles[0], angles[-1], 
                                           angular_vels[0], angular_vels[-1]],
                          aspect='auto', origin='lower', cmap='RdBu', vmin=-5, vmax=5)
    axes[1, 0].set_xlabel('Pole Angle (rad)')
    axes[1, 0].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 0].set_title('Q(s,1) - Q(s,0): Action Preference')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot optimal actions
    optimal_actions = (Q_diff > 0).astype(int)
    im = axes[1, 1].imshow(optimal_actions, extent=[angles[0], angles[-1], 
                                                    angular_vels[0], angular_vels[-1]],
                          aspect='auto', origin='lower', cmap='RdYlBu')
    axes[1, 1].set_xlabel('Pole Angle (rad)')
    axes[1, 1].set_ylabel('Angular Velocity (rad/s)')
    axes[1, 1].set_title('Optimal Actions (0=Left, 1=Right)')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('q_function_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    env.close()


if __name__ == "__main__":
    # Run all experiments
    print("Running DQN Experiments\n" + "="*50)
    
    # Demonstrate instability
    demonstrate_instability()
    
    # Compare replay strategies
    compare_replay_strategies()
    
    # Ablation study
    ablation_study()
    
    # Target network frequency
    target_network_update_frequency()
    
    # Visualize Q-function
    visualize_q_function()
    
    print("\n" + "="*50)
    print("All experiments completed! Check the generated PNG files.")