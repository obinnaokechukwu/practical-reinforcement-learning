import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


class DQNNetwork(nn.Module):
    """Neural network for approximating Q-values."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    Samples experiences based on their TD error magnitude.
    """
    
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, transition, td_error=None):
        """Store transition with priority."""
        if td_error is None:
            # New experiences get max priority
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        """Sample batch based on priorities."""
        if len(self.buffer) == 0:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Extract batch
        batch = [self.buffer[idx] for idx in indices]
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        return (states, actions, rewards, next_states, dones), \
               torch.FloatTensor(weights), indices
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Complete DQN agent with experience replay and target network.
    """
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64, target_update=10,
                 use_prioritized_replay=True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Neural networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.use_prioritized_replay = use_prioritized_replay
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = deque(maxlen=buffer_size)
        
        # Training metrics
        self.losses = []
        self.q_values = []
        self.update_count = 0
        
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            self.q_values.append(q_values.mean().item())
            return q_values.argmax(1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if self.use_prioritized_replay:
            # Calculate TD error for prioritization
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                
                current_q = self.q_network(state_tensor)[0, action]
                next_q = self.target_network(next_state_tensor).max(1)[0]
                target_q = reward + (1 - done) * self.gamma * next_q
                td_error = abs(target_q - current_q).item()
            
            self.memory.push((state, action, reward, next_state, done), td_error)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        if self.use_prioritized_replay:
            batch, weights, indices = self.memory.sample(self.batch_size)
            if batch is None:
                return
            states, actions, rewards, next_states, dones = batch
        else:
            batch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([t[0] for t in batch])
            actions = torch.LongTensor([t[1] for t in batch])
            rewards = torch.FloatTensor([t[2] for t in batch])
            next_states = torch.FloatTensor([t[3] for t in batch])
            dones = torch.FloatTensor([t[4] for t in batch])
            weights = torch.ones(self.batch_size)
            indices = None
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        
        self.optimizer.step()
        
        # Update priorities
        if self.use_prioritized_replay and indices is not None:
            td_errors_np = td_errors.detach().numpy().flatten()
            self.memory.update_priorities(indices, td_errors_np)
        
        # Record metrics
        self.losses.append(loss.item())
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def train_dqn(env_name='CartPole-v1', num_episodes=500, render_freq=100):
    """Train DQN on specified environment."""
    
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_dim, action_dim)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training DQN"):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            agent.train_step()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render
            if episode % render_freq == 0:
                env.render()
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"\nEpisode {episode}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Length: {avg_length:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Loss: {np.mean(agent.losses[-100:]):.4f}" if agent.losses else "N/A")
        
        # Check if solved
        if len(episode_rewards) >= 100:
            if np.mean(episode_rewards[-100:]) >= 195.0:  # CartPole-v1 solved
                print(f"\nEnvironment solved in {episode} episodes!")
                break
    
    env.close()
    
    return agent, episode_rewards, episode_lengths


def visualize_training(episode_rewards, episode_lengths, agent):
    """Visualize training progress."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue')
    if len(episode_rewards) > 20:
        smoothed = np.convolve(episode_rewards, np.ones(20)/20, mode='valid')
        axes[0, 0].plot(smoothed, color='blue', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='green')
    if len(episode_lengths) > 20:
        smoothed = np.convolve(episode_lengths, np.ones(20)/20, mode='valid')
        axes[0, 1].plot(smoothed, color='green', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Length')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss history
    if agent.losses:
        axes[1, 0].plot(agent.losses, alpha=0.5, color='red')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Q-value evolution
    if agent.q_values:
        axes[1, 1].plot(agent.q_values, alpha=0.5, color='purple')
        axes[1, 1].set_xlabel('Action Selection')
        axes[1, 1].set_ylabel('Average Q-value')
        axes[1, 1].set_title('Q-value Evolution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_agent(agent, env_name='CartPole-v1', num_episodes=10, render=True):
    """Evaluate trained agent."""
    
    env = gym.make(env_name)
    
    rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            if render:
                env.render()
            
            action = agent.select_action(state, training=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    env.close()
    
    print(f"\nAverage Reward over {num_episodes} episodes: {np.mean(rewards):.2f}")
    
    return rewards


if __name__ == "__main__":
    # Train DQN
    print("Training DQN on CartPole-v1...")
    agent, rewards, lengths = train_dqn(num_episodes=500, render_freq=1000)
    
    # Visualize training
    visualize_training(rewards, lengths, agent)
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    eval_rewards = evaluate_agent(agent, num_episodes=10, render=True)
    
    # Save model
    agent.save('cartpole_dqn.pth')
    print("\nModel saved to cartpole_dqn.pth")