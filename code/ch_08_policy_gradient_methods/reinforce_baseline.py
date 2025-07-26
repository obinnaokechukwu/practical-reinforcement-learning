"""
REINFORCE with Baseline Implementation

This module implements REINFORCE with a value function baseline to reduce
gradient variance, as demonstrated in Chapter 8.
"""

import torch
import torch.nn.functional as F
import numpy as np
from policy_network import PolicyNetwork, ValueNetwork


class REINFORCEWithBaseline:
    """
    REINFORCE algorithm with value function baseline for variance reduction.
    
    This implementation adds a learned baseline (value function) to reduce
    the variance of policy gradient estimates while maintaining unbiased gradients.
    """
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        """
        Initialize REINFORCE with baseline agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            lr: Learning rate for both networks
            gamma: Discount factor for returns
        """
        self.gamma = gamma
        
        # Separate networks for policy and value
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        
        # Separate optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Episode storage
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        # Training metrics
        self.episode_returns = []
        self.policy_losses = []
        self.value_losses = []
    
    def select_action(self, state):
        """
        Select action and compute state value.
        
        Args:
            state: Current state (numpy array)
            
        Returns:
            int: Selected action
        """
        state_tensor = torch.FloatTensor(state)
        
        # Get action distribution
        action_dist = self.policy_net(state_tensor)
        
        # Get state value
        value = self.value_net(state_tensor)
        
        # Sample action
        action = action_dist.sample()
        
        # Store log probability and value for later update
        self.log_probs.append(action_dist.log_prob(action))
        self.values.append(value)
        
        return action.item()
    
    def compute_returns(self):
        """
        Compute discounted returns for the episode.
        
        Returns:
            torch.Tensor: Discounted returns for each step
        """
        returns = []
        R = 0
        
        # Compute returns backwards from end of episode
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return torch.tensor(returns, dtype=torch.float32)
    
    def compute_advantages(self):
        """
        Compute advantage estimates using the baseline.
        
        Returns:
            tuple: (returns, values, advantages)
        """
        # Compute returns
        returns = self.compute_returns()
        
        # Get values as tensor
        values = torch.stack(self.values).squeeze()
        
        # Advantage = Return - Baseline
        advantages = returns - values.detach()  # Don't backprop through baseline
        
        # Normalize advantages (reduces variance further)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, values, advantages
    
    def update_policy(self):
        """
        Update both policy and value networks.
        
        Returns:
            tuple: (policy_loss, value_loss)
        """
        if not self.rewards:
            return 0.0, 0.0
        
        # Compute returns, values, and advantages
        returns, values, advantages = self.compute_advantages()
        
        # Get log probabilities
        log_probs = torch.stack(self.log_probs)
        
        # Policy loss: REINFORCE with advantage
        policy_loss = -(log_probs * advantages).sum()
        
        # Value loss: MSE between predicted and actual returns
        value_loss = F.mse_loss(values, returns)
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
        self.value_optimizer.step()
        
        # Store metrics
        self.episode_returns.append(sum(self.rewards))
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        
        # Clear episode data
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return policy_loss.item(), value_loss.item()
    
    def get_action_probabilities(self, state):
        """
        Get action probabilities for a given state (for analysis).
        
        Args:
            state: Current state (numpy array)
            
        Returns:
            numpy.ndarray: Action probabilities
        """
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            action_dist = self.policy_net(state_tensor)
            probs = action_dist.probs.numpy()
        
        return probs
    
    def get_state_value(self, state):
        """
        Get estimated state value (for analysis).
        
        Args:
            state: Current state (numpy array)
            
        Returns:
            float: Estimated state value
        """
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            value = self.value_net(state_tensor)
        
        return value.item()
    
    def save_model(self, path):
        """Save the trained networks."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_returns': self.episode_returns,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses
        }, path)
    
    def load_model(self, path):
        """Load trained networks."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.episode_returns = checkpoint['episode_returns']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']


def train_cartpole_baseline(num_episodes=1000, render=False):
    """
    Train REINFORCE with baseline on CartPole-v1.
    
    Args:
        num_episodes: Number of training episodes
        render: Whether to render the environment
        
    Returns:
        REINFORCEWithBaseline: Trained agent
    """
    import gym
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEWithBaseline(state_dim, action_dim)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            if render and episode % 100 == 0:
                env.render()
            
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.rewards.append(reward)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update networks
        policy_loss, value_loss = agent.update_policy()
        
        # Print progress
        if episode % 100 == 0:
            recent_returns = agent.episode_returns[-100:] if len(agent.episode_returns) >= 100 else agent.episode_returns
            avg_return = np.mean(recent_returns)
            print(f"Episode {episode}, Avg Return: {avg_return:.2f}, "
                  f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            # Check if solved
            if avg_return >= 195:
                print(f"Solved at episode {episode}!")
                break
    
    env.close()
    return agent


def compare_variance():
    """
    Compare gradient variance between REINFORCE and REINFORCE with baseline.
    This is a simplified version of the analysis from the chapter.
    """
    import gym
    import matplotlib.pyplot as plt
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create both agents
    vanilla_agent = __import__('reinforce').REINFORCE(state_dim, action_dim)
    baseline_agent = REINFORCEWithBaseline(state_dim, action_dim)
    
    agents = {
        'REINFORCE': vanilla_agent,
        'REINFORCE + Baseline': baseline_agent
    }
    
    results = {}
    
    for name, agent in agents.items():
        print(f"Training {name}...")
        episode_returns = []
        
        for episode in range(200):
            state = env.reset()
            agent.rewards = []  # Reset rewards
            
            while True:
                action = agent.select_action(state)
                state, reward, done, _ = env.step(action)
                agent.rewards.append(reward)
                
                if done:
                    break
            
            # Update agent
            if name == 'REINFORCE + Baseline':
                policy_loss, value_loss = agent.update_policy()
            else:
                agent.update_policy()
            
            episode_returns.append(sum(agent.rewards))
        
        results[name] = episode_returns
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, returns in results.items():
        # Raw returns
        plt.plot(returns, alpha=0.3, label=f'{name} (raw)')
        # Smoothed
        if len(returns) > 20:
            smoothed = np.convolve(returns, np.ones(20)/20, mode='valid')
            plt.plot(smoothed, linewidth=2, label=f'{name} (smoothed)')
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Variance comparison
    plt.subplot(1, 2, 2)
    variances = [np.var(returns[-50:]) for returns in results.values()]
    plt.bar(results.keys(), variances, alpha=0.7)
    plt.ylabel('Return Variance (last 50 episodes)')
    plt.title('Variance Reduction with Baseline')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/variance_comparison.png', dpi=150)
    plt.show()
    
    # Print statistics
    print("\nPerformance Comparison (last 50 episodes):")
    for name, returns in results.items():
        mean_return = np.mean(returns[-50:])
        variance = np.var(returns[-50:])
        print(f"{name}: Mean = {mean_return:.1f}, Variance = {variance:.1f}")
    
    env.close()


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    print("Testing REINFORCE with Baseline implementation...")
    
    # Create agent
    agent = REINFORCEWithBaseline(state_dim=4, action_dim=2)
    
    # Test action selection
    test_state = np.random.randn(4)
    action = agent.select_action(test_state)
    print(f"Selected action: {action}")
    print(f"State value estimate: {agent.get_state_value(test_state):.4f}")
    
    # Test return computation and advantage
    agent.rewards = [1, 1, 1, 0]  # Simple episode
    returns, values, advantages = agent.compute_advantages()
    print(f"Returns: {returns}")
    print(f"Advantages: {advantages}")
    
    # Test policy update
    policy_loss, value_loss = agent.update_policy()
    print(f"Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}")
    
    # Train on CartPole
    print("\nTraining REINFORCE with Baseline on CartPole-v1...")
    trained_agent = train_cartpole_baseline(num_episodes=300)
    
    print(f"Final average return: {np.mean(trained_agent.episode_returns[-100:]):.2f}")
    
    # Compare variance
    print("\nComparing variance between algorithms...")
    compare_variance()
    
    print("Training completed!")