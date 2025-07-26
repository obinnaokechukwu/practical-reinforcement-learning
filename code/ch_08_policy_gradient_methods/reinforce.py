"""
Basic REINFORCE Algorithm Implementation

This module implements the vanilla REINFORCE algorithm, which forms the foundation
of policy gradient methods.
"""

import torch
import torch.nn.functional as F
import numpy as np
from policy_network import PolicyNetwork


class REINFORCE:
    """
    Vanilla REINFORCE algorithm.
    
    This implementation follows the original REINFORCE algorithm by Williams (1992),
    which directly optimizes the policy using the policy gradient theorem.
    """
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        """
        Initialize REINFORCE agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            lr: Learning rate for policy optimization
            gamma: Discount factor for returns
        """
        self.gamma = gamma
        
        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
        
        # Training metrics
        self.episode_returns = []
        self.policy_losses = []
    
    def select_action(self, state):
        """
        Select action using current policy.
        
        Args:
            state: Current state (numpy array)
            
        Returns:
            int: Selected action
        """
        state_tensor = torch.FloatTensor(state)
        
        # Get action distribution
        action_dist = self.policy(state_tensor)
        
        # Sample action
        action = action_dist.sample()
        
        # Store log probability for later update
        self.log_probs.append(action_dist.log_prob(action))
        
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
    
    def update_policy(self):
        """
        Update policy using REINFORCE gradient estimate.
        
        Returns:
            float: Policy loss value
        """
        if not self.rewards:
            return 0.0
        
        # Compute returns
        returns = self.compute_returns()
        
        # Convert log probs to tensor
        log_probs = torch.stack(self.log_probs)
        
        # Normalize returns (optional, helps with stability)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        # Negative because we want to maximize expected return
        policy_loss = -(log_probs * returns).sum()
        
        # Gradient ascent step
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping to prevent large updates
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Store metrics
        self.episode_returns.append(sum(self.rewards))
        self.policy_losses.append(policy_loss.item())
        
        # Clear episode data
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
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
            action_dist = self.policy(state_tensor)
            probs = action_dist.probs.numpy()
        
        return probs
    
    def save_model(self, path):
        """Save the trained policy network."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_returns': self.episode_returns,
            'policy_losses': self.policy_losses
        }, path)
    
    def load_model(self, path):
        """Load a trained policy network."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_returns = checkpoint['episode_returns']
        self.policy_losses = checkpoint['policy_losses']


def train_cartpole_reinforce(num_episodes=1000, render=False):
    """
    Train REINFORCE agent on CartPole-v1.
    
    Args:
        num_episodes: Number of training episodes
        render: Whether to render the environment
        
    Returns:
        REINFORCE: Trained agent
    """
    import gym
    
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim)
    
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
        
        # Update policy
        loss = agent.update_policy()
        
        # Print progress
        if episode % 100 == 0:
            recent_returns = agent.episode_returns[-100:] if len(agent.episode_returns) >= 100 else agent.episode_returns
            avg_return = np.mean(recent_returns)
            print(f"Episode {episode}, Average Return: {avg_return:.2f}, Loss: {loss:.4f}")
            
            # Check if solved
            if avg_return >= 195:
                print(f"Solved at episode {episode}!")
                break
    
    env.close()
    return agent


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    print("Testing REINFORCE implementation...")
    
    # Create agent
    agent = REINFORCE(state_dim=4, action_dim=2)
    
    # Test action selection
    test_state = np.random.randn(4)
    action = agent.select_action(test_state)
    print(f"Selected action: {action}")
    
    # Test return computation
    agent.rewards = [1, 1, 1, 0]  # Simple episode
    returns = agent.compute_returns()
    print(f"Returns: {returns}")
    
    # Test policy update
    loss = agent.update_policy()
    print(f"Policy loss: {loss:.4f}")
    
    # Train on CartPole
    print("\nTraining REINFORCE on CartPole-v1...")
    trained_agent = train_cartpole_reinforce(num_episodes=500)
    
    print(f"Final average return: {np.mean(trained_agent.episode_returns[-100:]):.2f}")
    print("Training completed!")