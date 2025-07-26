"""
A2C (Advantage Actor-Critic) Agent Implementation

This module implements the A2C algorithm as demonstrated in Chapter 9,
following the step-by-step construction approach.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from actor_critic_network import ActorCriticNetwork


class A2C:
    """Advantage Actor-Critic implementation."""
    
    def __init__(self, state_dim, action_dim, lr=7e-4, gamma=0.99, 
                 value_coef=0.5, entropy_coef=0.01, n_steps=5):
        # Core hyperparameters
        self.gamma = gamma           # Discount factor
        self.value_coef = value_coef # Weight for value loss
        self.entropy_coef = entropy_coef  # Weight for entropy bonus
        self.n_steps = n_steps       # Steps before update
        
        # The shared actor-critic network
        self.ac_network = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)
        
        # Experience storage (for n-step returns)
        self.reset_storage()
        
        # Training metrics
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
    
    def reset_storage(self):
        """Clear experience storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state):
        """Select action and estimate state value simultaneously."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, value, entropy = self.ac_network.act(state_tensor)
        
        # Store all information needed for update
        self.states.append(state)
        self.actions.append(action.item())
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        
        return action.item()
    
    def store_transition(self, reward, done):
        """Store reward and termination signal."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_advantages(self, next_state):
        """Compute n-step returns and advantage estimates."""
        # Bootstrap from next state if not terminal
        with torch.no_grad():
            if self.dones[-1]:
                next_value = 0  # Terminal state has no future value
            else:
                next_value = self.ac_network.evaluate(
                    torch.FloatTensor(next_state).unsqueeze(0)
                ).item()
        
        # Compute n-step returns working backwards
        returns = []
        R = next_value
        
        for t in reversed(range(len(self.rewards))):
            R = self.rewards[t] + self.gamma * R * (1 - self.dones[t])
            returns.insert(0, R)
        
        # Advantages = Returns - Value estimates
        advantages = []
        for t in range(len(returns)):
            advantage = returns[t] - self.values[t]
            advantages.append(advantage)
        
        return returns, advantages
    
    def update(self, next_state):
        """Perform the A2C update step."""
        if len(self.rewards) < self.n_steps:
            return None  # Need enough experience
        
        # Compute targets
        returns, advantages = self.compute_advantages(next_state)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages (crucial for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        policy_logits, values = self.ac_network(states)
        dist = Categorical(logits=policy_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Three loss components
        policy_loss = -(log_probs * advantages.detach()).mean()  # Actor loss
        value_loss = F.mse_loss(values.squeeze(), returns)       # Critic loss
        entropy_loss = -entropy                                  # Exploration bonus
        
        # Combined loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear experience and log metrics
        self.reset_storage()
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def save_model(self, path):
        """Save the trained model."""
        torch.save({
            'ac_network_state_dict': self.ac_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses
        }, path)
    
    def load_model(self, path):
        """Load a trained model."""
        checkpoint = torch.load(path)
        self.ac_network.load_state_dict(checkpoint['ac_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']


def train_a2c_cartpole(num_episodes=1000, render=False):
    """Train A2C on CartPole-v1."""
    import gym
    
    env = gym.make('CartPole-v1')
    agent = A2C(state_dim=4, action_dim=2, n_steps=5)
    
    state = env.reset()
    episode_reward = 0
    step_count = 0
    
    for episode in range(num_episodes):
        while True:
            if render and episode % 100 == 0:
                env.render()
            
            # Select action using current policy
            action = agent.select_action(state)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(reward, done)
            
            episode_reward += reward
            step_count += 1
            
            # Update every n_steps or at episode end
            if step_count % agent.n_steps == 0 or done:
                metrics = agent.update(next_state)
                
                if metrics and episode % 100 == 0:
                    print(f"Episode {episode}: Policy Loss = {metrics['policy_loss']:.4f}")
            
            if done:
                agent.episode_rewards.append(episode_reward)
                
                if episode % 100 == 0:
                    avg_reward = np.mean(agent.episode_rewards[-100:])
                    print(f"Episode {episode}: Reward = {episode_reward}, Avg = {avg_reward:.2f}")
                    
                    # Check if solved
                    if avg_reward >= 195:
                        print(f"Solved at episode {episode}!")
                        break
                
                state = env.reset()
                episode_reward = 0
                break
            else:
                state = next_state
    
    env.close()
    return agent


def train_a2c_lunar_lander(num_episodes=2000):
    """Train A2C on LunarLander-v2."""
    import gym
    
    env = gym.make('LunarLander-v2')
    agent = A2C(state_dim=8, action_dim=4, n_steps=5, lr=1e-3)
    
    state = env.reset()
    episode_reward = 0
    step_count = 0
    
    for episode in range(num_episodes):
        while True:
            # Select action
            action = agent.select_action(state)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(reward, done)
            
            episode_reward += reward
            step_count += 1
            
            # Update every n_steps or at episode end
            if step_count % agent.n_steps == 0 or done:
                metrics = agent.update(next_state)
            
            if done:
                agent.episode_rewards.append(episode_reward)
                
                if episode % 100 == 0:
                    avg_reward = np.mean(agent.episode_rewards[-100:])
                    print(f"Episode {episode}: Reward = {episode_reward:.1f}, Avg = {avg_reward:.1f}")
                    
                    # Check if solved (200+ average over 100 episodes)
                    if avg_reward >= 200:
                        print(f"Solved at episode {episode}!")
                        break
                
                state = env.reset()
                episode_reward = 0
                break
            else:
                state = next_state
    
    env.close()
    return agent


# Example usage and testing
if __name__ == "__main__":
    print("Testing A2C implementation...")
    
    # Test basic functionality
    agent = A2C(state_dim=4, action_dim=2)
    
    # Test action selection
    test_state = np.random.randn(4)
    action = agent.select_action(test_state)
    print(f"Selected action: {action}")
    
    # Test storage and update
    agent.store_transition(1.0, False)
    agent.store_transition(1.0, False)
    agent.store_transition(1.0, False)
    agent.store_transition(1.0, False)
    agent.store_transition(0.0, True)
    
    next_state = np.random.randn(4)
    metrics = agent.update(next_state)
    
    if metrics:
        print(f"Policy loss: {metrics['policy_loss']:.4f}")
        print(f"Value loss: {metrics['value_loss']:.4f}")
        print(f"Entropy: {metrics['entropy']:.4f}")
    
    print("\nTraining A2C on CartPole...")
    trained_agent = train_a2c_cartpole(num_episodes=300)
    print(f"Final average reward: {np.mean(trained_agent.episode_rewards[-100:]):.2f}")
    
    print("A2C test completed!")