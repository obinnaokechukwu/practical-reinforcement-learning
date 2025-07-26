# Actor-Critic Methods Implementation

This directory contains implementations of Actor-Critic methods as covered in Chapter 9 of "Practical Reinforcement Learning".

## Overview

Actor-Critic methods combine policy-based and value-based approaches, using:
- **Actor**: Neural network that learns the policy π(a|s)
- **Critic**: Neural network that learns the value function V(s)

The critic provides low-variance advantage estimates to guide policy updates.

## Files

- `actor_critic_network.py`: Neural network architectures for actor-critic
- `a2c_agent.py`: A2C (Advantage Actor-Critic) implementation
- `a3c_worker.py`: A3C worker components (simplified version)
- `experiments.py`: Training comparisons and analysis
- `visualizations.py`: Tools for analyzing training progress

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train A2C on CartPole

```python
from a2c_agent import train_a2c_cartpole

# Train A2C agent
agent = train_a2c_cartpole(num_episodes=500)
print(f"Final performance: {agent.episode_rewards[-100:]}")
```

### 2. Train A2C on LunarLander

```python
from a2c_agent import train_a2c_lunar_lander

# Train on more challenging environment
agent = train_a2c_lunar_lander(num_episodes=1500)
```

### 3. Custom Training Loop

```python
from a2c_agent import A2C
import gym

env = gym.make('CartPole-v1')
agent = A2C(state_dim=4, action_dim=2, n_steps=5)

state = env.reset()
for episode in range(1000):
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(reward, done)
        
        # Update every n_steps or episode end
        if len(agent.rewards) >= agent.n_steps or done:
            metrics = agent.update(next_state)
        
        if done:
            state = env.reset()
            break
        else:
            state = next_state
```

## Key Concepts

### Actor-Critic Architecture

The shared network architecture uses common feature extraction layers followed by separate heads:

```python
# Shared features
features = shared_layers(state)

# Separate outputs
policy_logits = actor_head(features)  # Action probabilities
value = critic_head(features)         # State value estimate
```

### Advantage Estimation

A2C uses n-step returns to compute advantages:

```python
# Bootstrap from next state
R = next_value if not terminal else 0

# Compute n-step returns
for t in reversed(range(n_steps)):
    R = reward[t] + gamma * R * (1 - done[t])
    returns[t] = R

# Advantage = Return - Value estimate
advantages = returns - values
```

### Three-Part Loss Function

A2C optimizes three objectives simultaneously:

```python
policy_loss = -(log_probs * advantages).mean()     # Actor
value_loss = F.mse_loss(predicted_values, returns) # Critic  
entropy_loss = -entropy.mean()                     # Exploration

total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
```

## Algorithm Comparison

| Algorithm | Synchronous | Parallel | Memory | Complexity |
|-----------|-------------|----------|--------|------------|
| Vanilla AC| Yes         | No       | Low    | Low        |
| A2C       | Yes         | Optional | Medium | Medium     |
| A3C       | No          | Yes      | Low    | High       |

## Training Results

Typical performance on standard environments:

### CartPole-v1
- **Episodes to Solve**: ~200-400
- **Final Score**: 190-200
- **Training Time**: ~2-5 minutes

### LunarLander-v2  
- **Episodes to Solve**: ~800-1200
- **Final Score**: 200-250
- **Training Time**: ~15-30 minutes

## Hyperparameter Guidelines

### Learning Rates
- **Policy**: 7e-4 (standard)
- **Value**: Same as policy (shared optimizer)

### Loss Coefficients
- **Value coefficient**: 0.5 (balance actor/critic learning)
- **Entropy coefficient**: 0.01 (encourage exploration)

### N-Step Returns
- **CartPole**: 5 steps
- **LunarLander**: 5-10 steps
- **Atari**: 20+ steps

## Advanced Features

### Shared vs Separate Networks

```python
# Shared (default)
ac_network = ActorCriticNetwork(state_dim, action_dim)

# Separate networks (more flexibility)
policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)
```

### Generalized Advantage Estimation (GAE)

For reduced variance in advantage estimates:

```python
def compute_gae(rewards, values, next_value, gamma=0.99, tau=0.95):
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * tau * gae
        advantages.insert(0, gae)
        next_value = values[t]
    
    return advantages
```

### Continuous Action Spaces

For continuous control tasks, use Gaussian policies:

```python
# Output mean and log_std
mean, log_std = actor_head(features).chunk(2, dim=-1)
std = log_std.exp()

# Create Normal distribution
dist = Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action).sum(dim=-1)
```

## Debugging Tips

### Common Issues

1. **Diverging Policy Loss**: Lower learning rate or increase entropy coefficient
2. **High Value Loss**: Check return computation, ensure proper bootstrapping
3. **Poor Exploration**: Increase entropy coefficient or use different exploration
4. **Unstable Training**: Use gradient clipping, normalize advantages

### Monitoring Training

```python
# Key metrics to track
print(f"Policy Loss: {metrics['policy_loss']:.4f}")
print(f"Value Loss: {metrics['value_loss']:.4f}")  
print(f"Entropy: {metrics['entropy']:.4f}")
print(f"Episode Reward: {episode_reward}")
```

## References

- Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning"
- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 13
- [OpenAI Baselines A2C](https://github.com/openai/baselines/tree/master/baselines/a2c)