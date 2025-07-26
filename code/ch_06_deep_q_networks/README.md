# Deep Q-Networks (DQN) Implementation

This directory contains a complete implementation of Deep Q-Networks and its variants for solving the CartPole-v1 environment.

## Overview

The implementation demonstrates the key innovations that make DQN work:
- Neural network function approximation
- Experience replay for breaking correlations
- Target networks for stable learning
- Various algorithmic improvements (Double DQN, Dueling, Prioritized Replay)

## Files

- `dqn_network.py`: Neural network architectures (standard DQN, Dueling DQN, Noisy networks)
- `replay_buffer.py`: Experience replay implementations (uniform, prioritized, n-step)
- `dqn_agent.py`: Complete DQN agent with all components integrated
- `train_cartpole.py`: Training script with visualization and evaluation
- `experiments.py`: Experimental analysis and comparisons

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Train a basic DQN agent:
```bash
python train_cartpole.py
```

Train with improvements:
```bash
# Double DQN with prioritized replay
python train_cartpole.py --agent double_dqn --prioritized

# Dueling architecture
python train_cartpole.py --dueling

# Compare all variants
python train_cartpole.py --compare
```

## Key Concepts Demonstrated

### 1. Neural Network Function Approximation
- Replaces tabular Q-values with a neural network
- Enables generalization across similar states
- Handles continuous state spaces

### 2. Experience Replay
- Stores past experiences in a buffer
- Samples random minibatches for training
- Breaks temporal correlations
- Improves sample efficiency

### 3. Target Networks
- Separate network for computing targets
- Updated periodically (not every step)
- Prevents destructive feedback loops
- Stabilizes training

### 4. Algorithm Variants

**Double DQN**: Reduces overestimation bias by decoupling action selection and evaluation
```python
# Standard DQN
next_q = max(Q_target(s'))

# Double DQN  
a' = argmax(Q_online(s'))
next_q = Q_target(s', a')
```

**Dueling Architecture**: Separates value and advantage streams
```python
Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
```

**Prioritized Replay**: Samples important experiences more frequently
```python
P(i) = |TD_error_i|^α / Σ|TD_error_k|^α
```

## Expected Results

On CartPole-v1, the agent should:
- Start with random performance (~20 reward)
- Rapidly improve after 50-100 episodes
- Solve the environment (average reward ≥ 195) within 200-300 episodes
- Achieve near-perfect performance (499-500 reward) when fully trained

## Visualization

The training script produces several visualizations:
- Learning curves showing reward progression
- Episode length evolution
- Loss and Q-value trends
- Algorithm comparison plots

## Hyperparameters

Key hyperparameters and their typical values:
```python
learning_rate = 1e-3      # Not too high (instability) or low (slow)
gamma = 0.99              # High for long episodes
epsilon_decay = 0.995     # Gradual exploration decay
target_update = 10        # Update target network every 10 steps
batch_size = 64           # Balance between stability and speed
buffer_size = 10000       # Store recent experiences
```

## Troubleshooting

Common issues and solutions:

1. **Q-values exploding**: Reduce learning rate or increase target update frequency
2. **No improvement**: Check exploration (epsilon too low) or increase network capacity
3. **Unstable training**: Ensure replay buffer is large enough and batch size is reasonable
4. **Slow learning**: Increase learning rate or reduce epsilon decay rate

## Extensions

The code is designed to be easily extended:
- Add new network architectures in `dqn_network.py`
- Implement new replay strategies in `replay_buffer.py`
- Create new environments by changing `env_name` parameter
- Add visualization and analysis tools

## References

- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning with Double Q-learning (van Hasselt et al., 2016)](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures (Wang et al., 2016)](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay (Schaul et al., 2016)](https://arxiv.org/abs/1511.05952)