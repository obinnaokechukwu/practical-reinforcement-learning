# The DQN Zoo: Key Improvements to Deep Q-Networks

This directory contains implementations of major improvements to the original DQN algorithm, demonstrating how each addresses specific weaknesses.

## Overview

The DQN Zoo showcases three fundamental improvements:
1. **Double DQN**: Reduces overestimation bias
2. **Dueling DQN**: Improves learning efficiency through value-advantage decomposition
3. **Prioritized Experience Replay**: Focuses learning on important experiences

## Files

- `double_dqn.py`: Double DQN implementation showing overestimation reduction
- `dueling_dqn.py`: Dueling architecture that separates V(s) and A(s,a)
- `prioritized_replay.py`: Priority-based experience replay using TD errors
- `rainbow_dqn.py`: Combines all improvements into one agent
- `experiments.py`: Comprehensive comparisons and analysis

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Double DQN - Reducing Overestimation

```bash
python double_dqn.py
```

This demonstrates how Double DQN prevents the systematic overestimation of Q-values by decoupling action selection from evaluation.

Key insight: Use online network to select actions, target network to evaluate them.

### 2. Dueling DQN - Value-Advantage Decomposition

```bash
python dueling_dqn.py
```

Shows how separating state value V(s) from action advantages A(s,a) improves learning, especially in states where action choice doesn't matter much.

Key insight: Q(s,a) = V(s) + (A(s,a) - mean(A))

### 3. Prioritized Experience Replay

```bash
python prioritized_replay.py
```

Demonstrates how sampling based on TD error magnitude helps the agent focus on surprising or difficult experiences.

Key insight: Priority = |TD error|^α, with importance sampling to correct bias.

## Key Concepts

### Double DQN

Standard DQN uses the same network for selection and evaluation:
```python
# Standard DQN (overestimates)
Q_target = r + γ * max(Q_target(s', a'))
```

Double DQN decouples these operations:
```python
# Double DQN (reduces overestimation)
a* = argmax(Q_online(s', a'))  # Online network selects
Q_target = r + γ * Q_target(s', a*)  # Target network evaluates
```

### Dueling Architecture

Instead of directly outputting Q(s,a), the network computes:
- V(s): How good is this state?
- A(s,a): How much better is action a compared to others?

Then combines: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

Benefits:
- Learns V(s) faster when many actions have similar values
- Better generalization across actions
- More stable learning

### Prioritized Experience Replay

Samples experiences based on their "surprisingness" (TD error):

1. **Priority Calculation**: P(i) = |TD_error|^α + ε
2. **Sampling Probability**: P(i) / Σ P(j)
3. **Importance Sampling**: w(i) = (N * P(i))^(-β)

Benefits:
- Learns from rare but important events
- Faster convergence on difficult transitions
- More sample-efficient learning

## Experimental Results

Each improvement addresses specific issues:

| Algorithm | Problem Solved | Typical Improvement |
|-----------|---------------|-------------------|
| Double DQN | Overestimation bias | 10-30% better final performance |
| Dueling DQN | Learning efficiency | 20-50% faster learning |
| Prioritized Replay | Sample efficiency | 2-3x faster to solve |

## Rainbow DQN

The `rainbow_dqn.py` file combines all improvements:
- Double Q-learning for unbiased targets
- Dueling architecture for better representations
- Prioritized replay for sample efficiency
- Additional: n-step returns, noisy networks

```bash
python rainbow_dqn.py --compare
```

## Understanding the Improvements

### Visualization Tools

The implementations include extensive visualization:
- Q-value evolution over training
- Overestimation comparison (Standard vs Double)
- Value-Advantage decomposition (Dueling)
- Priority distribution analysis (PER)
- Learning curve comparisons

### Common Patterns

All improvements follow similar principles:
1. **Identify a specific weakness** in vanilla DQN
2. **Propose a targeted solution** 
3. **Minimal changes** to the base algorithm
4. **Composable** - can be combined with other improvements

## Practical Guidelines

When to use each improvement:

**Double DQN**: Always. Minimal computational cost, consistent improvements.

**Dueling DQN**: 
- Large action spaces
- Many states where action choice doesn't matter
- Problems with clear value structure

**Prioritized Replay**:
- Sparse rewards
- Rare important events
- When sample efficiency is critical

**Rainbow (All)**: 
- Maximum performance needed
- Computational resources available
- Complex environments

## References

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep RL](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298)