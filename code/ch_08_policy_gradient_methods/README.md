# Policy Gradient Methods Implementation

This directory contains implementations of the REINFORCE algorithm with and without baseline, as covered in Chapter 8 of "Practical Reinforcement Learning".

## Overview

Policy gradient methods directly optimize the policy parameters to maximize expected return. This implementation demonstrates:

1. **Vanilla REINFORCE**: Basic policy gradient with high variance
2. **REINFORCE with Baseline**: Variance reduction using a value function
3. **LunarLander Implementation**: Complete application to a challenging control task

## Files

- `policy_network.py`: Neural network architectures for policy and value functions
- `reinforce.py`: Basic REINFORCE algorithm implementation
- `reinforce_baseline.py`: REINFORCE with baseline for variance reduction
- `lunar_lander_agent.py`: Complete agent for LunarLander-v2 environment
- `experiments.py`: Comparison experiments and analysis tools
- `train_lunar_lander.py`: Training script for LunarLander
- `visualizations.py`: Analysis and visualization tools

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic REINFORCE on CartPole

```python
from reinforce import REINFORCE
import gym

env = gym.make('CartPole-v1')
agent = REINFORCE(state_dim=4, action_dim=2)

# Train for 500 episodes
for episode in range(500):
    state = env.reset()
    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.rewards.append(reward)
        state = next_state
        if done:
            break
    agent.update_policy()
```

### 2. REINFORCE with Baseline

```python
from reinforce_baseline import REINFORCEWithBaseline

agent = REINFORCEWithBaseline(state_dim=4, action_dim=2)
# Training loop same as above, but with baseline variance reduction
```

### 3. LunarLander Training

```bash
python train_lunar_lander.py --episodes 1000 --update-frequency 32
```

## Key Concepts

### Policy Gradient Theorem

The fundamental insight that enables policy gradients:

```
∇θ J(θ) = E_π[∇θ log π(a|s) * Q^π(s,a)]
```

Where:
- `J(θ)` is the expected return
- `π(a|s)` is the policy
- `Q^π(s,a)` is the action-value function

### Variance Reduction with Baseline

Subtracting a state-dependent baseline `b(s)` doesn't bias the gradient:

```
∇θ J(θ) = E_π[∇θ log π(a|s) * (Q^π(s,a) - b(s))]
```

Using `V^π(s)` as the baseline gives us the advantage function:
```
A^π(s,a) = Q^π(s,a) - V^π(s)
```

## Implementation Details

### Network Architecture

- **Policy Network**: Maps states to action probabilities
- **Value Network**: Estimates state values for baseline
- **Shared Architecture**: Optional shared backbone for efficiency

### Training Process

1. **Episode Collection**: Run episodes to collect trajectories
2. **Return Computation**: Calculate discounted returns for each step
3. **Advantage Estimation**: Compute advantages using baseline
4. **Policy Update**: Gradient ascent on policy parameters
5. **Value Update**: Minimize MSE loss for value function

### Key Challenges

1. **High Variance**: Policy gradients have inherently high variance
2. **Sample Efficiency**: Requires many episodes to learn
3. **Exploration**: Need to balance exploration vs exploitation
4. **Stability**: Large policy changes can be destructive

## Experimental Results

The implementations include experiments comparing:

- **REINFORCE vs REINFORCE+Baseline**: Variance reduction analysis
- **Learning Curves**: Performance over training episodes
- **Gradient Variance**: Measuring stability improvements
- **Sample Efficiency**: Episodes needed to solve tasks

Typical results show:
- Baseline reduces gradient variance by 30-50%
- Faster convergence with baseline
- More stable learning curves

## Usage Examples

### Variance Comparison

```python
from experiments import compare_variance_reduction
compare_variance_reduction()
```

### Training Visualization

```python
from visualizations import plot_training_progress
from train_lunar_lander import train_lunar_lander

agent = train_lunar_lander(episodes=1000)
plot_training_progress(agent)
```

### Policy Analysis

```python
from visualizations import analyze_policy
env = gym.make('LunarLander-v2')
analyze_policy(agent, env)
```

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 13
- Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning"
- Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning"

## Performance Benchmarks

| Environment | Algorithm | Episodes to Solve | Final Score |
|-------------|-----------|------------------|-------------|
| CartPole-v1 | REINFORCE | ~400 | 190+ |
| CartPole-v1 | REINFORCE+Baseline | ~250 | 195+ |
| LunarLander-v2 | REINFORCE+Baseline | ~800 | 200+ |

**Note**: Results may vary due to randomness. Run multiple seeds for statistical significance.