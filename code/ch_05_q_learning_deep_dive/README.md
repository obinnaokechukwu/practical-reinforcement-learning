# Q-Learning Deep Dive Implementation

This code implements Q-Learning algorithms for a custom Treasure Hunt environment, accompanying Chapter 5 of "Practical Reinforcement Learning".

## Contents

- `treasure_hunt_env.py`: Custom gridworld environment with treasures, traps, keys, and doors
- `q_learning.py`: Q-Learning implementations (basic and advanced)
- `experiments.py`: Comprehensive experiments demonstrating Q-learning concepts
- `requirements.txt`: Python dependencies

## Features

### Treasure Hunt Environment
- Multiple treasures with different reward values
- Traps that end episodes with negative rewards
- Keys and doors requiring sequential actions
- Battery system creating time pressure
- Rich state representation and visualization

### Q-Learning Implementations
1. **Basic Q-Learning**: Core algorithm with ε-greedy exploration
2. **Advanced Q-Learning**: Includes experience replay, adaptive learning rates
3. **Debugging Tools**: Diagnostic functions for troubleshooting
4. **SARSA Comparison**: On-policy vs off-policy learning

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from treasure_hunt_env import TreasureHuntEnv
from q_learning import TreasureHuntQLearning

# Create environment
env = TreasureHuntEnv(width=8, height=8, num_treasures=3, num_traps=4)

# Create and train agent
agent = TreasureHuntQLearning(env)
agent.train(num_episodes=1000)

# Demonstrate learned policy
reward, steps = agent.demonstrate_policy()
print(f"Total reward: {reward}, Steps taken: {steps}")
```

### Run All Experiments

```bash
python experiments.py
```

This generates several visualizations:
- `treasure_hunt_learning_curves.png`: Training progress over episodes
- `exploration_comparison.png`: Impact of exploration rate (ε)
- `learning_rate_comparison.png`: Effect of different learning rates
- `q_learning_vs_sarsa.png`: Off-policy vs on-policy comparison
- `experience_replay_impact.png`: Benefits of experience replay
- `q_values_heatmap.png`: Visualization of learned Q-values

## Experiments

1. **Basic Learning**: Demonstrates Q-learning on the treasure hunt task
2. **Exploration Analysis**: Shows how ε affects performance and coverage
3. **Learning Rate Study**: Compares different α values
4. **Initialization Impact**: Optimistic vs pessimistic Q-value initialization
5. **Algorithm Comparison**: Q-learning vs SARSA (off-policy vs on-policy)
6. **Experience Replay**: Shows sample efficiency improvements
7. **Debugging Demo**: How to diagnose Q-learning issues

## Key Concepts Demonstrated

### Off-Policy Learning
Q-learning learns optimal values while following an exploratory policy, demonstrated through consistent convergence regardless of exploration rate.

### Exploration-Exploitation
The ε-greedy strategy with decay shows how to balance discovering new strategies with exploiting known good actions.

### State Abstraction
The state representation (position, collected items, keys, battery) shows how to encode complex environments for tabular methods.

### Convergence Properties
Despite stochasticity and sparse rewards, Q-learning reliably converges to near-optimal policies.

## Customization

Create custom environments by modifying parameters:

```python
env = TreasureHuntEnv(
    width=12,
    height=12,
    num_treasures=5,
    num_traps=8
)
```

Adjust Q-learning hyperparameters:

```python
agent = TreasureHuntQLearning(
    env,
    alpha=0.2,      # Learning rate
    gamma=0.95,     # Discount factor
    epsilon=0.5     # Initial exploration rate
)
```

## Insights from Experiments

1. **Exploration is Critical**: Too little exploration (ε < 0.1) leads to suboptimal policies
2. **Learning Rate Matters**: α = 0.1 works well; too high causes instability
3. **Optimistic Initialization**: Starting with high Q-values encourages exploration
4. **Experience Replay**: Significantly improves sample efficiency
5. **Off-Policy Advantage**: Q-learning often outperforms SARSA in this domain

## Advanced Usage

### Custom Reward Shaping

```python
# Modify environment rewards
env.treasures[(7, 7)] = 100  # High-value treasure
env.rewards[(3, 3)] = -5     # Penalty location
```

### Debugging Q-Learning

```python
from q_learning import debug_q_learning

# Diagnose learning issues
debug_q_learning(agent, env)
```

### Extracting Optimal Policy

```python
# Get best action for any state
state = env._get_state()
best_action = max(agent.Q[state], key=agent.Q[state].get)
```