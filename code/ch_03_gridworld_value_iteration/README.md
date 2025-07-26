# Gridworld Value Iteration Implementation

This code implements Dynamic Programming algorithms for solving gridworld environments, accompanying Chapter 3 of "Practical Reinforcement Learning".

## Contents

- `gridworld.py`: Customizable gridworld environment with visualization
- `dynamic_programming.py`: Implementation of DP algorithms (Value Iteration, Policy Iteration, etc.)
- `experiments.py`: Comprehensive experiments demonstrating key concepts
- `requirements.txt`: Python dependencies

## Features

### Gridworld Environment
- Configurable grid size and layout
- Support for obstacles, terminal states, and custom rewards
- Stochastic transitions (slippery floors)
- Wind effects
- Rich visualization of values and policies

### Algorithms Implemented
1. **Policy Evaluation**: Evaluate a given policy
2. **Policy Iteration**: Alternate between evaluation and improvement
3. **Value Iteration**: Direct computation of optimal values
4. **Modified Policy Iteration**: Parameterized between VI and PI
5. **Prioritized Sweeping**: Efficient updates for sparse rewards

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from gridworld import GridWorld
from dynamic_programming import DynamicProgramming

# Create environment
env = GridWorld(
    height=4,
    width=4,
    start=(0, 0),
    terminals=[(3, 3)],
    obstacles=[(1, 1), (2, 1)]
)

# Solve with Value Iteration
dp = DynamicProgramming(env, gamma=0.9)
policy, V, info = dp.value_iteration()

# Visualize results
env.render_policy(policy, V)
```

### Run All Experiments

```bash
python experiments.py
```

This will generate several visualizations:
- `basic_gridworld_results.png`: Optimal policy and values
- `stochastic_comparison.png`: Effect of stochastic transitions
- `reward_shaping_comparison.png`: Different reward structures
- `algorithm_comparison.png`: Performance comparison
- `discount_factor_effect.png`: Impact of γ on behavior
- `value_propagation.png`: How values spread during iteration

## Experiments

1. **Basic Gridworld**: Simple pathfinding with obstacles
2. **Stochastic Transitions**: Handling uncertainty in movement
3. **Reward Shaping**: How rewards affect optimal behavior
4. **Algorithm Comparison**: Performance of different DP methods
5. **Discount Factor**: Short vs. long-term thinking
6. **Prioritized Sweeping**: Efficient computation for large spaces
7. **Value Propagation**: Visualizing the learning process

## Key Insights

- **Value Iteration vs Policy Iteration**: VI takes more iterations but each is cheaper
- **Stochasticity**: Uncertainty leads to more conservative policies
- **Discount Factor**: Low γ leads to myopic behavior, high γ considers long-term rewards
- **Prioritized Sweeping**: Dramatically faster for sparse reward environments

## Customization

Create your own environments:

```python
env = GridWorld(
    height=10,
    width=10,
    start=(0, 0),
    terminals=[(9, 9)],
    obstacles=[(5, i) for i in range(8)],
    rewards={(5, 9): 50, (7, 7): -20},
    wind={(3, 3): (0, 1)},  # Wind pushes right
    slip_prob=0.1  # 10% chance of slipping
)
```

## Performance Notes

- Matrix inversion for small environments (< 100 states)
- Iterative methods scale to thousands of states
- Prioritized sweeping best for sparse rewards
- Consider function approximation for continuous/large state spaces