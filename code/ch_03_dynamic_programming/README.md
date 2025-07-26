# Dynamic Programming Implementation

This directory contains a complete implementation of Dynamic Programming algorithms for solving gridworld environments.

## Files

- `gridworld.py`: Flexible gridworld environment supporting obstacles, stochastic transitions, and custom rewards
- `dp_solver.py`: Implementation of Policy Iteration, Value Iteration, Modified Policy Iteration, and Prioritized Sweeping
- `experiments.py`: Educational experiments demonstrating key DP concepts
- `main.py`: Example usage and demonstrations

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from gridworld import GridWorld
from dp_solver import DPSolver

# Create a simple gridworld
env = GridWorld(height=4, width=4, start=(0,0), terminals=[(3,3)])

# Solve using Value Iteration
solver = DPSolver(env, gamma=0.99)
policy, values, iterations = solver.value_iteration()

# Visualize results
env.render_policy(policy, values)
```

## Running Experiments

To run all educational experiments:

```bash
python experiments.py
```

This will generate:
- `discount_factor_experiment.png`: Shows how γ affects optimal policies
- `convergence_analysis.png`: Compares convergence of different algorithms
- `stochastic_planning.png`: Demonstrates planning under uncertainty
- `prioritized_sweeping.png`: Shows efficiency gains from smart updates

## Key Concepts Demonstrated

1. **Policy Iteration vs Value Iteration**: Trade-offs between number of iterations and cost per iteration
2. **Discount Factor Effects**: How γ influences planning horizon and optimal behavior
3. **Stochastic Planning**: Risk-aware policies in uncertain environments
4. **Prioritized Sweeping**: Efficient updates by focusing on high-error states

## Environment Features

The GridWorld environment supports:
- Configurable grid dimensions
- Obstacle placement
- Terminal states with rewards
- Wind fields (deterministic drift)
- Slippery surfaces (stochastic transitions)
- Custom reward structures

## Algorithm Implementations

All algorithms follow the standard RL conventions:
- State space: Integer indices
- Action space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
- Transition model: P[s][a] = [(prob, next_state, reward, done), ...]
- Discount factor γ for future rewards

## Visualization

The implementation includes rich visualization tools:
- Value function heatmaps with numeric displays
- Policy visualization with directional arrows
- Update count heatmaps for algorithm analysis
- Side-by-side comparisons for different settings