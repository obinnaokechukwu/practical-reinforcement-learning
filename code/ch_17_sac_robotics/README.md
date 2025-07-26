# SAC Implementation for Robotics

A robust implementation of Soft Actor-Critic (SAC) with automatic temperature tuning for continuous control tasks, with a focus on robotic environments.

## Features

- **Core SAC Algorithm**: Twin Q-networks, stochastic policy with squashed Gaussian
- **Automatic Temperature Tuning**: Maintains target entropy for optimal exploration
- **Advanced Features**:
  - Ensemble Q-functions for uncertainty estimation
  - Prioritized experience replay
  - Hindsight experience replay for goal-conditioned tasks
  - Comprehensive evaluation and analysis tools
- **Robotics Focus**: Configurations optimized for complex control tasks

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file:

```
gymnasium[mujoco]>=0.28.0
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.6.0
scipy>=1.10.0
pyyaml>=6.0
wandb>=0.15.0  # Optional, for experiment tracking
```

## Quick Start

### Training

Train SAC on Ant environment with default settings:

```bash
python train.py --env Ant-v4
```

Train with custom configuration:

```bash
python train.py --config configs/humanoid.yaml
```

Train with specific hyperparameters:

```bash
python train.py --env Walker2d-v4 \
    --total-timesteps 2000000 \
    --hidden-dims 256 256 \
    --lr-policy 3e-4 \
    --batch-size 256
```

### Evaluation

Evaluate a trained agent:

```bash
python evaluate.py models/sac_Ant-v4_final.pt --env Ant-v4
```

Evaluate with rendering:

```bash
python evaluate.py models/sac_Ant-v4_final.pt --env Ant-v4 --render
```

Record evaluation videos:

```bash
python evaluate.py models/sac_Ant-v4_final.pt --env Ant-v4 --record
```

### Analysis Tools

Analyze Q-value accuracy:

```bash
python evaluate.py models/sac_Ant-v4_final.pt --env Ant-v4 --analyze-q
```

Analyze policy entropy:

```bash
python evaluate.py models/sac_Ant-v4_final.pt --env Ant-v4 --analyze-entropy
```

Compare deterministic vs stochastic policies:

```bash
python evaluate.py models/sac_Ant-v4_final.pt --env Ant-v4 --compare-policies
```

## Algorithm Details

### Maximum Entropy Framework

SAC optimizes the maximum entropy objective:

```
J(π) = E[Σ γ^t (r(s_t, a_t) + α H(π(·|s_t)))]
```

This encourages:
- Exploration through entropy maximization
- Robustness to model errors
- Discovery of multiple solutions

### Key Components

1. **Twin Q-Networks**: Mitigate overestimation bias
2. **Squashed Gaussian Policy**: Actions bounded by tanh
3. **Automatic Temperature**: Maintains target entropy
4. **Soft Updates**: Stable target network updates

### Hyperparameters

#### Critical Parameters
- `lr_q`, `lr_policy`: Learning rates (default: 3e-4)
- `tau`: Target network update rate (default: 0.005)
- `gamma`: Discount factor (default: 0.99)
- `alpha`: Initial temperature (default: 0.2)

#### Training Parameters
- `batch_size`: Batch size for updates (default: 256)
- `buffer_size`: Replay buffer capacity (default: 1M)
- `learning_starts`: Steps before training (default: 10K)

## Environment Configurations

### Ant-v4
- 8-DOF quadruped robot
- Moderate difficulty
- ~3M steps to convergence

### Humanoid-v4
- 17-DOF bipedal robot
- Very challenging
- ~10M steps to convergence

### Custom Robotics Tasks
- See `configs/franka_reach.yaml` for manipulation example
- Easily adaptable to other robotic environments

## Advanced Usage

### Ensemble Q-Functions

Use ensemble for uncertainty-aware exploration:

```python
from sac_agent import SACAgentWithEnsemble

agent = SACAgentWithEnsemble(
    obs_dim=obs_dim,
    act_dim=act_dim,
    num_q_networks=5,
    uncertainty_threshold=0.1
)
```

### Prioritized Experience Replay

Enable prioritized replay for better sample efficiency:

```bash
python train.py --env Ant-v4 --prioritized-replay
```

### Goal-Conditioned SAC

For goal-reaching tasks, use the HindsightReplayBuffer:

```python
from replay_buffer import HindsightReplayBuffer

buffer = HindsightReplayBuffer(
    capacity=1_000_000,
    obs_shape=obs_shape,
    act_shape=act_shape,
    goal_shape=goal_shape,
    n_sampled_goal=4
)
```

## Troubleshooting

### Poor Performance
- Check if temperature is too high/low (monitor alpha value)
- Increase `learning_starts` for better initial exploration
- Try different network architectures

### Training Instability
- Reduce learning rates
- Increase `tau` for more stable target updates
- Check for environment bugs (NaN rewards, etc.)

### Slow Convergence
- Use prioritized replay
- Increase batch size and update frequency
- Check if reward scale is appropriate

## Performance Benchmarks

Expected performance (mean ± std over 5 seeds):

| Environment | SAC Performance | Training Time |
|-------------|----------------|---------------|
| HalfCheetah-v4 | 12000 ± 1000 | ~2 hours |
| Walker2d-v4 | 5000 ± 500 | ~2 hours |
| Ant-v4 | 6000 ± 800 | ~4 hours |
| Humanoid-v4 | 7000 ± 1000 | ~24 hours |

## Experiment Tracking

Track experiments with Weights & Biases:

```bash
python train.py --env Ant-v4 --wandb-project sac-robotics
```