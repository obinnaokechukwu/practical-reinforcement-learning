# PPO Implementation for Continuous Control

This is a robust, production-ready implementation of Proximal Policy Optimization (PPO) for continuous control tasks in Gymnasium (MuJoCo environments).

## Features

- **Modular Design**: Separate modules for networks, buffer, agent, and utilities
- **Multiple Network Architectures**: MLP, CNN, and discrete action support
- **Advanced Features**:
  - Generalized Advantage Estimation (GAE)
  - Observation and reward normalization
  - Learning rate scheduling (linear, cosine)
  - Early stopping based on KL divergence
  - Gradient clipping
  - Value function clipping
- **Parallel Environments**: Efficient data collection with vectorized environments
- **Comprehensive Logging**: Detailed training statistics and episode metrics
- **Evaluation Tools**: Scripts for testing, visualizing value function, and analyzing policy

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with:

```
gymnasium[mujoco]>=0.28.0
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.6.0
pyyaml>=6.0
tqdm>=4.65.0
```

## Quick Start

### Training

Train PPO on HalfCheetah with default settings:

```bash
python train.py --env HalfCheetah-v4
```

Train with custom hyperparameters:

```bash
python train.py --env Walker2d-v4 \
    --total-timesteps 2000000 \
    --learning-rate 3e-4 \
    --num-envs 16 \
    --lr-schedule cosine
```

Resume training from checkpoint:

```bash
python train.py --env HalfCheetah-v4 \
    --checkpoint ppo_HalfCheetah-v4_1000000.pt
```

### Evaluation

Evaluate a trained agent:

```bash
python evaluate.py ppo_HalfCheetah-v4_final.pt --env HalfCheetah-v4
```

Evaluate with rendering:

```bash
python evaluate.py ppo_HalfCheetah-v4_final.pt --env HalfCheetah-v4 --render
```

Record evaluation videos:

```bash
python evaluate.py ppo_HalfCheetah-v4_final.pt --env HalfCheetah-v4 --record
```

Analyze the learned value function:

```bash
python evaluate.py ppo_HalfCheetah-v4_final.pt --env HalfCheetah-v4 --analyze-value
```

## Configuration Files

Pre-configured settings for different environments are provided in the `configs/` directory:

- `halfcheetah.yaml`: Optimized for HalfCheetah-v4
- `walker2d.yaml`: Optimized for Walker2d-v4
- `humanoid.yaml`: Optimized for Humanoid-v4 (complex task)

To use a configuration file:

```bash
python train.py --config configs/humanoid.yaml
```

## Key Hyperparameters

### PPO Specific
- `clip_epsilon`: PPO clipping parameter (default: 0.2)
- `epochs`: Number of optimization epochs per update (default: 10)
- `mini_batch_size`: Size of mini-batches for optimization (default: 64)

### Training
- `num_envs`: Number of parallel environments (default: 8)
- `num_steps`: Steps per environment before update (default: 2048)
- `learning_rate`: Initial learning rate (default: 3e-4)
- `total_timesteps`: Total training steps (default: 1,000,000)

### GAE
- `gamma`: Discount factor (default: 0.99)
- `gae_lambda`: GAE parameter for bias-variance tradeoff (default: 0.95)

### Regularization
- `entropy_coef`: Entropy bonus coefficient (default: 0.01)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 0.5)
- `value_loss_coef`: Value function loss coefficient (default: 0.5)

## Algorithm Details

This implementation includes:

1. **Clipped Surrogate Objective**: Prevents large policy updates
2. **Value Function Clipping**: Optional clipping for value function updates
3. **Advantage Normalization**: Per-batch normalization of advantages
4. **Observation Normalization**: Running mean/std normalization
5. **Reward Normalization**: Optional reward scaling based on returns
6. **Early Stopping**: Stops updates if KL divergence exceeds threshold

## Extending the Code

### Adding Custom Environments

1. Ensure your environment follows the Gymnasium API
2. Register it with Gymnasium if needed
3. Use the standard training script

### Custom Network Architectures

Add your architecture to `networks.py`:

```python
class CustomPPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, ...):
        # Your implementation
        pass
```

### Custom Reward Shaping

Modify the environment wrapper in `utils.py` or create a new wrapper:

```python
class CustomRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Modify reward here
        shaped_reward = self.shape_reward(reward, obs)
        return obs, shaped_reward, done, truncated, info
```

## Troubleshooting

### Poor Performance
- Try different learning rate schedules (linear, cosine)
- Adjust `num_steps` and `num_envs` for more diverse data
- Tune `clip_epsilon` (try 0.1 or 0.3)
- Enable entropy bonus for exploration

### Training Instability
- Reduce learning rate
- Enable value function clipping
- Decrease `epochs` or enable early stopping
- Check observation/reward scales

### High Variance in Returns
- Increase `num_envs` for more stable statistics
- Use longer training (`total_timesteps`)
- Adjust GAE parameters (`gae_lambda`)

## Performance Benchmarks

Expected performance after 2M steps (8 parallel environments):

| Environment | Mean Return | Training Time |
|-------------|-------------|---------------|
| HalfCheetah-v4 | ~5000-6000 | ~2 hours |
| Walker2d-v4 | ~3000-4000 | ~2 hours |
| Humanoid-v4 | ~5000-6000 | ~10 hours* |

*Humanoid requires 10M steps for good performance
