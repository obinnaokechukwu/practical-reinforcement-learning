# Deep Q-Networks (DQN) Implementation

This code implements the Deep Q-Network algorithm for solving CartPole-v1, accompanying Chapter 6 of "Practical Reinforcement Learning".

## Contents

- `dqn.py`: Complete DQN implementation with experience replay and target networks
- `experiments.py`: Experiments demonstrating key concepts and ablations
- `requirements.txt`: Python dependencies

## Features

### Core DQN Components
1. **Neural Network Q-Function**: Two hidden layers with ReLU activation
2. **Experience Replay**: Both uniform and prioritized variants
3. **Target Network**: Stabilizes learning with periodic updates
4. **ε-Greedy Exploration**: With exponential decay schedule
5. **Gradient Clipping**: Prevents exploding gradients

### Advanced Features
- Prioritized experience replay based on TD error
- Importance sampling weights for bias correction
- Soft target network updates (optional)
- Comprehensive training metrics and visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from dqn import train_dqn

# Train DQN on CartPole
agent, rewards, lengths = train_dqn(env_name='CartPole-v1', num_episodes=500)

# Evaluate trained agent
from dqn import evaluate_agent
evaluate_agent(agent, num_episodes=10, render=True)
```

### Custom Configuration

```python
from dqn import DQNAgent
import gym

env = gym.make('CartPole-v1')
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=1e-3,                  # Learning rate
    gamma=0.99,               # Discount factor
    epsilon=1.0,              # Initial exploration rate
    epsilon_decay=0.995,      # Exploration decay
    buffer_size=10000,        # Replay buffer size
    batch_size=64,            # Training batch size
    target_update=10,         # Target network update frequency
    use_prioritized_replay=True
)
```

### Run All Experiments

```bash
python experiments.py
```

This generates several visualizations:
- `q_value_divergence.png`: Demonstrates instability without target networks
- `replay_comparison.png`: Uniform vs prioritized experience replay
- `ablation_study.png`: Impact of each DQN component
- `target_update_frequency.png`: Effect of target network update rate
- `q_function_visualization.png`: Learned Q-values across state space

## Experiments Explained

### 1. Neural Q-Learning Instability
Shows how Q-values explode without proper stabilization techniques, demonstrating the need for target networks and experience replay.

### 2. Experience Replay Comparison
Compares uniform sampling vs prioritized experience replay. Prioritized replay focuses on surprising transitions, often leading to faster learning.

### 3. Component Ablation Study
Tests four configurations:
- Full DQN (replay + target network)
- No target network
- No experience replay  
- Vanilla neural Q-learning

Results clearly show both components are essential for stable learning.

### 4. Target Update Frequency
Shows how different update frequencies affect learning:
- Too frequent (1 step): Less stable targets
- Too infrequent (100 steps): Slow learning
- Sweet spot: 5-20 steps for CartPole

### 5. Q-Function Visualization
Visualizes the learned Q-function across the state space, showing:
- Q-values for each action
- Action preferences (Q-value differences)
- Optimal policy regions

## Key Insights

1. **Stability is Crucial**: Neural networks can easily destabilize Q-learning without proper techniques
2. **Experience Replay**: Breaks correlations and enables data reuse
3. **Target Networks**: Provide stable targets for more reliable learning
4. **Prioritized Replay**: Can significantly speed up learning by focusing on important experiences
5. **Hyperparameters Matter**: Target update frequency, learning rate, and batch size significantly impact performance

## CartPole-v1 Benchmark

The environment is considered "solved" when the average reward over 100 consecutive episodes is ≥ 195.0. Our DQN typically solves it in:
- 150-250 episodes with proper hyperparameters
- 100-150 episodes with prioritized replay
- May fail to solve without target networks or replay

## Customization

### Different Environments

```python
# Train on different gym environments
agent, rewards, _ = train_dqn(env_name='MountainCar-v0', num_episodes=1000)
```

### Network Architecture

Modify the `DQNNetwork` class:
```python
class CustomDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
```

### Advanced Techniques

The codebase can be extended with:
- Double DQN: Decouple action selection and evaluation
- Dueling DQN: Separate value and advantage streams
- Rainbow DQN: Combine multiple improvements
- Noisy networks: Learnable exploration

## Troubleshooting

1. **Q-values exploding**: Reduce learning rate, increase target update frequency
2. **Slow learning**: Increase batch size, use prioritized replay
3. **High variance**: Increase buffer size, reduce epsilon decay rate
4. **Not solving**: Check reward clipping, network initialization

## References

- Original DQN paper: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- Prioritized replay: "Prioritized Experience Replay" (Schaul et al., 2016)
- Improvements: "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2018)