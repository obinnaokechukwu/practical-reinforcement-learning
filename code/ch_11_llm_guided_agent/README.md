# LLM-Guided Reinforcement Learning Framework

This directory contains a complete implementation of an LLM-guided RL system that integrates language models with reinforcement learning for more intuitive and efficient agent training.

## Overview

The framework combines three key components:

1. **Language-based Reward Models** - Use LLMs to interpret natural language task descriptions and provide reward signals
2. **LLM-guided Exploration** - Leverage language models to suggest interesting states and actions for efficient exploration  
3. **Hierarchical Policy Generation** - Decompose complex tasks into reusable skills using language understanding

## Components

### `llm_reward_model.py`
- `LanguageRewardModel`: Basic reward model using BERT/GPT to evaluate state-action pairs
- `RobustLanguageRewardModel`: Enhanced model with adversarial detection and uncertainty estimation
- Preference learning from human comparisons using Bradley-Terry model

### `exploration_guide.py`
- `LLMExplorationGuide`: Suggests exploration goals and computes curiosity bonuses
- `StateMemory`: Tracks visited states and computes novelty
- `CuriosityDrivenAgent`: RL agent that uses LLM guidance for exploration

### `hierarchical_policy.py`
- `LLMPolicyGenerator`: Generates skill policies from language descriptions
- `SkillLibrary`: Manages reusable skills with usage statistics
- `HierarchicalPolicy`: Executes high-level skill sequences

### `integrated_llm_agent.py`
- `IntegratedLLMAgent`: Complete system combining all components
- Natural language task specification
- Multi-objective reward optimization
- Automatic curriculum generation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from integrated_llm_agent import IntegratedLLMAgent, LLMGuidedConfig

# Configure agent
config = LLMGuidedConfig(
    env_name="CartPole-v1",
    llm_model="gpt2",
    use_reward_model=True,
    use_exploration_guide=True,
    use_skill_decomposition=True
)

# Create agent
agent = IntegratedLLMAgent(config)

# Define task in natural language
task = "Balance the pole by moving smoothly and keeping it upright"

# Train
agent.train(task, num_episodes=1000)

# Evaluate
results = agent.evaluate(task, num_episodes=10)
```

### Training with Custom Rewards

```python
from llm_reward_model import RobustLanguageRewardModel

# Create custom reward model
reward_model = RobustLanguageRewardModel(
    model_name="bert-base-uncased",
    state_dim=4,
    action_dim=2
)

# Train from preferences
from preference_data import load_preferences
preferences = load_preferences()
reward_model.train_from_preferences(preferences)
```

### Hierarchical Task Decomposition

```python
from hierarchical_policy import LLMPolicyGenerator

# Create policy generator
generator = LLMPolicyGenerator(
    base_model="gpt2",
    env_description="CartPole balancing task"
)

# Decompose complex task
task = "First center the cart, then make small adjustments to balance"
hierarchical_policy = generator.create_hierarchical_policy(task)
```

## Key Features

1. **Natural Language Task Specification**: Define RL tasks using plain English instead of reward functions
2. **Intelligent Exploration**: LLM suggests promising states to explore based on task understanding
3. **Skill Composition**: Automatically decompose complex tasks into reusable sub-skills
4. **Robust Rewards**: Defense against reward hacking with uncertainty estimation
5. **Curriculum Learning**: Automatic generation of learning curricula

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Human Task Description                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    LLM Components                        │
├─────────────────┬──────────────┬────────────────────────┤
│  Reward Model   │ Exploration  │  Policy Generator      │
│  (Language →    │   Guide      │  (Task → Skills)       │
│   Rewards)      │ (Curiosity)  │                        │
└────────┬────────┴──────┬───────┴────────┬───────────────┘
         │               │                 │
         ▼               ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│                    RL Agent                              │
│  • Combines extrinsic + intrinsic + language rewards    │
│  • Executes hierarchical policies                       │
│  • Adapts based on experience                           │
└─────────────────────────────────────────────────────────┘
```

## Experiments

Run the example experiments:

```bash
# Basic CartPole with LLM guidance
python integrated_llm_agent.py

# Compare with/without LLM components
python experiments/ablation_study.py

# Test on different environments
python experiments/multi_env_test.py
```

## Configuration Options

- `env_name`: Gym environment name
- `llm_model`: Pre-trained language model to use
- `use_reward_model`: Enable language-based rewards
- `use_exploration_guide`: Enable curiosity-driven exploration
- `use_skill_decomposition`: Enable hierarchical policies
- `intrinsic_reward_scale`: Weight for curiosity bonuses
- `exploration_fraction`: Fraction of exploratory actions

## Extending the Framework

1. **Add New Skills**: Extend `SkillLibrary` with domain-specific skills
2. **Custom Reward Models**: Subclass `LanguageRewardModel` for task-specific rewards
3. **Alternative LLMs**: Replace GPT-2 with more powerful models (GPT-3, Claude, etc.)
4. **Multi-modal Integration**: Add vision-language models for visual tasks

## Limitations

- Computational cost of LLM inference
- Quality depends on language model capabilities
- May require fine-tuning for specific domains
- Currently supports discrete action spaces

## Future Work

- Continuous action space support
- Multi-agent coordination through language
- Real-time adaptation of language instructions
- Integration with larger language models