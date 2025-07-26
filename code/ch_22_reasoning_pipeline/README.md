# Multi-Stage RL Pipeline for Reasoning Model Development

This implementation demonstrates the complete pipeline for developing reasoning models through reinforcement learning, as pioneered by DeepSeek R1 and similar advanced systems.

## Overview

The pipeline implements five key stages:
1. **Pure RL Training**: Train models from scratch without supervised fine-tuning
2. **Chain-of-Thought Emergence**: Develop step-by-step reasoning capabilities
3. **Rejection Sampling**: Generate diverse high-quality solutions
4. **Process Reward Models**: Train models to evaluate reasoning steps
5. **Knowledge Distillation**: Transfer capabilities to smaller models

## Installation

```bash
pip install torch transformers tqdm numpy wandb
pip install gymnasium  # For RL environments
```

## Quick Start

### Basic Usage

```python
from reasoning_pipeline import ReasoningModelPipeline, ReasoningPipelineConfig

# Define your reward function
def math_reward_fn(problem: str, solution: str) -> float:
    # Implement problem-specific evaluation
    # Return score between 0 and 1
    pass

# Configure pipeline
config = ReasoningPipelineConfig(
    base_model_name="EleutherAI/gpt-neo-125M",
    enable_pure_rl=True,
    enable_cot_training=True,
    enable_rejection_sampling=True,
    enable_prm_training=True,
    enable_distillation=True,
    output_dir="./reasoning_output"
)

# Create pipeline
pipeline = ReasoningModelPipeline(config, math_reward_fn)

# Run pipeline
results = pipeline.run_pipeline(
    train_problems=["What is 2+2?", "Solve x: 3x=9", ...],
    test_problems=["What is 5+7?", ...],
    student_model_name="microsoft/DialoGPT-small"  # For distillation
)
```

## Stage Details

### 1. Pure RL Training

Train language models using only reinforcement learning signals:

```python
from pure_rl_trainer import PureRLTrainer, PureRLConfig

config = PureRLConfig(
    model_name="gpt2",
    learning_rate=1e-5,
    temperature_start=1.2,  # High for exploration
    temperature_end=0.3,    # Low for exploitation
    entropy_coef_start=0.1  # Encourage diversity
)

trainer = PureRLTrainer(config, reward_fn)
trainer.train(problems, num_iterations=1000)
```

Key innovations:
- No supervised fine-tuning required
- Temperature and entropy scheduling
- KL divergence constraints
- Value head for advantage estimation

### 2. Chain-of-Thought Training

Encourage structured reasoning through specialized rewards:

```python
from cot_emergence import ChainOfThoughtTrainer, CoTConfig

config = CoTConfig(
    model_name="gpt2",
    thinking_reward_weight=0.3,  # Reward thinking process
    answer_reward_weight=0.7,    # Reward final answer
    force_thinking_steps=3       # Minimum reasoning steps
)

cot_trainer = ChainOfThoughtTrainer(config, reward_fn)
cot_trainer.train_with_cot_rewards(problems)
```

Features:
- Separate rewards for thinking vs answers
- Step clarity and logical flow evaluation
- Pattern analysis (step-based, causal, etc.)

### 3. Rejection Sampling

Generate diverse high-quality solutions:

```python
from rejection_sampling import DiversityAwareSampler, RejectionSamplingConfig

config = RejectionSamplingConfig(
    num_samples=16,
    min_score_threshold=0.7,
    diversity_bonus=0.1
)

sampler = DiversityAwareSampler(model, tokenizer, reward_fn, config)

# Get diverse solutions
solutions = sampler.sample_diverse_solutions(problem, num_diverse=5)

# Create ensemble
ensemble = sampler.ensemble_solutions(problem, solutions)
```

Capabilities:
- MMR-based diverse sampling
- Iterative refinement
- Ensemble generation
- Beam search with rejection

### 4. Process Reward Models

Train models to evaluate reasoning quality step-by-step:

```python
from process_rewards import PRMTrainer, ProcessRewardEvaluator, PRMConfig

# Generate training data
generator = PRMDataGenerator(outcome_reward_fn, config)
prm_data = generator.generate_prm_dataset(solved_problems)

# Train PRM
trainer = PRMTrainer(config)
trainer.train(prm_data)

# Use for evaluation
evaluator = ProcessRewardEvaluator(prm_model, tokenizer)
result = evaluator.evaluate_solution(problem, solution)
print(f"Weakest step: {result['weakest_step']}")
```

Features:
- Automatic perturbation generation
- Step-wise quality assessment
- Context-aware evaluation
- Comparative solution ranking

### 5. Knowledge Distillation

Transfer reasoning capabilities to smaller models:

```python
from distillation import ProgressiveDistiller, DistillationConfig

config = DistillationConfig(
    teacher_model_name="./trained_model",
    student_model_name="gpt2-small",
    temperature=3.0,
    curriculum_learning=True
)

distiller = ProgressiveDistiller(config)
distiller.progressive_distillation(problems)
```

Innovations:
- Progressive curriculum learning
- Reasoning preservation metrics
- Custom distillation loss
- Difficulty-aware training

## Complete Example

See `examples/full_pipeline.py` for a complete working example:

```python
# Mathematical reasoning example
math_problems = [
    "Find the derivative of f(x) = x^2 + 3x",
    "Solve the system: 2x + y = 5, x - y = 1",
    "What is the integral of sin(x)cos(x)?",
    # ... more problems
]

# Run full pipeline
pipeline = ReasoningModelPipeline(config, math_reward_fn)
results = pipeline.run_pipeline(
    train_problems=math_problems,
    test_problems=test_math_problems
)

# Access trained models
pure_rl_model = results['models']['pure_rl']
cot_model = results['models']['cot']
prm_model = results['models']['prm']
distilled_model = results['models']['distilled']
```

## Evaluation Metrics

The pipeline tracks comprehensive metrics:

- **Reasoning Quality**: Step clarity, logical flow, completeness
- **Solution Quality**: Correctness, efficiency, robustness
- **Diversity**: Pattern distribution, solution variety
- **Efficiency**: Training time, model size, inference speed

## Advanced Configuration

### Custom Reward Functions

```python
class CompoundRewardFunction:
    def __init__(self):
        self.correctness_weight = 0.5
        self.clarity_weight = 0.3
        self.efficiency_weight = 0.2
    
    def __call__(self, problem: str, solution: str) -> float:
        correctness = self.evaluate_correctness(problem, solution)
        clarity = self.evaluate_clarity(solution)
        efficiency = self.evaluate_efficiency(solution)
        
        return (self.correctness_weight * correctness +
                self.clarity_weight * clarity +
                self.efficiency_weight * efficiency)
```

### Multi-Stage Rewards

```python
# Different rewards for different training stages
stage_rewards = {
    'exploration': lambda p, s: diversity_reward(p, s),
    'exploitation': lambda p, s: accuracy_reward(p, s),
    'refinement': lambda p, s: combined_reward(p, s)
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Convergence**: Adjust learning rates and temperature schedules
3. **Poor Diversity**: Increase entropy coefficient and diversity bonus
4. **Weak Reasoning**: Extend CoT training iterations

### Performance Tips

- Use mixed precision training (fp16) for efficiency
- Implement gradient checkpointing for large models
- Cache model outputs during rejection sampling
- Parallelize PRM data generation

## References

- DeepSeek R1 Technical Report
- Constitutional AI (Anthropic)
- Process Reward Models (OpenAI)
- Knowledge Distillation for LLMs