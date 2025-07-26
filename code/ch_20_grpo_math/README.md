# Group Relative Policy Optimization (GRPO) for Mathematical Reasoning

This implementation demonstrates GRPO, the algorithm behind DeepSeek's breakthrough in AI reasoning capabilities. GRPO eliminates the value function entirely, using group-based advantage estimation for efficient reinforcement learning of language models.

## Key Features

- **Value-Function-Free**: Eliminates the need for a separate critic network
- **Memory Efficient**: ~50% memory reduction compared to PPO
- **Group-Based Advantages**: Uses relative performance within response groups
- **Format Rewards**: Encourages structured reasoning with `<think></think>` and `<answer></answer>` tags
- **Mathematical Focus**: Optimized for mathematical reasoning tasks

## Project Structure

```
code/ch_20_grpo_math/
├── grpo/                   # Core GRPO implementation
│   ├── __init__.py
│   ├── trainer.py          # Main GRPO trainer
│   ├── rewards.py          # Reward functions (math, code, format)
│   ├── data.py            # Dataset utilities
│   └── utils.py           # Helper functions
├── scripts/               # Training and evaluation
│   ├── train_grpo.py      # Main training script
│   ├── evaluate.py        # Model evaluation
│   └── distill.py         # Knowledge distillation
├── configs/               # Configuration files
│   ├── grpo_math.yaml     # Math reasoning config
│   └── grpo_code.yaml     # Code generation config
├── data/                  # Sample datasets
│   ├── math_problems.json # Mathematical problems
│   └── code_problems.json # Programming problems
└── README.md
```

## Installation

1. **Clone the repository** (or extract this chapter's code)

2. **Install dependencies**:
```bash
pip install torch transformers datasets tqdm pyyaml numpy matplotlib wandb
```

3. **Install optional dependencies for code evaluation**:
```bash
pip install sympy  # For mathematical expression evaluation
```

## Quick Start

### 1. Train a GRPO Model

```bash
cd code/ch_20_grpo_math

# Train on mathematical reasoning
python scripts/train_grpo.py \
  --config configs/grpo_math.yaml \
  --output-dir ./outputs/grpo_math_run1 \
  --wandb-project grpo-experiments

# Train on code generation
python scripts/train_grpo.py \
  --config configs/grpo_code.yaml \
  --output-dir ./outputs/grpo_code_run1
```

### 2. Evaluate the Model

```bash
# Evaluate on test set
python scripts/evaluate.py \
  --model-path ./outputs/grpo_math_run1/final \
  --dataset-path data/math_problems.json \
  --output-dir ./outputs/evaluation_results \
  --num-samples 100
```

### 3. Distill to a Smaller Model

```bash
# Create a distilled version
python scripts/distill.py \
  --teacher-model ./outputs/grpo_math_run1/final \
  --student-model microsoft/DialoGPT-small \
  --dataset-path data/math_problems.json \
  --output-dir ./outputs/distilled_model \
  --num-samples 500
```

## Configuration

### Key GRPO Parameters

- **`group_size`**: Number of responses generated per prompt (default: 8)
- **`clip_epsilon`**: PPO clipping parameter (default: 0.2)
- **`kl_coef`**: KL divergence penalty weight (default: 0.04)
- **`temperature`**: Generation temperature (default: 0.8)

### Reward Function Configuration

- **`format_reward_weight`**: Weight for format compliance (default: 0.1)
- **`correct_reward_weight`**: Weight for answer correctness (default: 1.0)
- **`partial_credit`**: Enable partial credit for reasonable attempts (default: true)

## Understanding the Output

### Training Logs

The trainer outputs several key metrics:

- **`loss`**: Total GRPO loss
- **`mean_reward`**: Average reward across response groups
- **`kl_div`**: KL divergence from reference model
- **`clip_fraction`**: Fraction of advantages that were clipped

### Evaluation Metrics

- **`mean_reward`**: Average reward on evaluation set
- **`format_compliance_rate`**: Percentage of responses following the format
- **`high_reward_rate`**: Percentage of responses with reward > 0.8

### Example Response Format

GRPO trains the model to use this structured format:

```
<think>
Let me solve this step by step.
First, I need to identify what type of problem this is.
This is asking for the area of a rectangle.
The formula for area is length × width.
Given: length = 8 cm, width = 5 cm
Area = 8 × 5 = 40 cm²
</think>

<answer>
40
</answer>
```

## Memory Requirements

### Compared to PPO

For a 7B parameter model:

| Component | PPO | GRPO | Savings |
|-----------|-----|------|---------|
| Policy | 14GB | 14GB | - |
| Value Model | 14GB | 0GB | 14GB |
| Reference | 14GB | 14GB | - |
| Optimizer | 56GB | 28GB | 28GB |
| **Total** | **98GB** | **56GB** | **42GB (43%)** |

### GPU Requirements

- **Minimum**: 24GB GPU (RTX 3090/4090) for small models
- **Recommended**: 40GB GPU (A100) for 7B models
- **Optimal**: 80GB GPU (A100) for larger models and bigger batches

## Advanced Usage

### Custom Reward Functions

```python
from grpo import FormatReward, CompositeRewardFunction

# Create custom reward
def custom_math_reward(prompt, response):
    # Your custom logic here
    return reward_score

# Combine multiple rewards
composite_reward = CompositeRewardFunction({
    'format': (FormatReward(weight=0.1), 0.3),
    'custom': (custom_math_reward, 0.7)
})
```

### Multi-GPU Training

For distributed training, modify the configuration:

```yaml
# In your config file
device: "cuda"
fp16: true
gradient_accumulation_steps: 4
batch_size: 1  # Per GPU
```

Then run with:
```bash
torchrun --nproc_per_node=4 scripts/train_grpo.py --config configs/grpo_math.yaml
```

### Monitoring with Weights & Biases

```bash
python scripts/train_grpo.py \
  --config configs/grpo_math.yaml \
  --output-dir ./outputs/experiment1 \
  --wandb-project my-grpo-project
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `group_size`
2. **Low Rewards**: Check reward function implementation
3. **Poor Format Compliance**: Increase `format_reward_weight`
4. **High KL Divergence**: Reduce `learning_rate` or increase `kl_coef`

### Performance Tips

1. **Use FP16**: Enable `fp16: true` for memory savings
2. **Tune Group Size**: Smaller groups (4-6) for harder problems
3. **Gradient Accumulation**: Increase steps if batch size is too small
4. **Warmup**: Use sufficient warmup steps for stable training

## Examples and Results

The `data/` directory contains sample problems of varying difficulty:

- **Easy**: Basic arithmetic, simple algebra
- **Medium**: Quadratic equations, geometry, word problems  
- **Hard**: Calculus, proofs, complex combinatorics

Training typically shows:
- **Format compliance**: 95%+ after 1000 steps
- **Mean reward improvement**: 0.2 → 0.6+ over training
- **Response length**: Automatically adapts to problem complexity