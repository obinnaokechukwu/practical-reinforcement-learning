# RLHF Pipeline Implementation

A complete implementation of Reinforcement Learning from Human Feedback (RLHF) for training helpful, harmless, and honest language models.

## Overview

This implementation follows the three-stage RLHF pipeline:

1. **Supervised Fine-Tuning (SFT)**: Fine-tune a base language model on high-quality demonstrations
2. **Reward Model Training**: Train a model to predict human preferences from comparison data
3. **PPO Training**: Use reinforcement learning to optimize the language model against the reward model

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file:

```
transformers>=4.30.0
torch>=2.0.0
datasets>=2.10.0
accelerate>=0.20.0
trl>=0.7.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
pyyaml>=6.0
wandb  # Optional, for experiment tracking
```

## Quick Start

### 1. Prepare Your Data

The pipeline expects three types of data:

**SFT Data** (`data/sft_data.json`):
```json
[
  {
    "prompt": "What is machine learning?",
    "response": "Machine learning is..."
  }
]
```

**Preference Data** (`data/preference_data.json`):
```json
[
  {
    "prompt": "Explain quantum computing",
    "chosen": "Quantum computing leverages...",
    "rejected": "Quantum computers are magic..."
  }
]
```

**Prompts** (`data/prompts.json`):
```json
[
  "What is artificial intelligence?",
  "How can I learn a new language?"
]
```

### 2. Run the Pipeline

#### Stage 1: Supervised Fine-Tuning
```bash
python scripts/train_sft.py \
  --model_name gpt2 \
  --data_path data/sft_data.json \
  --output_dir models/sft_model \
  --num_epochs 3 \
  --batch_size 8
```

#### Stage 2: Reward Model Training
```bash
python scripts/train_reward_model.py \
  --base_model models/sft_model \
  --data_path data/preference_data.json \
  --output_dir models/reward_model \
  --num_epochs 3 \
  --batch_size 4
```

#### Stage 3: RLHF with PPO
```bash
python scripts/train_rlhf.py \
  --sft_model_path models/sft_model \
  --reward_model_path models/reward_model \
  --prompts_path data/prompts.json \
  --output_dir models/rlhf_model \
  --num_episodes 1000
```

### 3. Using Configuration Files

For easier management, use the provided config file:

```bash
# Edit configs/rlhf_config.yaml with your settings
python scripts/train_pipeline.py --config configs/rlhf_config.yaml
```

## Key Features

### Comprehensive Evaluation
- **Helpfulness**: Measures response quality and informativeness
- **Harmlessness**: Checks refusal of harmful requests
- **Honesty**: Evaluates uncertainty expression and calibration

### Safety Mechanisms
- Reward hacking detection
- KL divergence constraints
- Safety filtering for harmful content

### Training Monitoring
- Real-time metrics tracking
- Training curve visualization
- Alert system for training issues

### Memory Efficiency
- Gradient checkpointing support
- Mixed precision training (fp16)
- LoRA/PEFT support for large models

## Advanced Usage

### Custom Reward Models

Implement your own reward model:

```python
from src.reward_model import RewardModel

class CustomRewardModel(RewardModel):
    def forward(self, input_ids, attention_mask):
        # Your implementation
        return rewards
```

### Ensemble Reward Models

Use multiple reward models for robustness:

```python
from src.reward_model import RewardModelEnsemble

ensemble = RewardModelEnsemble(
    model_paths=['model1', 'model2', 'model3'],
    tokenizer=tokenizer
)
```

### Custom Evaluation Metrics

Add your own evaluation criteria:

```python
from src.utils import RLHFEvaluator

evaluator = RLHFEvaluator(model, tokenizer)
evaluator.eval_categories['custom'] = [
    "Your custom test prompts"
]
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use fp16 training
   - Consider using LoRA for large models

2. **Reward Hacking**
   - Increase KL penalty coefficient
   - Use reward model ensemble
   - Add explicit length penalties
   - Monitor response diversity

3. **Mode Collapse**
   - Increase entropy coefficient
   - Use temperature sampling
   - Add diversity rewards
   - Check for repetitive responses

4. **Poor Performance**
   - Ensure quality of SFT data
   - Verify preference data quality
   - Tune PPO hyperparameters
   - Check for proper reward scaling

## Experiment Tracking

Enable Weights & Biases logging:

```bash
export WANDB_PROJECT="rlhf-experiments"
python scripts/train_rlhf.py --use_wandb
```

## Model Deployment

After training, deploy your RLHF model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("models/rlhf_model/policy")
tokenizer = AutoTokenizer.from_pretrained("models/rlhf_model/tokenizer")

# Generate responses
prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```
