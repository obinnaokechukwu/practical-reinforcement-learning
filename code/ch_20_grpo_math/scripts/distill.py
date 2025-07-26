#!/usr/bin/env python3
"""
Distillation script for GRPO-trained models.
Creates smaller, faster models that maintain reasoning capabilities.
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm
import numpy as np

# Add the grpo package to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from grpo.utils import extract_thinking, extract_answer


class DistillationDataset(Dataset):
    """Dataset for distillation from teacher model responses."""
    
    def __init__(self, prompts, teacher_responses, tokenizer, max_length=2048):
        self.prompts = prompts
        self.teacher_responses = teacher_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare training examples
        self.examples = []
        for prompt, response in zip(prompts, teacher_responses):
            # Create input-output pair
            full_text = f"{prompt}\n\n{response}"
            
            # Tokenize
            tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors='pt'
            )
            
            self.examples.append({
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': tokens['input_ids'].squeeze().clone()
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def generate_teacher_responses(teacher_model, teacher_tokenizer, 
                             prompts, device='cuda', 
                             temperature=0.1, max_length=2048):
    """Generate high-quality responses from teacher model."""
    teacher_responses = []
    
    print(f"Generating teacher responses for {len(prompts)} prompts...")
    
    for prompt in tqdm(prompts, desc="Generating"):
        # Tokenize prompt
        inputs = teacher_tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = teacher_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_k=50,
                top_p=0.95,
                pad_token_id=teacher_tokenizer.pad_token_id,
                eos_token_id=teacher_tokenizer.eos_token_id
            )
        
        # Decode response (remove prompt)
        response = teacher_tokenizer.decode(
            outputs[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        teacher_responses.append(response)
    
    return teacher_responses


def filter_high_quality_responses(prompts, responses, reward_fn, 
                                min_reward=0.5):
    """Filter responses based on quality threshold."""
    filtered_prompts = []
    filtered_responses = []
    
    print(f"Filtering responses with minimum reward {min_reward}...")
    
    for prompt, response in tqdm(zip(prompts, responses), 
                                desc="Filtering", total=len(prompts)):
        reward = reward_fn(prompt, response)
        
        if reward >= min_reward:
            filtered_prompts.append(prompt)
            filtered_responses.append(response)
    
    print(f"Kept {len(filtered_prompts)}/{len(prompts)} responses "
          f"({len(filtered_prompts)/len(prompts):.1%})")
    
    return filtered_prompts, filtered_responses


class DistillationTrainer(Trainer):
    """Custom trainer for distillation with additional losses."""
    
    def __init__(self, *args, teacher_model=None, alpha=0.7, temperature=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha  # Weight for distillation loss
        self.temperature = temperature  # Temperature for soft targets
        
        if self.teacher_model:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute distillation loss combining CE and KL losses."""
        labels = inputs.get("labels")
        
        # Student forward pass
        outputs = model(**inputs)
        student_logits = outputs.get('logits')
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # If teacher model is available, add distillation loss
        total_loss = ce_loss
        
        if self.teacher_model:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.get('logits')
            
            # Soft targets with temperature
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            # KL divergence loss
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Combine losses
            total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return (total_loss, outputs) if return_outputs else total_loss


def setup_student_model(teacher_model_path, student_model_name, device='cuda'):
    """Set up student model (potentially smaller) for distillation."""
    # Load teacher for reference
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
    
    # Load student model (could be smaller)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    # Set pad tokens
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    
    return teacher_model, teacher_tokenizer, student_model, student_tokenizer


def main():
    parser = argparse.ArgumentParser(description='Distill GRPO-trained model')
    parser.add_argument('--teacher-model', type=str, required=True,
                       help='Path to teacher model checkpoint')
    parser.add_argument('--student-model', type=str, required=True,
                       help='Student model name or path')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset for distillation')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save distilled model')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to use for distillation')
    parser.add_argument('--min-reward', type=float, default=0.5,
                       help='Minimum reward threshold for response filtering')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Weight for distillation loss (vs CE loss)')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Temperature for soft targets')
    parser.add_argument('--generation-temperature', type=float, default=0.1,
                       help='Temperature for teacher response generation')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--save-steps', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading teacher and student models...")
    teacher_model, teacher_tokenizer, student_model, student_tokenizer = setup_student_model(
        args.teacher_model, args.student_model, args.device
    )
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    from grpo import MathDataset
    dataset = MathDataset(args.dataset_path)
    
    # Sample subset if specified
    if args.num_samples:
        indices = np.random.choice(
            len(dataset), 
            min(args.num_samples, len(dataset)), 
            replace=False
        )
        prompts = [dataset[i]['prompt'] for i in indices]
    else:
        prompts = [dataset[i]['prompt'] for i in range(len(dataset))]
    
    print(f"Using {len(prompts)} prompts for distillation")
    
    # Generate teacher responses
    teacher_responses = generate_teacher_responses(
        teacher_model, teacher_tokenizer, prompts,
        device=args.device,
        temperature=args.generation_temperature
    )
    
    # Filter high-quality responses
    from grpo import MathRewardFunction, FormatReward
    reward_fn = MathRewardFunction(
        format_reward=FormatReward(weight=0.1),
        correct_weight=1.0
    )
    
    filtered_prompts, filtered_responses = filter_high_quality_responses(
        prompts, teacher_responses, reward_fn, args.min_reward
    )
    
    # Create distillation dataset
    print("Creating distillation dataset...")
    distill_dataset = DistillationDataset(
        filtered_prompts, filtered_responses, student_tokenizer
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True if args.device == 'cuda' else False,
        report_to=None  # Disable wandb for now
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=distill_dataset,
        data_collator=data_collator,
        teacher_model=teacher_model,
        alpha=args.alpha,
        temperature=args.temperature
    )
    
    # Print training info
    print(f"\nDistillation Configuration:")
    print(f"  Teacher: {args.teacher_model}")
    print(f"  Student: {args.student_model}")
    print(f"  Training samples: {len(distill_dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Alpha (distill weight): {args.alpha}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Epochs: {args.num_epochs}")
    print()
    
    # Train
    print("Starting distillation training...")
    trainer.train()
    
    # Save final model
    print("Saving final distilled model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    student_tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    
    # Save training info
    config = {
        'teacher_model': args.teacher_model,
        'student_model': args.student_model,
        'training_samples': len(distill_dataset),
        'min_reward_threshold': args.min_reward,
        'alpha': args.alpha,
        'temperature': args.temperature,
        'generation_temperature': args.generation_temperature,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs
    }
    
    with open(os.path.join(args.output_dir, "distillation_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save some example teacher responses for inspection
    examples_path = os.path.join(args.output_dir, "teacher_examples.json")
    examples = [
        {
            'prompt': p,
            'teacher_response': r,
            'thinking': extract_thinking(r),
            'answer': extract_answer(r)
        }
        for p, r in zip(filtered_prompts[:10], filtered_responses[:10])
    ]
    
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"\nDistillation completed! Model saved to {args.output_dir}")
    print(f"Training config saved to distillation_config.json")
    print(f"Teacher examples saved to teacher_examples.json")


if __name__ == "__main__":
    main()