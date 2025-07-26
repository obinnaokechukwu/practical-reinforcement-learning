#!/usr/bin/env python3
"""
Train a model using Group Relative Policy Optimization (GRPO).
Implements DeepSeek's approach for mathematical reasoning.
"""

import argparse
import json
import os
import yaml
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from tqdm import tqdm

# Add the grpo package to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from grpo import (
    GRPOTrainer, GRPOConfig,
    MathRewardFunction, FormatReward, CompositeRewardFunction,
    MathDataset, create_data_loader
)
from grpo.utils import (
    save_training_checkpoint, 
    plot_training_metrics,
    log_generation_examples,
    create_grpo_summary_report
)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_reward_function(config: dict) -> callable:
    """Set up the reward function based on configuration."""
    format_reward = FormatReward(weight=config.get('format_reward_weight', 0.1))
    
    if config.get('task_type') == 'math':
        task_reward = MathRewardFunction(
            format_reward=format_reward,
            correct_weight=config.get('correct_reward_weight', 1.0),
            partial_credit=config.get('partial_credit', True)
        )
    else:
        # Default to math for now
        task_reward = MathRewardFunction(
            format_reward=format_reward,
            correct_weight=config.get('correct_reward_weight', 1.0)
        )
    
    return task_reward


def setup_datasets(config: dict):
    """Set up training and validation datasets."""
    # Training dataset
    train_dataset = MathDataset(
        data_path=config['data']['train_path'],
        max_prompt_length=config.get('max_prompt_length', 512),
        problem_types=config['data'].get('problem_types')
    )
    
    train_dataloader = create_data_loader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    # Validation dataset (optional)
    eval_dataloader = None
    if config['data'].get('eval_path'):
        eval_dataset = MathDataset(
            data_path=config['data']['eval_path'],
            max_prompt_length=config.get('max_prompt_length', 512),
            problem_types=config['data'].get('problem_types')
        )
        
        eval_dataloader = create_data_loader(
            eval_dataset,
            batch_size=config.get('eval_batch_size', 8),
            shuffle=False,
            num_workers=config.get('num_workers', 0)
        )
    
    return train_dataloader, eval_dataloader


def setup_model_and_trainer(config: dict, reward_fn: callable):
    """Set up the model and GRPO trainer."""
    # Model configuration
    model_config = GRPOConfig(
        model_name_or_path=config['model']['name_or_path'],
        tokenizer_name_or_path=config['model'].get('tokenizer_name_or_path'),
        
        # GRPO hyperparameters
        group_size=config.get('group_size', 8),
        clip_epsilon=config.get('clip_epsilon', 0.2),
        kl_coef=config.get('kl_coef', 0.04),
        target_kl=config.get('target_kl', 0.01),
        
        # Generation settings
        max_prompt_length=config.get('max_prompt_length', 512),
        max_response_length=config.get('max_response_length', 2048),
        temperature=config.get('temperature', 0.8),
        top_k=config.get('top_k', 50),
        top_p=config.get('top_p', 0.95),
        
        # Training settings
        learning_rate=config.get('learning_rate', 5e-6),
        batch_size=config.get('batch_size', 4),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        warmup_steps=config.get('warmup_steps', 100),
        
        # Optimization
        adam_epsilon=config.get('adam_epsilon', 1e-8),
        weight_decay=config.get('weight_decay', 0.01),
        
        # Device and precision
        device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        fp16=config.get('fp16', False),
        
        # Logging
        log_interval=config.get('log_interval', 10),
        save_interval=config.get('save_interval', 1000)
    )
    
    # Create trainer
    trainer = GRPOTrainer(model_config, reward_fn)
    
    return trainer, model_config


def main():
    parser = argparse.ArgumentParser(description='Train model with GRPO')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save outputs')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--wandb-project', type=str, default=None,
                       help='Weights & Biases project name')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run a few steps for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Initialize Weights & Biases (optional)
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=config
        )
    
    # Set up reward function
    print("Setting up reward function...")
    reward_fn = setup_reward_function(config)
    
    # Set up datasets
    print("Loading datasets...")
    train_dataloader, eval_dataloader = setup_datasets(config)
    
    # Set up model and trainer
    print("Initializing model and trainer...")
    trainer, model_config = setup_model_and_trainer(config, reward_fn)
    
    # Resume from checkpoint if specified
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Print training information
    print(f"\nTraining Configuration:")
    print(f"  Model: {model_config.model_name_or_path}")
    print(f"  Group size: {model_config.group_size}")
    print(f"  Batch size: {model_config.batch_size}")
    print(f"  Learning rate: {model_config.learning_rate}")
    print(f"  Device: {model_config.device}")
    print(f"  Training samples: {len(train_dataloader.dataset)}")
    if eval_dataloader:
        print(f"  Evaluation samples: {len(eval_dataloader.dataset)}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Training parameters
    num_epochs = config.get('num_epochs', 1)
    max_steps = config.get('max_steps', None)
    
    if args.dry_run:
        print("Running in dry-run mode (limited steps)")
        max_steps = 5
        num_epochs = 1
    
    # Training loop
    print("Starting training...")
    training_history = []
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training loop
            epoch_metrics = []
            progress_bar = tqdm(
                train_dataloader, 
                desc=f"Training Epoch {epoch + 1}"
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Extract prompts from batch
                if isinstance(batch, dict):
                    prompts = batch['prompts']
                else:
                    prompts = batch
                
                # Training step
                metrics = trainer.train_step(prompts)
                epoch_metrics.append(metrics)
                training_history.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'reward': f"{metrics['mean_reward']:.3f}",
                    'kl': f"{metrics['kl_div']:.4f}"
                })
                
                # Log to wandb
                if args.wandb_project:
                    wandb.log(metrics)
                
                # Save checkpoint
                if trainer.global_step % model_config.save_interval == 0:
                    checkpoint_path = os.path.join(
                        args.output_dir, 
                        f"checkpoint-{trainer.global_step}"
                    )
                    save_training_checkpoint(
                        trainer, epoch, trainer.global_step, 
                        metrics, args.output_dir
                    )
                    print(f"\nSaved checkpoint to {checkpoint_path}")
                
                # Log examples occasionally
                if trainer.global_step % (model_config.log_interval * 10) == 0:
                    example_log_path = os.path.join(
                        args.output_dir, 
                        f"examples_step_{trainer.global_step}.txt"
                    )
                    # Generate some examples for logging
                    test_prompts = prompts[:2]  # Log first 2 prompts
                    groups = trainer.generate_groups(test_prompts)
                    
                    responses_groups = [group['responses'] for group in groups]
                    rewards_groups = [group['rewards'] for group in groups]
                    
                    log_generation_examples(
                        test_prompts, responses_groups, rewards_groups,
                        example_log_path
                    )
                
                # Early stopping for dry run
                if args.dry_run and batch_idx >= 3:
                    break
                
                # Max steps check
                if max_steps and trainer.global_step >= max_steps:
                    break
            
            # Evaluation
            if eval_dataloader and not args.dry_run:
                print("\nRunning evaluation...")
                eval_metrics = trainer.evaluate(eval_dataloader)
                print(f"Evaluation results: {eval_metrics}")
                
                if args.wandb_project:
                    wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()})
            
            # End of epoch summary
            if epoch_metrics:
                avg_epoch_metrics = {
                    k: sum(m.get(k, 0) for m in epoch_metrics) / len(epoch_metrics)
                    for k in epoch_metrics[0].keys()
                }
                print(f"\nEpoch {epoch + 1} Summary:")
                for k, v in avg_epoch_metrics.items():
                    print(f"  {k}: {v:.4f}")
            
            # Max steps check
            if max_steps and trainer.global_step >= max_steps:
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "final")
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")
    
    # Plot training metrics
    if training_history:
        metrics_plot_path = os.path.join(args.output_dir, "training_metrics.png")
        plot_training_metrics(training_history, metrics_plot_path)
        print(f"Saved training metrics plot to {metrics_plot_path}")
        
        # Save training history
        with open(os.path.join(args.output_dir, "training_history.json"), 'w') as f:
            json.dump(training_history, f, indent=2)
    
    # Final evaluation
    eval_results = {}
    if eval_dataloader and not args.dry_run:
        print("\nRunning final evaluation...")
        eval_results = trainer.evaluate(eval_dataloader)
        print(f"Final evaluation results: {eval_results}")
        
        # Save evaluation results
        with open(os.path.join(args.output_dir, "eval_results.json"), 'w') as f:
            json.dump(eval_results, f, indent=2)
    
    # Create summary report
    summary_path = os.path.join(args.output_dir, "training_summary.txt")
    create_grpo_summary_report(
        training_history, eval_results, config, summary_path
    )
    print(f"Saved training summary to {summary_path}")
    
    # Close wandb
    if args.wandb_project:
        wandb.finish()
    
    print(f"\nTraining completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()