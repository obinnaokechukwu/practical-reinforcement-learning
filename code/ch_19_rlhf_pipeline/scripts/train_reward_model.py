#!/usr/bin/env python3
"""
Script to train the reward model using preference data.
Second stage of the RLHF pipeline.
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoConfig
from src.reward_model import RewardModel, RewardModelTrainer
from src.dataset import create_preference_dataloader
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Train reward model for RLHF")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to SFT model or base model name")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to preference data")
    parser.add_argument("--output_dir", type=str, default="./models/reward_model",
                        help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--eval_split", type=float, default=0.1,
                        help="Fraction of data to use for evaluation")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config and create reward model
    print("Creating reward model...")
    config = AutoConfig.from_pretrained(args.base_model)
    reward_model = RewardModel(config, base_model_name_or_path=args.base_model)
    
    # Create trainer
    trainer = RewardModelTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Load dataset
    print(f"Loading preference data from: {args.data_path}")
    full_dataloader = create_preference_dataloader(
        data_path=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False  # We'll split first
    )
    
    # Split into train and eval
    dataset = full_dataloader.dataset
    eval_size = int(len(dataset) * args.eval_split)
    train_size = len(dataset) - eval_size
    
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=full_dataloader.collate_fn
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=full_dataloader.collate_fn
    )
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Training loop
    print("Starting training...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    best_eval_accuracy = 0
    train_losses = []
    eval_accuracies = []
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Training
        epoch_losses = []
        epoch_accuracies = []
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            metrics = trainer.train_step(batch)
            
            epoch_losses.append(metrics['loss'])
            epoch_accuracies.append(metrics['accuracy'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.3f}",
                'margin': f"{metrics['reward_margin']:.3f}"
            })
            
            # Log metrics
            if (batch_idx + 1) % args.logging_steps == 0:
                avg_loss = sum(epoch_losses[-args.logging_steps:]) / args.logging_steps
                avg_acc = sum(epoch_accuracies[-args.logging_steps:]) / args.logging_steps
                trainer.training_history.append({
                    'step': trainer.global_step,
                    'loss': avg_loss,
                    'accuracy': avg_acc
                })
            
            # Save checkpoint
            if (batch_idx + 1) % args.save_steps == 0:
                checkpoint_dir = f"{args.output_dir}/checkpoint-{trainer.global_step}"
                trainer.save_model(checkpoint_dir)
        
        # Evaluation
        print("Evaluating...")
        eval_metrics = trainer.evaluate(eval_dataloader)
        
        print(f"Eval metrics: {eval_metrics}")
        eval_accuracies.append(eval_metrics['eval_accuracy'])
        
        # Save best model
        if eval_metrics['eval_accuracy'] > best_eval_accuracy:
            best_eval_accuracy = eval_metrics['eval_accuracy']
            trainer.save_model(f"{args.output_dir}/best_model")
            print(f"New best model saved with accuracy: {best_eval_accuracy:.3f}")
    
    # Save final model
    trainer.save_model(args.output_dir)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    steps = [h['step'] for h in trainer.training_history]
    losses = [h['loss'] for h in trainer.training_history]
    plt.plot(steps, losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    accuracies = [h['accuracy'] for h in trainer.training_history]
    plt.plot(steps, accuracies, label='Train')
    
    # Add eval accuracies
    eval_steps = [i * len(train_dataloader) for i in range(1, len(eval_accuracies) + 1)]
    plt.plot(eval_steps, eval_accuracies, 'o-', label='Eval')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Preference Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/training_curves.png", dpi=150)
    plt.close()
    
    # Save training summary
    summary = {
        "base_model": args.base_model,
        "training_data": args.data_path,
        "num_epochs": args.num_epochs,
        "final_train_accuracy": accuracies[-1] if accuracies else 0,
        "best_eval_accuracy": best_eval_accuracy,
        "total_steps": trainer.global_step,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_length": args.max_length
        }
    }
    
    with open(f"{args.output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining complete! Best eval accuracy: {best_eval_accuracy:.3f}")
    
    # Test the reward model
    print("\nTesting reward model on example comparisons:")
    test_examples = [
        {
            "prompt": "What is the capital of France?",
            "response1": "The capital of France is Paris.",
            "response2": "I don't know."
        },
        {
            "prompt": "How do I bake a cake?",
            "response1": "To bake a cake, you'll need flour, eggs, sugar, and butter. Mix the ingredients, pour into a pan, and bake at 350°F for 30 minutes.",
            "response2": "Baking a cake involves many steps and ingredients."
        }
    ]
    
    reward_model.eval()
    for example in test_examples:
        print(f"\nPrompt: {example['prompt']}")
        
        # Score both responses
        for i, response_key in enumerate(['response1', 'response2']):
            text = f"Human: {example['prompt']}\n\nAssistant: {example[response_key]}"
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=args.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                reward = reward_model.compute_reward(**inputs)
            
            print(f"Response {i+1}: {example[response_key]}")
            print(f"Reward: {reward.item():.3f}")


if __name__ == "__main__":
    main()