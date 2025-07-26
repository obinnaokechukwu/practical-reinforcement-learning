#!/usr/bin/env python3
"""
Script to train the SFT (Supervised Fine-Tuning) model.
First stage of the RLHF pipeline.
"""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from src.dataset import create_sft_dataloader, SFTDataset
import json


def main():
    parser = argparse.ArgumentParser(description="Train SFT model for RLHF")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Base model name or path")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to SFT training data")
    parser.add_argument("--output_dir", type=str, default="./models/sft_model",
                        help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    train_dataset = SFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    print(f"Dataset size: {len(train_dataset)} examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/tensorboard for now
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training info
    training_info = {
        "base_model": args.model_name,
        "training_data": args.data_path,
        "num_epochs": args.num_epochs,
        "final_loss": trainer.state.log_history[-1].get("loss", "N/A"),
        "total_steps": trainer.state.global_step
    }
    
    with open(f"{args.output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("Training complete!")
    
    # Test the model
    print("\nTesting the model with a sample prompt:")
    test_prompt = "Human: What is machine learning?\n\nAssistant:"
    
    model.eval()
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")


if __name__ == "__main__":
    main()