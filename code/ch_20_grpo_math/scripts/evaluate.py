#!/usr/bin/env python3
"""
Evaluate a GRPO-trained model on mathematical reasoning tasks.
"""

import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

# Add the grpo package to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from grpo import (
    MathRewardFunction, FormatReward,
    MathDataset, create_data_loader
)
from grpo.utils import (
    extract_thinking, extract_answer, check_format_compliance,
    analyze_response_quality, compute_group_statistics
)


def load_model_and_tokenizer(model_path: str, device: str = 'cuda'):
    """Load trained model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, 
                     max_length: int = 2048,
                     temperature: float = 0.1,
                     device: str = 'cuda') -> str:
    """Generate a single response for evaluation."""
    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (remove prompt)
    response = tokenizer.decode(
        outputs[0, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response


def evaluate_dataset(model, tokenizer, dataset, 
                    reward_fn, device: str = 'cuda',
                    num_samples: int = None,
                    temperature: float = 0.1) -> dict:
    """Evaluate model on entire dataset."""
    
    if num_samples:
        # Sample subset for faster evaluation
        indices = np.random.choice(
            len(dataset), 
            min(num_samples, len(dataset)), 
            replace=False
        )
        eval_data = [dataset[i] for i in indices]
    else:
        eval_data = [dataset[i] for i in range(len(dataset))]
    
    results = {
        'responses': [],
        'rewards': [],
        'format_compliance': [],
        'response_qualities': []
    }
    
    print(f"Evaluating on {len(eval_data)} samples...")
    
    for item in tqdm(eval_data, desc="Evaluating"):
        prompt = item['prompt']
        ground_truth = item.get('ground_truth', '')
        
        # Generate response
        response = generate_response(
            model, tokenizer, prompt, 
            temperature=temperature, device=device
        )
        
        # Compute reward
        reward = reward_fn(prompt, response)
        
        # Check format compliance
        format_ok = check_format_compliance(response)
        
        # Analyze response quality
        quality = analyze_response_quality(response)
        
        # Store results
        results['responses'].append({
            'prompt': prompt,
            'response': response,
            'ground_truth': ground_truth,
            'reward': reward,
            'format_ok': format_ok,
            'thinking': extract_thinking(response),
            'answer': extract_answer(response)
        })
        results['rewards'].append(reward)
        results['format_compliance'].append(format_ok)
        results['response_qualities'].append(quality)
    
    return results


def compute_evaluation_metrics(results: dict) -> dict:
    """Compute comprehensive evaluation metrics."""
    rewards = results['rewards']
    format_compliance = results['format_compliance']
    qualities = results['response_qualities']
    
    metrics = {
        # Reward statistics
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'median_reward': np.median(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        
        # Format compliance
        'format_compliance_rate': np.mean(format_compliance),
        
        # Response quality
        'avg_response_length': np.mean([q['total_length'] for q in qualities]),
        'avg_thinking_length': np.mean([q['thinking_length'] for q in qualities]),
        'avg_answer_length': np.mean([q['answer_length'] for q in qualities]),
        'avg_thinking_steps': np.mean([q['thinking_steps'] for q in qualities]),
        'equation_usage_rate': np.mean([q['uses_equations'] for q in qualities]),
        'code_usage_rate': np.mean([q['uses_code'] for q in qualities]),
        
        # Success rates
        'high_reward_rate': np.mean([r > 0.8 for r in rewards]),
        'medium_reward_rate': np.mean([r > 0.5 for r in rewards]),
        'low_reward_rate': np.mean([r < 0.2 for r in rewards])
    }
    
    return metrics


def save_detailed_results(results: dict, output_path: str):
    """Save detailed evaluation results."""
    # Prepare data for JSON serialization
    serializable_results = {
        'responses': results['responses'],
        'summary': {
            'total_samples': len(results['rewards']),
            'format_compliance_rate': np.mean(results['format_compliance']),
            'mean_reward': np.mean(results['rewards'])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def save_human_readable_examples(results: dict, output_path: str, 
                                num_examples: int = 10):
    """Save human-readable examples for inspection."""
    responses = results['responses']
    
    # Sort by reward (highest first)
    sorted_responses = sorted(
        responses, 
        key=lambda x: x['reward'], 
        reverse=True
    )
    
    with open(output_path, 'w') as f:
        f.write("GRPO Model Evaluation Examples\n")
        f.write("=" * 80 + "\n\n")
        
        # Top examples
        f.write("TOP PERFORMING EXAMPLES:\n")
        f.write("-" * 40 + "\n\n")
        
        for i, example in enumerate(sorted_responses[:num_examples//2]):
            f.write(f"Example {i+1} (Reward: {example['reward']:.3f})\n")
            f.write("=" * 60 + "\n")
            f.write(f"Prompt: {example['prompt']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Response:\n{example['response']}\n")
            if example['ground_truth']:
                f.write(f"\nGround Truth: {example['ground_truth']}\n")
            f.write("\n" + "=" * 60 + "\n\n")
        
        # Bottom examples  
        f.write("\nLOWEST PERFORMING EXAMPLES:\n")
        f.write("-" * 40 + "\n\n")
        
        for i, example in enumerate(sorted_responses[-num_examples//2:]):
            f.write(f"Example {i+1} (Reward: {example['reward']:.3f})\n")
            f.write("=" * 60 + "\n")
            f.write(f"Prompt: {example['prompt']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Response:\n{example['response']}\n")
            if example['ground_truth']:
                f.write(f"\nGround Truth: {example['ground_truth']}\n")
            f.write("\n" + "=" * 60 + "\n\n")


def create_evaluation_report(metrics: dict, config: dict, 
                           output_path: str):
    """Create a comprehensive evaluation report."""
    with open(output_path, 'w') as f:
        f.write("GRPO Model Evaluation Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {config.get('model_path', 'Unknown')}\n")
        f.write(f"Dataset: {config.get('dataset_path', 'Unknown')}\n")
        f.write(f"Evaluation samples: {config.get('num_samples', 'All')}\n")
        f.write(f"Temperature: {config.get('temperature', 0.1)}\n")
        f.write("\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Reward: {metrics['mean_reward']:.4f}\n")
        f.write(f"Reward Std Dev: {metrics['std_reward']:.4f}\n")
        f.write(f"Median Reward: {metrics['median_reward']:.4f}\n")
        f.write(f"Min Reward: {metrics['min_reward']:.4f}\n")
        f.write(f"Max Reward: {metrics['max_reward']:.4f}\n")
        f.write("\n")
        
        f.write("SUCCESS RATES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"High Reward (>0.8): {metrics['high_reward_rate']:.1%}\n")
        f.write(f"Medium Reward (>0.5): {metrics['medium_reward_rate']:.1%}\n")
        f.write(f"Low Reward (<0.2): {metrics['low_reward_rate']:.1%}\n")
        f.write("\n")
        
        f.write("FORMAT AND QUALITY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Format Compliance: {metrics['format_compliance_rate']:.1%}\n")
        f.write(f"Avg Response Length: {metrics['avg_response_length']:.0f} chars\n")
        f.write(f"Avg Thinking Length: {metrics['avg_thinking_length']:.0f} chars\n")
        f.write(f"Avg Answer Length: {metrics['avg_answer_length']:.0f} chars\n")
        f.write(f"Avg Thinking Steps: {metrics['avg_thinking_steps']:.1f}\n")
        f.write(f"Equation Usage: {metrics['equation_usage_rate']:.1%}\n")
        f.write(f"Code Usage: {metrics['code_usage_rate']:.1%}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GRPO-trained model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to evaluation dataset (JSON)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save evaluation results')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Generation temperature')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for evaluation')
    parser.add_argument('--batch-eval', action='store_true',
                       help='Use batch evaluation for speed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # Set up reward function
    format_reward = FormatReward(weight=0.1)
    reward_fn = MathRewardFunction(
        format_reward=format_reward,
        correct_weight=1.0,
        partial_credit=True
    )
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = MathDataset(
        data_path=args.dataset_path,
        max_prompt_length=512
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Run evaluation
    config = {
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
        'num_samples': args.num_samples,
        'temperature': args.temperature,
        'device': args.device
    }
    
    results = evaluate_dataset(
        model, tokenizer, dataset, reward_fn,
        device=args.device,
        num_samples=args.num_samples,
        temperature=args.temperature
    )
    
    # Compute metrics
    print("Computing evaluation metrics...")
    metrics = compute_evaluation_metrics(results)
    
    # Print summary
    print("\nEVALUATION SUMMARY:")
    print("=" * 50)
    print(f"Samples evaluated: {len(results['rewards'])}")
    print(f"Mean reward: {metrics['mean_reward']:.4f}")
    print(f"Format compliance: {metrics['format_compliance_rate']:.1%}")
    print(f"High reward rate (>0.8): {metrics['high_reward_rate']:.1%}")
    print(f"Medium reward rate (>0.5): {metrics['medium_reward_rate']:.1%}")
    
    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    
    # Detailed results (JSON)
    detailed_path = os.path.join(args.output_dir, 'detailed_results.json')
    save_detailed_results(results, detailed_path)
    print(f"Detailed results saved to {detailed_path}")
    
    # Metrics (JSON)
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Human-readable examples
    examples_path = os.path.join(args.output_dir, 'examples.txt')
    save_human_readable_examples(results, examples_path)
    print(f"Examples saved to {examples_path}")
    
    # Evaluation report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    create_evaluation_report(metrics, config, report_path)
    print(f"Evaluation report saved to {report_path}")
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()