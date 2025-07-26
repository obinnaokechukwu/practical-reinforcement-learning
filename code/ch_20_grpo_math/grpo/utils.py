"""
Utility functions for GRPO implementation.
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from datetime import datetime
import os


def extract_thinking(response: str) -> Optional[str]:
    """Extract thinking content from response."""
    match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_answer(response: str) -> Optional[str]:
    """Extract answer content from response."""
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def check_format_compliance(response: str) -> bool:
    """Check if response follows the expected format."""
    has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))
    
    if not (has_think and has_answer):
        return False
    
    # Check ordering
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    if think_match and answer_match:
        return think_match.end() <= answer_match.start()
    
    return False


def compute_group_statistics(rewards: List[float]) -> Dict[str, float]:
    """Compute statistics for a group of rewards."""
    rewards_array = np.array(rewards)
    
    return {
        'mean': float(rewards_array.mean()),
        'std': float(rewards_array.std()),
        'min': float(rewards_array.min()),
        'max': float(rewards_array.max()),
        'median': float(np.median(rewards_array))
    }


def prepare_prompts_for_batch(prompts: List[str], 
                            tokenizer: AutoTokenizer,
                            max_length: int = 512) -> Dict[str, torch.Tensor]:
    """Prepare prompts for batch processing."""
    # Tokenize all prompts
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return encoded


def compute_token_level_kl(logits1: torch.Tensor, 
                          logits2: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence at token level."""
    # Convert to log probabilities
    log_probs1 = torch.log_softmax(logits1, dim=-1)
    log_probs2 = torch.log_softmax(logits2, dim=-1)
    
    # Compute KL divergence
    probs1 = torch.exp(log_probs1)
    kl = (probs1 * (log_probs1 - log_probs2)).sum(dim=-1)
    
    # Mask and average
    masked_kl = kl * mask
    avg_kl = masked_kl.sum() / mask.sum()
    
    return avg_kl


def analyze_response_quality(response: str) -> Dict[str, any]:
    """Analyze various aspects of response quality."""
    analysis = {
        'total_length': len(response),
        'has_format': check_format_compliance(response),
        'thinking_length': 0,
        'answer_length': 0,
        'thinking_steps': 0,
        'uses_equations': False,
        'uses_code': False
    }
    
    # Extract sections
    thinking = extract_thinking(response)
    answer = extract_answer(response)
    
    if thinking:
        analysis['thinking_length'] = len(thinking)
        # Count reasoning steps (sentences or newlines)
        analysis['thinking_steps'] = max(
            thinking.count('.'),
            thinking.count('\n')
        )
        # Check for mathematical notation
        analysis['uses_equations'] = bool(
            re.search(r'[\$=\+\-\*\/\^]', thinking)
        )
    
    if answer:
        analysis['answer_length'] = len(answer)
        # Check for code blocks
        analysis['uses_code'] = bool(
            re.search(r'```|def |class |import ', answer)
        )
    
    return analysis


def save_training_checkpoint(trainer, epoch: int, step: int, 
                           metrics: Dict, output_dir: str):
    """Save training checkpoint with all necessary information."""
    checkpoint_dir = os.path.join(
        output_dir, 
        f"checkpoint-epoch{epoch}-step{step}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model and tokenizer
    trainer.policy.save_pretrained(checkpoint_dir)
    trainer.tokenizer.save_pretrained(checkpoint_dir)
    
    # Save training state
    state = {
        'epoch': epoch,
        'step': step,
        'metrics': metrics,
        'optimizer_state': trainer.optimizer.state_dict(),
        'config': trainer.config.__dict__ if hasattr(trainer, 'config') else {},
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(state, os.path.join(checkpoint_dir, 'training_state.pt'))
    
    # Save metrics history
    with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def plot_training_metrics(metrics_history: List[Dict], output_path: str):
    """Plot training metrics over time."""
    if not metrics_history:
        return
    
    # Extract metric names
    metric_names = list(metrics_history[0].keys())
    metric_names = [m for m in metric_names if m != 'step']
    
    # Create subplots
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(
        (n_metrics + 1) // 2, 2, 
        figsize=(12, 4 * ((n_metrics + 1) // 2))
    )
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        if i < len(axes):
            steps = [m.get('step', i) for i, m in enumerate(metrics_history)]
            values = [m.get(metric_name, 0) for m in metrics_history]
            
            axes[i].plot(steps, values)
            axes[i].set_title(metric_name)
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel('Value')
            axes[i].grid(True)
    
    # Remove extra subplots
    for i in range(len(metric_names), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_reasoning_trace(prompt: str, thinking: str, 
                          answer: str) -> str:
    """Create a formatted reasoning trace for logging."""
    trace = f"PROMPT: {prompt}\n"
    trace += f"{'-' * 80}\n"
    trace += f"THINKING:\n{thinking}\n"
    trace += f"{'-' * 80}\n"
    trace += f"ANSWER: {answer}\n"
    trace += f"{'=' * 80}\n"
    return trace


def batch_compute_advantages(rewards_groups: List[List[float]]) -> List[List[float]]:
    """Compute advantages for multiple groups efficiently."""
    all_advantages = []
    
    for rewards in rewards_groups:
        rewards_array = np.array(rewards)
        mean = rewards_array.mean()
        std = rewards_array.std() + 1e-8
        
        advantages = (rewards_array - mean) / std
        all_advantages.append(advantages.tolist())
    
    return all_advantages


def filter_responses_by_advantage(responses: List[Dict], 
                                 threshold: float = 0.1) -> List[Dict]:
    """Filter responses based on advantage magnitude."""
    filtered = []
    
    for response in responses:
        if abs(response.get('advantage', 0)) >= threshold:
            filtered.append(response)
    
    return filtered


def compute_effective_batch_size(group_size: int, 
                               num_prompts: int,
                               advantage_threshold: float = 0.1,
                               expected_advantage_rate: float = 0.6) -> int:
    """Estimate effective batch size after advantage filtering."""
    total_responses = group_size * num_prompts
    expected_valid = int(total_responses * expected_advantage_rate)
    return max(1, expected_valid)


def log_generation_examples(prompts: List[str], 
                          responses_groups: List[List[str]], 
                          rewards_groups: List[List[float]],
                          output_file: str,
                          num_examples: int = 3):
    """Log example generations for inspection."""
    with open(output_file, 'w') as f:
        for i, (prompt, responses, rewards) in enumerate(
            zip(prompts[:num_examples], 
                responses_groups[:num_examples], 
                rewards_groups[:num_examples])
        ):
            f.write(f"Example {i+1}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"{'-' * 80}\n")
            
            # Sort by reward
            sorted_pairs = sorted(
                zip(responses, rewards), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for j, (response, reward) in enumerate(sorted_pairs):
                f.write(f"\nResponse {j+1} (Reward: {reward:.3f}):\n")
                f.write(response)
                f.write(f"\n{'-' * 40}\n")
            
            f.write(f"\n{'=' * 80}\n\n")


def estimate_memory_usage(model_size: int, 
                         batch_size: int,
                         group_size: int,
                         sequence_length: int) -> Dict[str, float]:
    """Estimate memory usage for GRPO training."""
    # Model parameters (policy + reference)
    model_memory_gb = (model_size * 2 * 2) / 1e9  # 2 bytes per param, 2 models
    
    # Gradients
    gradient_memory_gb = (model_size * 2) / 1e9
    
    # Optimizer states (AdamW: 2 moments per parameter)
    optimizer_memory_gb = (model_size * 2 * 2) / 1e9
    
    # Activations (rough estimate)
    # batch_size * group_size * sequence_length * hidden_size * layers
    hidden_size = int(model_size ** 0.5)  # Rough approximation
    layers = 32  # Typical for 7B model
    activation_memory_gb = (
        batch_size * group_size * sequence_length * 
        hidden_size * layers * 2
    ) / 1e9
    
    total_memory_gb = (
        model_memory_gb + gradient_memory_gb + 
        optimizer_memory_gb + activation_memory_gb
    )
    
    return {
        'model_memory_gb': model_memory_gb,
        'gradient_memory_gb': gradient_memory_gb,
        'optimizer_memory_gb': optimizer_memory_gb,
        'activation_memory_gb': activation_memory_gb,
        'total_memory_gb': total_memory_gb
    }


class ResponseCache:
    """Cache for storing generated responses to avoid regeneration."""
    
    def __init__(self, cache_size: int = 10000):
        self.cache_size = cache_size
        self.cache = {}
        self.access_counts = {}
    
    def get_cache_key(self, prompt: str, temperature: float, 
                      top_k: int, top_p: float) -> str:
        """Generate cache key for prompt and generation parameters."""
        import hashlib
        key_str = f"{prompt}|{temperature}|{top_k}|{top_p}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, temperature: float, 
            top_k: int, top_p: float) -> Optional[List[str]]:
        """Retrieve cached responses if available."""
        key = self.get_cache_key(prompt, temperature, top_k, top_p)
        
        if key in self.cache:
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            return self.cache[key]
        
        return None
    
    def put(self, prompt: str, temperature: float, 
            top_k: int, top_p: float, responses: List[str]):
        """Store responses in cache."""
        if len(self.cache) >= self.cache_size:
            # Evict least accessed
            min_key = min(
                self.access_counts.keys(), 
                key=lambda k: self.access_counts.get(k, 0)
            )
            del self.cache[min_key]
            del self.access_counts[min_key]
        
        key = self.get_cache_key(prompt, temperature, top_k, top_p)
        self.cache[key] = responses
        self.access_counts[key] = 1


def create_grpo_summary_report(
    training_history: List[Dict],
    eval_results: Dict,
    config: Dict,
    output_path: str
):
    """Create a comprehensive training summary report."""
    report = []
    report.append("GRPO Training Summary Report")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("")
    
    # Configuration summary
    report.append("Configuration:")
    report.append("-" * 40)
    for key, value in config.items():
        report.append(f"  {key}: {value}")
    report.append("")
    
    # Training progress
    report.append("Training Progress:")
    report.append("-" * 40)
    
    if training_history:
        # Final metrics
        final_metrics = training_history[-1]
        report.append(f"Total steps: {final_metrics.get('step', len(training_history))}")
        report.append(f"Final loss: {final_metrics.get('loss', 'N/A'):.4f}")
        report.append(f"Final mean reward: {final_metrics.get('mean_reward', 'N/A'):.3f}")
        report.append(f"Final KL divergence: {final_metrics.get('kl_div', 'N/A'):.4f}")
        report.append("")
        
        # Best metrics
        best_reward_idx = max(
            range(len(training_history)), 
            key=lambda i: training_history[i].get('mean_reward', -float('inf'))
        )
        best_metrics = training_history[best_reward_idx]
        report.append(f"Best reward achieved at step {best_metrics.get('step', best_reward_idx)}:")
        report.append(f"  Mean reward: {best_metrics.get('mean_reward', 'N/A'):.3f}")
        report.append("")
    
    # Evaluation results
    if eval_results:
        report.append("Evaluation Results:")
        report.append("-" * 40)
        for key, value in eval_results.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.3f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))