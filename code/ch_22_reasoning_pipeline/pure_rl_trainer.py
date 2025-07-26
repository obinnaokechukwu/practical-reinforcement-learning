"""
Pure RL training for LLMs without supervised fine-tuning.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import wandb


@dataclass
class PureRLConfig:
    """Configuration for pure RL training."""
    model_name: str
    learning_rate: float = 1e-5
    kl_coef: float = 0.02
    clip_epsilon: float = 0.2
    value_clip: float = 0.2
    
    # Generation settings
    max_length: int = 2048
    temperature_start: float = 1.2
    temperature_end: float = 0.3
    temperature_decay: float = 0.995
    
    # Training settings
    batch_size: int = 4
    mini_batch_size: int = 1
    ppo_epochs: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Exploration settings
    entropy_coef_start: float = 0.1
    entropy_coef_end: float = 0.001
    entropy_decay: float = 0.99
    
    # Device settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = False
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    use_wandb: bool = False


class PureRLTrainer:
    """Pure RL trainer for LLMs without SFT."""
    
    def __init__(self, config: PureRLConfig, reward_fn):
        self.config = config
        self.reward_fn = reward_fn
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Policy and value models
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        ).to(config.device)
        
        # Value head
        hidden_size = self.model.config.hidden_size
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        ).to(config.device)
        
        # Reference model for KL penalty
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.fp16 else torch.float32
        ).to(config.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': config.learning_rate},
            {'params': self.value_head.parameters(), 'lr': config.learning_rate * 2}
        ], eps=1e-5)
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=config.learning_rate * 0.1
        )
        
        # Training state
        self.global_step = 0
        self.current_temperature = config.temperature_start
        self.current_entropy_coef = config.entropy_coef_start
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(project="pure-rl-training", config=dataclasses.asdict(config))
    
    def train(self, problems: List[str], num_iterations: int = 1000):
        """Main training loop for pure RL."""
        
        print("Starting Pure RL Training")
        print("=" * 50)
        print(f"Model: {self.config.model_name}")
        print(f"Problems: {len(problems)}")
        print(f"Iterations: {num_iterations}")
        print()
        
        for iteration in range(num_iterations):
            # Collect trajectories
            trajectories = self.collect_trajectories(problems)
            
            # Compute advantages
            self.compute_advantages(trajectories)
            
            # PPO update
            stats = self.ppo_update(trajectories)
            
            # Update hyperparameters
            self.update_hyperparameters()
            
            # Logging
            if iteration % self.config.log_interval == 0:
                self.log_iteration(iteration, stats)
            
            # Checkpoint
            if iteration % self.config.save_interval == 0:
                self.save_checkpoint(f"pure_rl_checkpoint_{iteration}")
            
            self.global_step += 1
    
    def collect_trajectories(self, problems: List[str]) -> List[Dict]:
        """Collect trajectories with current policy."""
        trajectories = []
        
        # Sample batch of problems
        if len(problems) > self.config.batch_size:
            batch_problems = np.random.choice(
                problems, self.config.batch_size, replace=False
            )
        else:
            batch_problems = problems
        
        for problem in batch_problems:
            # Generate response with value estimates
            response_data = self.generate_with_values(problem)
            
            # Skip if generation failed
            if not response_data['tokens']:
                continue
            
            # Compute reward
            reward = self.reward_fn(problem, response_data['response'])
            
            trajectory = {
                'problem': problem,
                'response': response_data['response'],
                'tokens': response_data['tokens'],
                'log_probs': response_data['log_probs'],
                'values': response_data['values'],
                'hidden_states': response_data['hidden_states'],
                'reward': reward,
                'advantages': None,
                'returns': None
            }
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def generate_with_values(self, prompt: str) -> Dict:
        """Generate response while collecting values and log probs."""
        # Encode prompt
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True,
            max_length=self.config.max_length // 2
        ).to(self.config.device)
        
        generated_tokens = []
        log_probs = []
        values = []
        hidden_states_list = []
        
        current_ids = inputs['input_ids']
        past_key_values = None
        
        # Generation loop
        for step in range(self.config.max_length - inputs['input_ids'].shape[1]):
            with torch.no_grad():
                # Forward pass
                outputs = self.model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                hidden = outputs.hidden_states[-1][:, -1, :]
                
                # Get value estimate
                value = self.value_head(hidden)
                values.append(value.squeeze().item())
                hidden_states_list.append(hidden.cpu())
                
                # Sample action (token)
                probs = F.softmax(logits / self.current_temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Store log prob
                log_prob = torch.log(probs[0, next_token.item()])
                log_probs.append(log_prob.item())
                
                # Update sequence
                generated_tokens.append(next_token.item())
                current_ids = next_token
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode response
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'response': response,
            'tokens': generated_tokens,
            'log_probs': log_probs,
            'values': values,
            'hidden_states': hidden_states_list
        }
    
    def compute_advantages(self, trajectories: List[Dict]):
        """Compute advantages using GAE."""
        gamma = 0.99
        lam = 0.95
        
        for traj in trajectories:
            # Create reward signal
            rewards = [0] * (len(traj['values']) - 1) + [traj['reward']]
            values = traj['values']
            
            # Ensure we have values for all positions
            if len(values) != len(rewards):
                # Pad values if necessary
                values = values[:len(rewards)]
                rewards = rewards[:len(values)]
            
            # Compute returns and advantages
            returns = []
            advantages = []
            
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value - values[t]
                gae = delta + gamma * lam * gae
                
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            traj['advantages'] = advantages
            traj['returns'] = returns
    
    def ppo_update(self, trajectories: List[Dict]) -> Dict:
        """Perform PPO update on collected trajectories."""
        stats = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'kl_div': 0,
            'clip_fraction': 0,
            'total_loss': 0
        }
        
        if not trajectories:
            return stats
        
        # Prepare batch data
        batch_data = self._prepare_batch_data(trajectories)
        
        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Shuffle data
            indices = np.random.permutation(len(batch_data['tokens']))
            
            # Mini-batch updates
            for mb_start in range(0, len(indices), self.config.mini_batch_size):
                mb_indices = indices[mb_start:mb_start + self.config.mini_batch_size]
                
                # Get mini-batch
                mb_data = {
                    k: [v[i] for i in mb_indices] 
                    for k, v in batch_data.items()
                }
                
                # Compute losses
                losses = self._compute_ppo_losses(mb_data)
                
                # Total loss
                total_loss = (
                    losses['policy_loss'] + 
                    0.5 * losses['value_loss'] - 
                    self.current_entropy_coef * losses['entropy']
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Update stats
                for key, value in losses.items():
                    stats[key] += value.item()
                stats['total_loss'] += total_loss.item()
        
        # Average stats
        num_updates = self.config.ppo_epochs * (len(trajectories) // self.config.mini_batch_size)
        for key in stats:
            stats[key] /= max(num_updates, 1)
        
        return stats
    
    def _prepare_batch_data(self, trajectories: List[Dict]) -> Dict:
        """Prepare batch data from trajectories."""
        batch_data = {
            'prompts': [],
            'tokens': [],
            'old_log_probs': [],
            'advantages': [],
            'returns': [],
            'hidden_states': []
        }
        
        for traj in trajectories:
            num_tokens = len(traj['tokens'])
            
            # Ensure all lists have same length
            tokens = traj['tokens'][:num_tokens]
            old_log_probs = traj['log_probs'][:num_tokens]
            advantages = traj['advantages'][:num_tokens]
            returns = traj['returns'][:num_tokens]
            hidden_states = traj['hidden_states'][:num_tokens]
            
            # Add to batch
            batch_data['prompts'].extend([traj['problem']] * num_tokens)
            batch_data['tokens'].extend(tokens)
            batch_data['old_log_probs'].extend(old_log_probs)
            batch_data['advantages'].extend(advantages)
            batch_data['returns'].extend(returns)
            batch_data['hidden_states'].extend(hidden_states)
        
        return batch_data
    
    def _compute_ppo_losses(self, mb_data: Dict) -> Dict:
        """Compute PPO losses for mini-batch."""
        # Reconstruct sequences for forward pass
        prompts = mb_data['prompts']
        tokens = mb_data['tokens']
        old_log_probs = torch.tensor(mb_data['old_log_probs']).to(self.config.device)
        advantages = torch.tensor(mb_data['advantages']).to(self.config.device)
        returns = torch.tensor(mb_data['returns']).to(self.config.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass to get new log probs
        new_log_probs = []
        new_values = []
        entropy_list = []
        kl_divs = []
        
        for i in range(len(prompts)):
            # Prepare input
            prompt_ids = self.tokenizer.encode(prompts[i], return_tensors='pt').to(self.config.device)
            token_id = torch.tensor([tokens[i]]).to(self.config.device)
            
            # Get model outputs
            with torch.cuda.amp.autocast(enabled=self.config.fp16):
                outputs = self.model(prompt_ids, output_hidden_states=True)
                logits = outputs.logits[:, -1, :]
                hidden = outputs.hidden_states[-1][:, -1, :]
                
                # Get value
                value = self.value_head(hidden).squeeze()
                new_values.append(value)
                
                # Get log prob
                log_probs = F.log_softmax(logits, dim=-1)
                new_log_prob = log_probs[0, token_id]
                new_log_probs.append(new_log_prob)
                
                # Entropy
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum()
                entropy_list.append(entropy)
                
                # KL divergence with reference model
                with torch.no_grad():
                    ref_outputs = self.ref_model(prompt_ids)
                    ref_logits = ref_outputs.logits[:, -1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                kl = (probs * (log_probs - ref_log_probs)).sum()
                kl_divs.append(kl)
        
        # Stack tensors
        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(new_values)
        entropy = torch.stack(entropy_list).mean()
        kl_div = torch.stack(kl_divs).mean()
        
        # PPO losses
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Policy loss
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
        )
        policy_loss = torch.max(policy_loss1, policy_loss2).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values, returns)
        
        # Clip fraction
        clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
        
        # KL penalty
        policy_loss = policy_loss + self.config.kl_coef * kl_div
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'kl_div': kl_div,
            'clip_fraction': clip_fraction
        }
    
    def update_hyperparameters(self):
        """Update temperature and entropy coefficient."""
        self.current_temperature = max(
            self.config.temperature_end,
            self.current_temperature * self.config.temperature_decay
        )
        
        self.current_entropy_coef = max(
            self.config.entropy_coef_end,
            self.current_entropy_coef * self.config.entropy_decay
        )
        
        self.scheduler.step()
    
    def log_iteration(self, iteration: int, stats: Dict):
        """Log training statistics."""
        print(f"\nIteration {iteration}")
        print(f"Temperature: {self.current_temperature:.3f}")
        print(f"Entropy Coef: {self.current_entropy_coef:.4f}")
        print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
        
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
        
        if self.config.use_wandb:
            wandb.log({
                'iteration': iteration,
                'temperature': self.current_temperature,
                'entropy_coef': self.current_entropy_coef,
                **stats
            })
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'value_head_state': self.value_head.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'temperature': self.current_temperature,
            'entropy_coef': self.current_entropy_coef,
            'config': self.config
        }
        
        torch.save(checkpoint, f"{path}.pt")
        self.model.save_pretrained(f"{path}_hf")
        self.tokenizer.save_pretrained(f"{path}_hf")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(f"{path}.pt")
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.value_head.load_state_dict(checkpoint['value_head_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.global_step = checkpoint['global_step']
        self.current_temperature = checkpoint['temperature']
        self.current_entropy_coef = checkpoint['entropy_coef']