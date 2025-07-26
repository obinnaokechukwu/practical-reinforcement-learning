"""
Integrated LLM-Guided RL Agent

This module combines all components (reward modeling, exploration guidance,
and hierarchical policies) into a complete LLM-guided RL system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import gym
import matplotlib.pyplot as plt
from collections import deque
import json
import os

# Import our components
from llm_reward_model import RobustLanguageRewardModel
from exploration_guide import LLMExplorationGuide, CuriosityDrivenAgent
from hierarchical_policy import LLMPolicyGenerator, HierarchicalPolicy


@dataclass
class LLMGuidedConfig:
    """Configuration for LLM-guided RL agent."""
    env_name: str = "CartPole-v1"
    llm_model: str = "gpt2"
    use_reward_model: bool = True
    use_exploration_guide: bool = True
    use_skill_decomposition: bool = True
    intrinsic_reward_scale: float = 0.1
    extrinsic_reward_scale: float = 0.5
    language_reward_scale: float = 0.5
    skill_switching_threshold: int = 20
    exploration_fraction: float = 0.2
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    update_frequency: int = 4


class ExperienceBuffer:
    """Experience replay buffer for training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Dict[str, Any]):
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


class IntegratedLLMAgent:
    """
    Complete LLM-guided RL agent integrating all components.
    """
    
    def __init__(self, config: LLMGuidedConfig):
        self.config = config
        self.env = gym.make(config.env_name)
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Initialize components based on config
        self._initialize_components()
        
        # Training state
        self.experience_buffer = ExperienceBuffer()
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = -float('inf')
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'intrinsic_rewards': [],
            'language_rewards': [],
            'exploration_goals_achieved': [],
            'skills_used': [],
            'training_losses': []
        }
    
    def _initialize_components(self):
        """Initialize all components based on configuration."""
        # Reward model
        if self.config.use_reward_model:
            print("Initializing language reward model...")
            self.reward_model = RobustLanguageRewardModel(
                model_name=self.config.llm_model,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=128
            )
        else:
            self.reward_model = None
        
        # Exploration guide
        if self.config.use_exploration_guide:
            print("Initializing exploration guide...")
            self.exploration_guide = LLMExplorationGuide(
                env_name=self.config.env_name,
                model_name=self.config.llm_model,
                state_dim=self.state_dim,
                action_dim=self.action_dim
            )
        else:
            self.exploration_guide = None
        
        # Policy generator and hierarchical policy
        if self.config.use_skill_decomposition:
            print("Initializing hierarchical policy generator...")
            self.policy_generator = LLMPolicyGenerator(
                base_model=self.config.llm_model,
                env_description=self._get_env_description(),
                state_dim=self.state_dim,
                action_dim=self.action_dim
            )
            self.hierarchical_policy = None  # Created per task
        else:
            self.policy_generator = None
            self.hierarchical_policy = None
        
        # Fallback simple policy
        self.simple_policy = self._create_simple_policy()
        self.current_policy = self.simple_policy
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.simple_policy.parameters(), 
            lr=self.config.learning_rate
        )
    
    def _get_env_description(self) -> str:
        """Generate environment description."""
        descriptions = {
            "CartPole-v1": "Cart-pole balancing: Move cart left/right to keep pole upright",
            "MountainCar-v0": "Mountain car: Build momentum to reach the goal at the top",
            "LunarLander-v2": "Lunar lander: Control thrusters to land safely"
        }
        return descriptions.get(self.config.env_name, f"Environment: {self.config.env_name}")
    
    def _create_simple_policy(self) -> nn.Module:
        """Create a simple neural network policy."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, self.action_dim)
        )
    
    def train(
        self, 
        task_description: str, 
        num_episodes: int = 1000,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the agent on a task described in natural language.
        """
        print(f"\n{'='*60}")
        print(f"Training on task: {task_description}")
        print(f"{'='*60}\n")
        
        # Step 1: Task analysis and setup
        self._setup_for_task(task_description)
        
        # Step 2: Training loop
        for episode in range(num_episodes):
            self.episode_count = episode
            episode_metrics = self._run_episode(task_description)
            
            # Update metrics
            self._update_metrics(episode_metrics)
            
            # Logging
            if episode % 100 == 0:
                self._log_progress(episode, num_episodes)
            
            # Save checkpoint
            if save_path and episode % 500 == 0:
                self._save_checkpoint(save_path, episode)
            
            # Curriculum update
            if episode % 200 == 0 and self.exploration_guide:
                self._update_exploration_curriculum(task_description)
        
        print(f"\nTraining completed!")
        return self.metrics
    
    def _setup_for_task(self, task_description: str):
        """Setup agent for specific task."""
        # Generate hierarchical policy if enabled
        if self.config.use_skill_decomposition and self.policy_generator:
            print("Decomposing task into skills...")
            self.hierarchical_policy = self.policy_generator.create_hierarchical_policy(
                task_description
            )
            self.current_policy = self.hierarchical_policy
            
            # Update optimizer for hierarchical policy
            trainable_params = []
            for module in self.hierarchical_policy.modules():
                trainable_params.extend(module.parameters())
            self.policy_optimizer = optim.Adam(
                trainable_params,
                lr=self.config.learning_rate
            )
        
        # Generate exploration curriculum
        if self.exploration_guide:
            print("Generating exploration curriculum...")
            curriculum = self.exploration_guide.generate_exploration_curriculum(
                task_description
            )
            print("Exploration objectives:")
            for i, obj in enumerate(curriculum, 1):
                print(f"  {i}. {obj}")
    
    def _run_episode(self, task_description: str) -> Dict[str, Any]:
        """Run a single episode."""
        state = self.env.reset()
        episode_reward = 0
        episode_intrinsic = 0
        episode_language = 0
        trajectory = []
        skills_used = set()
        
        done = False
        step = 0
        
        # Reset hierarchical policy if used
        if isinstance(self.current_policy, HierarchicalPolicy):
            self.current_policy.reset()
        
        while not done and step < 500:
            # Select action
            action, action_info = self._select_action(state, task_description)
            
            # Track skills
            if 'current_skill' in action_info:
                skills_used.add(action_info['current_skill'])
            
            # Execute action
            next_state, env_reward, done, info = self.env.step(action)
            
            # Compute rewards
            rewards = self._compute_rewards(
                state, action, next_state, env_reward, 
                task_description, done
            )
            
            # Store experience
            experience = {
                'state': state,
                'action': action,
                'next_state': next_state,
                'done': done,
                **rewards,
                **action_info
            }
            
            self.experience_buffer.push(experience)
            trajectory.append(experience)
            
            # Update state
            state = next_state
            episode_reward += rewards['total_reward']
            episode_intrinsic += rewards.get('intrinsic_reward', 0)
            episode_language += rewards.get('language_reward', 0)
            step += 1
            self.total_steps += 1
            
            # Training update
            if self.total_steps % self.config.update_frequency == 0:
                self._training_step()
        
        # Episode-level updates
        if self.exploration_guide:
            # Update curiosity model
            curiosity_experiences = [
                (e['state'], e['action'], e['next_state'], e['total_reward'])
                for e in trajectory
            ]
            self.exploration_guide.update_curiosity_model(curiosity_experiences)
        
        return {
            'total_reward': episode_reward,
            'intrinsic_reward': episode_intrinsic,
            'language_reward': episode_language,
            'trajectory_length': len(trajectory),
            'skills_used': list(skills_used),
            'goals_achieved': len(self.exploration_guide.achieved_goals) 
                              if self.exploration_guide else 0
        }
    
    def _select_action(
        self, 
        state: np.ndarray, 
        task_context: str
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action using current policy with exploration."""
        state_tensor = torch.FloatTensor(state)
        
        # Exploration vs exploitation
        if np.random.random() < self.config.exploration_fraction:
            # Exploration
            if self.exploration_guide:
                # Get LLM suggestion
                suggested = self.exploration_guide.get_action_suggestion(
                    state, list(range(self.action_dim))
                )
                if suggested is not None:
                    return suggested, {'exploration': True, 'llm_suggested': True}
            
            # Random exploration
            action = self.env.action_space.sample()
            return action, {'exploration': True, 'llm_suggested': False}
        
        # Exploitation
        if isinstance(self.current_policy, HierarchicalPolicy):
            action, info = self.current_policy(state_tensor)
            info['exploration'] = False
            return action, info
        else:
            # Simple policy
            with torch.no_grad():
                logits = self.current_policy(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
            
            return action, {'exploration': False}
    
    def _compute_rewards(
        self, 
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        env_reward: float,
        task_description: str,
        done: bool
    ) -> Dict[str, float]:
        """Compute all reward components."""
        rewards = {'env_reward': env_reward}
        
        # Language-based reward
        if self.reward_model:
            with torch.no_grad():
                lang_reward_info = self.reward_model.forward_with_uncertainty(
                    task_description,
                    torch.FloatTensor(state).unsqueeze(0),
                    torch.tensor([action])
                )
                language_reward = lang_reward_info['reward'].item()
                rewards['language_reward'] = language_reward
        else:
            rewards['language_reward'] = 0
        
        # Intrinsic/curiosity reward
        if self.exploration_guide and not done:
            intrinsic = self.exploration_guide.compute_curiosity_bonus(
                state, action, next_state, task_description
            )
            rewards['intrinsic_reward'] = intrinsic
        else:
            rewards['intrinsic_reward'] = 0
        
        # Combine rewards
        total_reward = (
            self.config.extrinsic_reward_scale * env_reward +
            self.config.language_reward_scale * rewards['language_reward'] +
            self.config.intrinsic_reward_scale * rewards['intrinsic_reward']
        )
        
        rewards['total_reward'] = total_reward
        
        return rewards
    
    def _training_step(self):
        """Perform one training step."""
        if len(self.experience_buffer) < self.config.batch_size:
            return
        
        # Sample batch
        batch = self.experience_buffer.sample(self.config.batch_size)
        
        # Prepare tensors
        states = torch.FloatTensor([e['state'] for e in batch])
        actions = torch.LongTensor([e['action'] for e in batch])
        rewards = torch.FloatTensor([e['total_reward'] for e in batch])
        next_states = torch.FloatTensor([e['next_state'] for e in batch])
        dones = torch.FloatTensor([e['done'] for e in batch])
        
        # Simple policy gradient update (REINFORCE-style)
        # In practice, would use more sophisticated algorithms
        
        # Get action probabilities
        logits = self.simple_policy(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute returns (simple version)
        returns = rewards  # Could compute discounted returns
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy loss
        policy_loss = -(selected_log_probs * returns).mean()
        
        # Entropy bonus for exploration
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss - 0.01 * entropy
        
        # Update
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.simple_policy.parameters(), 1.0)
        self.policy_optimizer.step()
        
        self.metrics['training_losses'].append(loss.item())
    
    def _update_metrics(self, episode_metrics: Dict[str, Any]):
        """Update tracking metrics."""
        self.metrics['episode_rewards'].append(episode_metrics['total_reward'])
        self.metrics['intrinsic_rewards'].append(episode_metrics['intrinsic_reward'])
        self.metrics['language_rewards'].append(episode_metrics['language_reward'])
        self.metrics['exploration_goals_achieved'].append(episode_metrics['goals_achieved'])
        self.metrics['skills_used'].append(episode_metrics['skills_used'])
        
        # Update best reward
        if episode_metrics['total_reward'] > self.best_reward:
            self.best_reward = episode_metrics['total_reward']
    
    def _log_progress(self, episode: int, total_episodes: int):
        """Log training progress."""
        recent_rewards = self.metrics['episode_rewards'][-100:]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0
        
        recent_intrinsic = self.metrics['intrinsic_rewards'][-100:]
        avg_intrinsic = np.mean(recent_intrinsic) if recent_intrinsic else 0
        
        print(f"\nEpisode {episode}/{total_episodes}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Best Reward: {self.best_reward:.2f}")
        print(f"  Average Intrinsic: {avg_intrinsic:.3f}")
        
        if self.metrics['exploration_goals_achieved']:
            total_goals = sum(self.metrics['exploration_goals_achieved'])
            print(f"  Exploration Goals Achieved: {total_goals}")
        
        if isinstance(self.current_policy, HierarchicalPolicy):
            skill_metrics = self.current_policy.get_metrics()
            print("  Skill Usage:")
            for metric, value in skill_metrics.items():
                print(f"    {metric}: {value:.2f}")
    
    def _update_exploration_curriculum(self, task_description: str):
        """Update exploration objectives based on progress."""
        if not self.exploration_guide:
            return
        
        # Generate new exploration goal based on current state
        recent_states = list(self.experience_buffer.buffer)[-100:]
        if recent_states:
            avg_state = np.mean([e['state'] for e in recent_states], axis=0)
            history = [{'reward': e['total_reward']} for e in recent_states[-10:]]
            
            new_goal = self.exploration_guide.suggest_exploration_goal(
                avg_state, task_description, history
            )
            
            print(f"\nNew exploration goal: {new_goal.description}")
    
    def _save_checkpoint(self, save_path: str, episode: int):
        """Save training checkpoint."""
        os.makedirs(save_path, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'config': self.config,
            'metrics': self.metrics,
            'best_reward': self.best_reward
        }
        
        # Save models
        if self.simple_policy:
            torch.save(
                self.simple_policy.state_dict(),
                os.path.join(save_path, f'policy_ep{episode}.pt')
            )
        
        # Save metrics
        with open(os.path.join(save_path, f'metrics_ep{episode}.json'), 'w') as f:
            # Convert non-serializable items
            serializable_metrics = {
                k: v for k, v in self.metrics.items()
                if k != 'skills_used'  # Skip complex structures
            }
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"  Checkpoint saved to {save_path}")
    
    def evaluate(
        self, 
        task_description: str, 
        num_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        """
        print(f"\nEvaluating on: {task_description}")
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            
            if isinstance(self.current_policy, HierarchicalPolicy):
                self.current_policy.reset()
            
            done = False
            while not done and steps < 500:
                # Deterministic action selection
                action, _ = self._select_action(state, task_description)
                
                if render:
                    self.env.render()
                
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(steps)
        
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'max_reward': np.max(eval_rewards),
            'min_reward': np.min(eval_rewards)
        }
        
        print(f"Evaluation Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Episode Length: {results['mean_length']:.1f}")
        print(f"  Best/Worst: {results['max_reward']:.1f}/{results['min_reward']:.1f}")
        
        return results
    
    def visualize_training(self, save_path: Optional[str] = None):
        """Visualize training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax = axes[0, 0]
        rewards = self.metrics['episode_rewards']
        ax.plot(rewards, alpha=0.3, label='Raw')
        if len(rewards) > 20:
            smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
            ax.plot(range(19, len(rewards)), smoothed, label='Smoothed', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Learning Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Reward components
        ax = axes[0, 1]
        if self.metrics['intrinsic_rewards']:
            ax.plot(self.metrics['intrinsic_rewards'], alpha=0.7, label='Intrinsic')
        if self.metrics['language_rewards']:
            ax.plot(self.metrics['language_rewards'], alpha=0.7, label='Language')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Exploration progress
        ax = axes[1, 0]
        if self.metrics['exploration_goals_achieved']:
            cumulative_goals = np.cumsum(self.metrics['exploration_goals_achieved'])
            ax.plot(cumulative_goals, linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Cumulative Goals Achieved')
            ax.set_title('Exploration Progress')
            ax.grid(True, alpha=0.3)
        
        # Training loss
        ax = axes[1, 1]
        if self.metrics['training_losses']:
            losses = self.metrics['training_losses']
            ax.plot(losses, alpha=0.5)
            if len(losses) > 100:
                smoothed = np.convolve(losses, np.ones(100)/100, mode='valid')
                ax.plot(range(99, len(losses)), smoothed, linewidth=2)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training visualization saved to {save_path}")
        else:
            plt.show()


# Demo and example usage
def main():
    """
    Demonstrate the integrated LLM-guided RL system.
    """
    # Configure agent
    config = LLMGuidedConfig(
        env_name="CartPole-v1",
        llm_model="gpt2",
        use_reward_model=True,
        use_exploration_guide=True,
        use_skill_decomposition=True,
        intrinsic_reward_scale=0.1,
        exploration_fraction=0.3
    )
    
    # Create agent
    print("Initializing Integrated LLM-Guided Agent...")
    agent = IntegratedLLMAgent(config)
    
    # Define task in natural language
    task = """Balance the pole for as long as possible by moving the cart.
    Try to keep movements smooth and avoid jerky motions.
    The pole should remain as upright as possible."""
    
    # Train
    print("\nStarting training...")
    metrics = agent.train(
        task_description=task,
        num_episodes=500,
        save_path="checkpoints/llm_guided"
    )
    
    # Evaluate
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(task, num_episodes=20)
    
    # Visualize results
    agent.visualize_training("training_progress.png")
    
    # Print final statistics
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total Episodes: {len(metrics['episode_rewards'])}")
    print(f"Best Reward: {agent.best_reward:.2f}")
    print(f"Final Average (last 100): {np.mean(metrics['episode_rewards'][-100:]):.2f}")
    
    if agent.exploration_guide:
        print(f"\nExploration Statistics:")
        stats = agent.exploration_guide.state_memory.get_statistics()
        print(f"  Unique States Visited: {stats['unique_states']}")
        print(f"  Total Goals Achieved: {sum(metrics['exploration_goals_achieved'])}")
    
    if isinstance(agent.current_policy, HierarchicalPolicy):
        print(f"\nSkill Usage Statistics:")
        skill_stats = agent.policy_generator.skill_library.get_statistics()
        print(f"  Total Skills: {skill_stats['total_skills']}")
        print(f"  Average Success Rate: {skill_stats['avg_success_rate']:.2%}")
        if skill_stats['most_used_skill']:
            print(f"  Most Used Skill: {skill_stats['most_used_skill']}")


if __name__ == "__main__":
    main()