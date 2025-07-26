"""
LLM-Guided Exploration for RL

This module implements exploration strategies that use language models
to suggest interesting states, actions, and subgoals for more efficient learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import gym
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


@dataclass
class ExplorationGoal:
    """Represents an exploration goal suggested by the LLM."""
    description: str
    target_state: Optional[np.ndarray]
    priority: float
    steps_remaining: int
    achieved: bool = False


class StateMemory:
    """Memory system for tracking visited states and their properties."""
    
    def __init__(self, capacity: int = 10000, state_dim: int = 4):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = deque(maxlen=capacity)
        self.visit_counts = {}
        self.state_values = deque(maxlen=capacity)
        self.state_descriptions = deque(maxlen=capacity)
        
    def add(self, state: np.ndarray, value: float = 0.0, description: str = ""):
        """Add a state to memory."""
        self.states.append(state.copy())
        self.state_values.append(value)
        self.state_descriptions.append(description)
        
        # Update visit counts
        state_key = self._state_to_key(state)
        self.visit_counts[state_key] = self.visit_counts.get(state_key, 0) + 1
    
    def get_novelty(self, state: np.ndarray) -> float:
        """Compute novelty score for a state."""
        state_key = self._state_to_key(state)
        visits = self.visit_counts.get(state_key, 0)
        
        # Novelty decreases with visit count
        novelty = 1.0 / (1.0 + visits)
        
        # Also consider distance to nearest neighbor
        if len(self.states) > 0:
            distances = [np.linalg.norm(state - s) for s in self.states]
            min_distance = min(distances)
            distance_novelty = 1.0 - np.exp(-min_distance)
            novelty = 0.5 * novelty + 0.5 * distance_novelty
        
        return novelty
    
    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key."""
        # Discretize for hashing
        discretized = np.round(state, decimals=2)
        return str(discretized.tolist())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            'total_states': len(self.states),
            'unique_states': len(self.visit_counts),
            'avg_visits': np.mean(list(self.visit_counts.values())) if self.visit_counts else 0,
            'avg_value': np.mean(self.state_values) if self.state_values else 0
        }


class LLMExplorationGuide:
    """
    Use LLMs to guide exploration in RL environments.
    """
    
    def __init__(
        self, 
        env_name: str,
        model_name: str = "gpt2",
        state_dim: int = 4,
        action_dim: int = 2,
        use_state_abstraction: bool = True
    ):
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_state_abstraction = use_state_abstraction
        
        # Language model components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # State memory
        self.state_memory = StateMemory(capacity=10000, state_dim=state_dim)
        
        # Exploration goals
        self.current_goals: List[ExplorationGoal] = []
        self.achieved_goals: List[ExplorationGoal] = []
        
        # Curiosity model
        self.curiosity_model = self._build_curiosity_model()
        
        # State abstraction model
        if use_state_abstraction:
            self.state_abstractor = self._build_state_abstractor()
    
    def _build_curiosity_model(self) -> nn.Module:
        """Build neural network for curiosity computation."""
        return nn.Sequential(
            nn.Linear(self.state_dim * 2 + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output curiosity score in [0, 1]
        )
    
    def _build_state_abstractor(self) -> nn.Module:
        """Build model for state abstraction/description."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Compressed representation
        )
    
    def suggest_exploration_goal(
        self, 
        current_state: np.ndarray, 
        task_description: str, 
        history: List[Dict[str, Any]]
    ) -> ExplorationGoal:
        """
        Suggest an interesting exploration goal based on current context.
        """
        # Generate context for LLM
        context = self._create_exploration_context(
            current_state, task_description, history
        )
        
        # Generate goal suggestion
        prompt = f"""{context}

Based on the current situation, suggest an exploration goal that would help the agent learn more effectively. The goal should be:
1. Achievable but challenging
2. Likely to reveal new information about the environment
3. Relevant to the task

Exploration goal: """
        
        # Generate text (in practice, would use more sophisticated generation)
        goal_text = self._generate_text(prompt, max_length=50)
        
        # Parse goal into concrete target
        goal = self._parse_exploration_goal(goal_text, current_state)
        
        # Add to current goals
        self.current_goals.append(goal)
        
        return goal
    
    def _create_exploration_context(
        self, 
        state: np.ndarray, 
        task: str, 
        history: List[Dict[str, Any]]
    ) -> str:
        """Create context string for LLM."""
        # State description
        state_desc = self._state_to_description(state)
        
        # Recent history summary
        history_summary = self._summarize_history(history)
        
        # Memory statistics
        mem_stats = self.state_memory.get_statistics()
        
        context = f"""
Environment: {self.env_name}
Task: {task}
Current State: {state_desc}
Recent History: {history_summary}
Exploration Statistics:
- States visited: {mem_stats['unique_states']}
- Average state value: {mem_stats['avg_value']:.2f}
Current Goals: {[g.description for g in self.current_goals if not g.achieved]}
"""
        return context
    
    def _state_to_description(self, state: np.ndarray) -> str:
        """Convert numerical state to text description."""
        if self.env_name == "CartPole-v1":
            return f"Cart position: {state[0]:.2f}, Cart velocity: {state[1]:.2f}, Pole angle: {state[2]:.2f}, Pole velocity: {state[3]:.2f}"
        elif self.env_name == "MountainCar-v0":
            return f"Position: {state[0]:.2f}, Velocity: {state[1]:.2f}"
        else:
            return f"State vector: {state.tolist()}"
    
    def _summarize_history(self, history: List[Dict[str, Any]], max_steps: int = 10) -> str:
        """Summarize recent history."""
        if not history:
            return "No history yet"
        
        recent = history[-max_steps:]
        total_reward = sum(h.get('reward', 0) for h in recent)
        
        return f"Last {len(recent)} steps: Total reward = {total_reward:.2f}"
    
    def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the language model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + max_length,
                num_return_sequences=1,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part
        generated = generated[len(prompt):].strip()
        
        return generated
    
    def _parse_exploration_goal(self, goal_text: str, current_state: np.ndarray) -> ExplorationGoal:
        """Parse text goal into structured format."""
        # Simple heuristic parsing (in practice, would use more sophisticated NLP)
        
        # Default goal
        goal = ExplorationGoal(
            description=goal_text,
            target_state=None,
            priority=0.5,
            steps_remaining=100
        )
        
        # Extract priority if mentioned
        if "high priority" in goal_text.lower():
            goal.priority = 0.9
        elif "low priority" in goal_text.lower():
            goal.priority = 0.3
        
        # Extract target state if mentioned
        if "reach" in goal_text.lower() or "get to" in goal_text.lower():
            # Create a target state based on heuristics
            if self.env_name == "CartPole-v1":
                if "left" in goal_text.lower():
                    goal.target_state = np.array([-1.0, 0, 0, 0])
                elif "right" in goal_text.lower():
                    goal.target_state = np.array([1.0, 0, 0, 0])
                elif "upright" in goal_text.lower():
                    goal.target_state = current_state.copy()
                    goal.target_state[2] = 0  # Pole angle = 0
            elif self.env_name == "MountainCar-v0":
                if "top" in goal_text.lower() or "goal" in goal_text.lower():
                    goal.target_state = np.array([0.5, 0])
                elif "valley" in goal_text.lower():
                    goal.target_state = np.array([-0.5, 0])
        
        return goal
    
    def compute_curiosity_bonus(
        self, 
        state: np.ndarray, 
        action: int, 
        next_state: np.ndarray, 
        task_context: str
    ) -> float:
        """
        Compute intrinsic reward based on curiosity and goal progress.
        """
        # Base curiosity from neural network
        curiosity_input = np.concatenate([
            state, 
            np.eye(self.action_dim)[action],  # One-hot action
            next_state
        ])
        
        with torch.no_grad():
            base_curiosity = self.curiosity_model(
                torch.FloatTensor(curiosity_input)
            ).item()
        
        # Novelty bonus
        novelty = self.state_memory.get_novelty(next_state)
        
        # Goal progress bonus
        goal_bonus = self._compute_goal_progress(state, next_state)
        
        # Combine bonuses
        total_bonus = (
            0.4 * base_curiosity + 
            0.4 * novelty + 
            0.2 * goal_bonus
        )
        
        # Add state to memory
        self.state_memory.add(next_state, value=total_bonus)
        
        return total_bonus
    
    def _compute_goal_progress(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """Compute progress towards current goals."""
        if not self.current_goals:
            return 0.0
        
        max_progress = 0.0
        
        for goal in self.current_goals:
            if goal.achieved:
                continue
            
            if goal.target_state is not None:
                # Distance-based progress
                dist_before = np.linalg.norm(state - goal.target_state)
                dist_after = np.linalg.norm(next_state - goal.target_state)
                progress = max(0, (dist_before - dist_after) / (dist_before + 1e-6))
                
                # Check if goal achieved
                if dist_after < 0.1:  # Threshold
                    goal.achieved = True
                    self.achieved_goals.append(goal)
                    progress += 1.0  # Bonus for achieving goal
                
                max_progress = max(max_progress, progress * goal.priority)
            
            # Update steps remaining
            goal.steps_remaining -= 1
            if goal.steps_remaining <= 0:
                goal.achieved = True  # Timeout
        
        # Clean up achieved goals
        self.current_goals = [g for g in self.current_goals if not g.achieved]
        
        return max_progress
    
    def generate_exploration_curriculum(self, task_description: str) -> List[str]:
        """
        Generate a curriculum of exploration objectives.
        """
        prompt = f"""
Task: {task_description}
Environment: {self.env_name}

Create a curriculum of 5 exploration objectives that would help an agent learn this task effectively. 
The objectives should progress from simple to complex.

Exploration Curriculum:
1. """
        
        curriculum_text = self._generate_text(prompt, max_length=200)
        
        # Parse into list (simple splitting)
        lines = curriculum_text.split('\n')
        curriculum = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                objective = line.lstrip('0123456789.-').strip()
                if objective:
                    curriculum.append(objective)
        
        # Ensure we have at least some objectives
        if not curriculum:
            curriculum = [
                "Explore the state space boundaries",
                "Find stable positions",
                "Test extreme actions",
                "Discover state transitions",
                "Achieve the main task goal"
            ]
        
        return curriculum[:5]  # Limit to 5 objectives
    
    def update_curiosity_model(self, experiences: List[Tuple]):
        """
        Update the curiosity model based on prediction errors.
        """
        if len(experiences) < 32:
            return
        
        optimizer = torch.optim.Adam(self.curiosity_model.parameters(), lr=1e-3)
        
        # Train for a few epochs
        for _ in range(5):
            # Sample batch
            batch_idx = np.random.choice(len(experiences), 32)
            batch = [experiences[i] for i in batch_idx]
            
            # Prepare data
            states = torch.FloatTensor([e[0] for e in batch])
            actions = torch.LongTensor([e[1] for e in batch])
            next_states = torch.FloatTensor([e[2] for e in batch])
            rewards = torch.FloatTensor([e[3] for e in batch])
            
            # Create input
            action_one_hot = torch.zeros(32, self.action_dim)
            action_one_hot.scatter_(1, actions.unsqueeze(1), 1)
            
            curiosity_input = torch.cat([states, action_one_hot, next_states], dim=1)
            
            # Predict curiosity
            predicted_curiosity = self.curiosity_model(curiosity_input).squeeze()
            
            # Target: higher curiosity for higher rewards (simplified)
            target_curiosity = torch.sigmoid(rewards)
            
            # Loss
            loss = nn.MSELoss()(predicted_curiosity, target_curiosity)
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def get_action_suggestion(self, state: np.ndarray, available_actions: List[int]) -> Optional[int]:
        """
        Suggest an action that might lead to interesting outcomes.
        """
        if not self.current_goals or np.random.random() > 0.3:
            return None  # Let the agent decide
        
        # Get highest priority goal with target state
        active_goals = [g for g in self.current_goals if not g.achieved and g.target_state is not None]
        if not active_goals:
            return None
        
        goal = max(active_goals, key=lambda g: g.priority)
        
        # Simple heuristic: choose action that moves towards goal
        best_action = None
        min_distance = float('inf')
        
        for action in available_actions:
            # Predict next state (simple linear approximation)
            if self.env_name == "CartPole-v1":
                # Rough approximation
                next_state = state.copy()
                if action == 0:  # Left
                    next_state[1] -= 0.1  # Velocity
                else:  # Right
                    next_state[1] += 0.1
                next_state[0] += next_state[1] * 0.02  # Position
            else:
                next_state = state  # Fallback
            
            dist = np.linalg.norm(next_state - goal.target_state)
            if dist < min_distance:
                min_distance = dist
                best_action = action
        
        return best_action


# Curiosity-driven agent using LLM guidance
class CuriosityDrivenAgent:
    """
    RL agent that uses LLM-guided curiosity for exploration.
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        llm_guide: LLMExplorationGuide,
        intrinsic_reward_weight: float = 0.5
    ):
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Value network
        self.value_function = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.llm_guide = llm_guide
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.recent_states = deque(maxlen=100)
        self.task_context = ""
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=3e-4)
    
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action with optional LLM-guided exploration.
        """
        state_tensor = torch.FloatTensor(state)
        
        # Get policy distribution
        with torch.no_grad():
            action_logits = self.policy(state_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
        
        if explore and np.random.random() < 0.2:
            # Get LLM suggestion
            available_actions = list(range(len(action_probs)))
            suggested_action = self.llm_guide.get_action_suggestion(state, available_actions)
            
            if suggested_action is not None:
                # Bias towards suggested action
                action_probs[suggested_action] *= 2.0
                action_probs /= action_probs.sum()
        
        # Sample action
        action = torch.multinomial(action_probs, 1).item()
        
        # Track states
        self.recent_states.append(state)
        
        return action
    
    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        running_return = 0
        
        for r in reversed(rewards):
            running_return = r + gamma * running_return
            returns.insert(0, running_return)
        
        return torch.FloatTensor(returns)
    
    def update(self, trajectory: List[Dict[str, Any]]):
        """
        Update policy and value function using trajectory.
        """
        # Extract data
        states = torch.FloatTensor([t['state'] for t in trajectory])
        actions = torch.LongTensor([t['action'] for t in trajectory])
        rewards = [t['reward'] for t in trajectory]
        
        # Add intrinsic rewards
        intrinsic_rewards = []
        for i in range(len(trajectory) - 1):
            curiosity = self.llm_guide.compute_curiosity_bonus(
                trajectory[i]['state'],
                trajectory[i]['action'],
                trajectory[i+1]['state'],
                self.task_context
            )
            intrinsic_rewards.append(curiosity)
        intrinsic_rewards.append(0)  # No curiosity for last state
        
        # Combine rewards
        total_rewards = [
            r + self.intrinsic_reward_weight * ir 
            for r, ir in zip(rewards, intrinsic_rewards)
        ]
        
        # Compute returns
        returns = self.compute_returns(total_rewards)
        
        # Update value function
        values = self.value_function(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy (REINFORCE with baseline)
        with torch.no_grad():
            values = self.value_function(states).squeeze()
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        action_logits = self.policy(states)
        action_log_probs = torch.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        policy_loss = -(selected_log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'avg_curiosity': np.mean(intrinsic_rewards)
        }


# Demo
if __name__ == "__main__":
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Create exploration guide
    guide = LLMExplorationGuide(
        env_name="CartPole-v1",
        model_name="gpt2",
        state_dim=4,
        action_dim=2
    )
    
    # Generate exploration curriculum
    task = "Balance the pole for as long as possible"
    curriculum = guide.generate_exploration_curriculum(task)
    
    print("Exploration Curriculum:")
    for i, objective in enumerate(curriculum, 1):
        print(f"{i}. {objective}")
    
    # Create agent
    agent = CuriosityDrivenAgent(
        state_dim=4,
        action_dim=2,
        llm_guide=guide
    )
    agent.task_context = task
    
    # Training loop
    print("\nTraining with curiosity-driven exploration...")
    
    episode_rewards = []
    curiosity_bonuses = []
    
    for episode in range(100):
        state = env.reset()
        trajectory = []
        total_reward = 0
        total_curiosity = 0
        
        # Occasionally suggest new exploration goals
        if episode % 20 == 0:
            history = [{'reward': r} for r in episode_rewards[-10:]]
            goal = guide.suggest_exploration_goal(state, task, history)
            print(f"\nEpisode {episode} - New exploration goal: {goal.description}")
        
        done = False
        while not done:
            action = agent.act(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            # Track curiosity
            if not done:
                curiosity = guide.compute_curiosity_bonus(
                    state, action, next_state, task
                )
                total_curiosity += curiosity
            
            state = next_state
            total_reward += reward
        
        # Update agent
        update_info = agent.update(trajectory)
        
        episode_rewards.append(total_reward)
        curiosity_bonuses.append(total_curiosity)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_curiosity = np.mean(curiosity_bonuses[-10:])
            print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                  f"Avg Curiosity = {avg_curiosity:.3f}")
    
    env.close()
    print("\nTraining complete!")
    
    # Print exploration statistics
    stats = guide.state_memory.get_statistics()
    print(f"\nExploration Statistics:")
    print(f"Unique states visited: {stats['unique_states']}")
    print(f"Goals achieved: {len(guide.achieved_goals)}")