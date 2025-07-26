"""
Hierarchical Policy Generation using LLMs

This module implements hierarchical RL policies where high-level skills
are generated and composed based on language descriptions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import OrderedDict
import gym
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class Skill:
    """Represents a reusable skill."""
    name: str
    description: str
    policy: nn.Module
    termination_condition: Callable
    max_steps: int = 50
    learned_from: str = "generated"  # "generated", "demonstrated", "discovered"


class SkillLibrary:
    """Manages a library of reusable skills."""
    
    def __init__(self):
        self.skills: Dict[str, Skill] = OrderedDict()
        self.skill_usage_count: Dict[str, int] = {}
        self.skill_success_rate: Dict[str, float] = {}
        self.skill_embeddings: Dict[str, torch.Tensor] = {}
    
    def add_skill(self, skill: Skill):
        """Add a skill to the library."""
        self.skills[skill.name] = skill
        self.skill_usage_count[skill.name] = 0
        self.skill_success_rate[skill.name] = 0.5  # Initial estimate
    
    def get_skill(self, name: str) -> Optional[Skill]:
        """Retrieve a skill by name."""
        return self.skills.get(name)
    
    def find_similar_skills(self, description: str, top_k: int = 3) -> List[str]:
        """Find skills with similar descriptions."""
        # In practice, would use embeddings for similarity
        # For now, simple keyword matching
        description_lower = description.lower()
        scores = {}
        
        for name, skill in self.skills.items():
            skill_desc_lower = skill.description.lower()
            # Count common words
            desc_words = set(description_lower.split())
            skill_words = set(skill_desc_lower.split())
            common_words = desc_words.intersection(skill_words)
            scores[name] = len(common_words) / max(len(desc_words), 1)
        
        # Sort by score
        sorted_skills = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in sorted_skills[:top_k] if score > 0]
    
    def update_skill_stats(self, skill_name: str, success: bool):
        """Update usage statistics for a skill."""
        if skill_name in self.skills:
            self.skill_usage_count[skill_name] += 1
            # Running average of success rate
            alpha = 0.1
            current_rate = self.skill_success_rate[skill_name]
            self.skill_success_rate[skill_name] = (
                (1 - alpha) * current_rate + alpha * float(success)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            'total_skills': len(self.skills),
            'total_usage': sum(self.skill_usage_count.values()),
            'avg_success_rate': np.mean(list(self.skill_success_rate.values())),
            'most_used_skill': max(self.skill_usage_count, key=self.skill_usage_count.get)
            if self.skill_usage_count else None
        }


class SkillPolicy(nn.Module):
    """Base class for skill policies."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action logits."""
        return self.network(state)
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action given state."""
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            logits = self.forward(state_tensor)
            probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            action = torch.multinomial(probs, 1).item()
        
        return action


class LLMPolicyGenerator:
    """
    Generate RL policies from language descriptions of skills.
    """
    
    def __init__(
        self, 
        base_model: str = "gpt2",
        env_description: str = "",
        state_dim: int = 4,
        action_dim: int = 2
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.env_description = env_description
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Skill library
        self.skill_library = SkillLibrary()
        
        # Initialize with basic skills
        self._initialize_basic_skills()
    
    def _initialize_basic_skills(self):
        """Initialize library with basic primitive skills."""
        # Basic movement skills
        basic_skills = [
            {
                'name': 'move_left',
                'description': 'Move or push to the left',
                'action_bias': 0  # For discrete action spaces
            },
            {
                'name': 'move_right',
                'description': 'Move or push to the right',
                'action_bias': 1
            },
            {
                'name': 'stay_still',
                'description': 'Maintain current position',
                'action_bias': None  # Alternate actions
            }
        ]
        
        for skill_def in basic_skills:
            skill = self._create_basic_skill(skill_def)
            self.skill_library.add_skill(skill)
    
    def _create_basic_skill(self, skill_def: Dict[str, Any]) -> Skill:
        """Create a basic skill from definition."""
        # Create biased policy
        policy = BiasedSkillPolicy(
            self.state_dim, 
            self.action_dim,
            action_bias=skill_def.get('action_bias')
        )
        
        # Simple termination after fixed steps
        def termination(state, steps):
            return steps >= 20
        
        return Skill(
            name=skill_def['name'],
            description=skill_def['description'],
            policy=policy,
            termination_condition=termination,
            max_steps=20,
            learned_from="generated"
        )
    
    def decompose_task(self, task_description: str) -> List[str]:
        """
        Decompose a complex task into a sequence of skills.
        """
        # Check if we can reuse existing skills
        similar_skills = self.skill_library.find_similar_skills(task_description)
        
        prompt = f"""
Environment: {self.env_description}
Task: {task_description}
Available skills: {list(self.skill_library.skills.keys())}
Similar existing skills: {similar_skills}

Break down this task into a sequence of simpler skills. Each skill should be:
1. A clear, atomic action or behavior
2. Reusable in other contexts
3. Achievable with a simple policy

If existing skills can be reused, prefer those. Otherwise, suggest new skills.

Skill sequence:
1."""
        
        # Generate decomposition
        generated = self._generate_text(prompt, max_length=150)
        
        # Parse skills
        skills = self._parse_skill_sequence(generated)
        
        return skills
    
    def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using language model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=min(len(inputs.input_ids[0]) + max_length, 512),
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip()
    
    def _parse_skill_sequence(self, generated_text: str) -> List[str]:
        """Parse generated text into skill sequence."""
        skills = []
        lines = generated_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered items or bullet points
            if line and (line[0].isdigit() or line.startswith('-')):
                # Extract skill name/description
                skill_text = line.lstrip('0123456789.-').strip()
                if skill_text:
                    # Check if it's an existing skill
                    skill_lower = skill_text.lower()
                    matched = False
                    
                    for existing_name in self.skill_library.skills:
                        if existing_name in skill_lower:
                            skills.append(existing_name)
                            matched = True
                            break
                    
                    if not matched:
                        # New skill needed
                        skills.append(skill_text)
        
        # Fallback if parsing fails
        if not skills:
            skills = ['move_right', 'stay_still', 'move_left']
        
        return skills
    
    def generate_skill_policy(self, skill_description: str) -> Skill:
        """
        Generate a new skill policy from description.
        """
        # Check if similar skill exists
        similar = self.skill_library.find_similar_skills(skill_description)
        if similar and similar[0] in self.skill_library.skills:
            # Adapt existing skill
            base_skill = self.skill_library.get_skill(similar[0])
            print(f"Adapting existing skill '{similar[0]}' for '{skill_description}'")
            
            # Create variant
            new_policy = self._create_skill_variant(base_skill, skill_description)
        else:
            # Generate new skill
            print(f"Generating new skill for '{skill_description}'")
            new_policy = self._generate_new_skill(skill_description)
        
        # Create termination condition
        termination = self._generate_termination_condition(skill_description)
        
        # Create skill object
        skill = Skill(
            name=skill_description.replace(' ', '_').lower()[:20],
            description=skill_description,
            policy=new_policy,
            termination_condition=termination,
            max_steps=50
        )
        
        # Add to library
        self.skill_library.add_skill(skill)
        
        return skill
    
    def _create_skill_variant(self, base_skill: Skill, new_description: str) -> nn.Module:
        """Create a variant of an existing skill."""
        # Copy base policy
        new_policy = SkillPolicy(self.state_dim, self.action_dim)
        new_policy.load_state_dict(base_skill.policy.state_dict())
        
        # Add small random perturbation for diversity
        with torch.no_grad():
            for param in new_policy.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        return new_policy
    
    def _generate_new_skill(self, description: str) -> nn.Module:
        """Generate a completely new skill policy."""
        # Analyze description for hints
        description_lower = description.lower()
        
        # Create policy with appropriate inductive bias
        if any(word in description_lower for word in ['left', 'decrease', 'reduce']):
            policy = BiasedSkillPolicy(self.state_dim, self.action_dim, action_bias=0)
        elif any(word in description_lower for word in ['right', 'increase', 'more']):
            policy = BiasedSkillPolicy(self.state_dim, self.action_dim, action_bias=1)
        elif any(word in description_lower for word in ['balance', 'maintain', 'stable']):
            policy = AdaptiveSkillPolicy(self.state_dim, self.action_dim)
        else:
            policy = SkillPolicy(self.state_dim, self.action_dim)
        
        return policy
    
    def _generate_termination_condition(self, description: str) -> Callable:
        """Generate termination condition for skill."""
        description_lower = description.lower()
        
        # Time-based termination
        if any(word in description_lower for word in ['brief', 'quick', 'short']):
            max_steps = 10
        elif any(word in description_lower for word in ['long', 'extended', 'sustained']):
            max_steps = 100
        else:
            max_steps = 50
        
        # State-based termination
        if 'until' in description_lower or 'reach' in description_lower:
            def termination(state, steps):
                # Check for goal conditions
                if 'stable' in description_lower and len(state) > 2:
                    # Check if pole is upright (for CartPole)
                    return abs(state[2]) < 0.1 or steps >= max_steps
                else:
                    return steps >= max_steps
        else:
            def termination(state, steps):
                return steps >= max_steps
        
        return termination
    
    def create_hierarchical_policy(self, task_description: str) -> 'HierarchicalPolicy':
        """
        Create a complete hierarchical policy for a task.
        """
        # Decompose task
        skill_sequence = self.decompose_task(task_description)
        print(f"Task decomposition: {skill_sequence}")
        
        # Generate or retrieve policies for each skill
        skill_policies = OrderedDict()
        for skill_name in skill_sequence:
            if skill_name in self.skill_library.skills:
                skill = self.skill_library.get_skill(skill_name)
            else:
                skill = self.generate_skill_policy(skill_name)
            skill_policies[skill.name] = skill
        
        # Create high-level controller
        controller = HighLevelController(
            skill_sequence=list(skill_policies.keys()),
            state_dim=self.state_dim
        )
        
        return HierarchicalPolicy(
            controller=controller,
            skill_policies=skill_policies,
            skill_library=self.skill_library
        )


class BiasedSkillPolicy(SkillPolicy):
    """Skill policy with action bias."""
    
    def __init__(self, state_dim: int, action_dim: int, action_bias: Optional[int] = None):
        super().__init__(state_dim, action_dim)
        self.action_bias = action_bias
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = super().forward(state)
        
        # Apply bias
        if self.action_bias is not None and self.action_bias < self.action_dim:
            logits[self.action_bias] += 2.0  # Strong bias
        
        return logits


class AdaptiveSkillPolicy(SkillPolicy):
    """Skill policy that adapts based on state."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim)
        
        # Additional adaptation network
        self.adapter = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base_logits = super().forward(state)
        adaptation = self.adapter(state)
        
        # Combine base policy with adaptation
        return base_logits + 0.5 * adaptation


class HighLevelController(nn.Module):
    """High-level controller for skill selection."""
    
    def __init__(self, skill_sequence: List[str], state_dim: int):
        super().__init__()
        self.skill_sequence = skill_sequence
        self.num_skills = len(skill_sequence)
        self.state_dim = state_dim
        
        # Neural network for skill selection
        self.network = nn.Sequential(
            nn.Linear(state_dim + self.num_skills, 128),  # State + one-hot skill history
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_skills)
        )
        
        # Skill transition preferences (learned)
        self.transition_matrix = nn.Parameter(
            torch.zeros(self.num_skills, self.num_skills)
        )
    
    def forward(self, state: torch.Tensor, skill_history: torch.Tensor) -> torch.Tensor:
        """
        Select next skill based on state and history.
        
        Args:
            state: Current state
            skill_history: One-hot encoding of previous skill
        
        Returns:
            Skill selection probabilities
        """
        # Combine state and history
        features = torch.cat([state, skill_history], dim=-1)
        
        # Get base preferences
        logits = self.network(features)
        
        # Apply transition preferences
        if skill_history.sum() > 0:  # If we have a previous skill
            prev_skill_idx = torch.argmax(skill_history)
            transition_bonus = self.transition_matrix[prev_skill_idx]
            logits = logits + transition_bonus
        
        return torch.softmax(logits, dim=-1)
    
    def get_skill_name(self, skill_idx: int) -> str:
        """Get skill name from index."""
        return self.skill_sequence[skill_idx]


class HierarchicalPolicy(nn.Module):
    """
    Complete hierarchical policy with skill decomposition.
    """
    
    def __init__(
        self, 
        controller: HighLevelController,
        skill_policies: Dict[str, Skill],
        skill_library: SkillLibrary
    ):
        super().__init__()
        self.controller = controller
        self.skill_policies = nn.ModuleDict({
            name: skill.policy for name, skill in skill_policies.items()
        })
        self.skills = skill_policies
        self.skill_library = skill_library
        
        # Execution state
        self.current_skill_name = None
        self.current_skill = None
        self.skill_step = 0
        self.skill_history = torch.zeros(controller.num_skills)
        
        # Metrics
        self.skill_execution_count = {name: 0 for name in skill_policies}
        self.skill_success_count = {name: 0 for name in skill_policies}
    
    def forward(self, state: torch.Tensor) -> Tuple[int, Dict[str, Any]]:
        """
        Execute hierarchical policy.
        
        Returns:
            action: Selected action
            info: Dictionary with execution information
        """
        # Check if we need to select a new skill
        if self._should_select_new_skill(state.numpy()):
            self._select_new_skill(state)
        
        # Execute current skill
        action = self.current_skill.policy.act(state.numpy())
        self.skill_step += 1
        
        info = {
            'current_skill': self.current_skill_name,
            'skill_step': self.skill_step,
            'skill_history': self.skill_history.clone()
        }
        
        return action, info
    
    def _should_select_new_skill(self, state: np.ndarray) -> bool:
        """Determine if we should switch skills."""
        if self.current_skill is None:
            return True
        
        # Check termination condition
        terminated = self.current_skill.termination_condition(state, self.skill_step)
        
        # Check max steps
        if self.skill_step >= self.current_skill.max_steps:
            terminated = True
        
        return terminated
    
    def _select_new_skill(self, state: torch.Tensor):
        """Select and switch to a new skill."""
        # Mark previous skill completion
        if self.current_skill_name:
            success = self._evaluate_skill_success(state.numpy())
            self.skill_success_count[self.current_skill_name] += int(success)
            self.skill_library.update_skill_stats(self.current_skill_name, success)
        
        # Get skill probabilities from controller
        skill_probs = self.controller(state, self.skill_history)
        
        # Sample skill
        skill_idx = torch.multinomial(skill_probs, 1).item()
        skill_name = self.controller.get_skill_name(skill_idx)
        
        # Update state
        self.current_skill_name = skill_name
        self.current_skill = self.skills[skill_name]
        self.skill_step = 0
        self.skill_execution_count[skill_name] += 1
        
        # Update history
        self.skill_history.zero_()
        self.skill_history[skill_idx] = 1.0
    
    def _evaluate_skill_success(self, state: np.ndarray) -> bool:
        """Evaluate if the skill was executed successfully."""
        # Simple heuristic - skill succeeded if we didn't fail
        # In practice, would have skill-specific success criteria
        if hasattr(state, '__len__') and len(state) > 2:
            # For CartPole: success if pole is still upright
            return abs(state[2]) < 0.5
        return True
    
    def reset(self):
        """Reset execution state."""
        self.current_skill_name = None
        self.current_skill = None
        self.skill_step = 0
        self.skill_history.zero_()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        metrics = {}
        for skill_name in self.skill_execution_count:
            count = self.skill_execution_count[skill_name]
            if count > 0:
                success_rate = self.skill_success_count[skill_name] / count
                metrics[f'{skill_name}_usage'] = count
                metrics[f'{skill_name}_success_rate'] = success_rate
        
        return metrics
    
    def adapt_online(self, trajectories: List[List[Dict]]):
        """
        Adapt the hierarchical policy based on experience.
        """
        # Update controller based on skill transitions
        optimizer = optim.Adam(self.controller.parameters(), lr=1e-3)
        
        for trajectory in trajectories:
            # Extract skill transitions and rewards
            skill_transitions = []
            rewards = []
            
            for i in range(len(trajectory) - 1):
                if trajectory[i]['skill'] != trajectory[i+1]['skill']:
                    skill_transitions.append({
                        'state': trajectory[i]['state'],
                        'from_skill': trajectory[i]['skill'],
                        'to_skill': trajectory[i+1]['skill'],
                        'reward': sum(t['reward'] for t in trajectory[i:i+10])  # Future reward
                    })
            
            # Update controller to prefer high-reward transitions
            for transition in skill_transitions:
                state = torch.FloatTensor(transition['state'])
                from_idx = self.controller.skill_sequence.index(transition['from_skill'])
                to_idx = self.controller.skill_sequence.index(transition['to_skill'])
                reward = transition['reward']
                
                # Gradient ascent on good transitions
                skill_history = torch.zeros(self.controller.num_skills)
                skill_history[from_idx] = 1.0
                
                probs = self.controller(state.unsqueeze(0), skill_history.unsqueeze(0))
                log_prob = torch.log(probs[0, to_idx] + 1e-8)
                
                loss = -log_prob * reward  # Negative for gradient ascent
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


# Demo
if __name__ == "__main__":
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Create policy generator
    generator = LLMPolicyGenerator(
        base_model="gpt2",
        env_description="CartPole: Move cart left/right to balance a pole",
        state_dim=4,
        action_dim=2
    )
    
    # Generate hierarchical policy for a complex task
    task = "Balance the pole by first centering the cart, then making small adjustments to keep it stable"
    
    print(f"Task: {task}")
    print("\nGenerating hierarchical policy...")
    
    hierarchical_policy = generator.create_hierarchical_policy(task)
    
    print(f"\nSkills in policy: {list(hierarchical_policy.skills.keys())}")
    
    # Test the policy
    print("\nTesting hierarchical policy...")
    
    total_rewards = []
    
    for episode in range(10):
        state = env.reset()
        hierarchical_policy.reset()
        
        episode_reward = 0
        trajectory = []
        
        done = False
        step = 0
        
        while not done and step < 500:
            state_tensor = torch.FloatTensor(state)
            action, info = hierarchical_policy(state_tensor)
            
            next_state, reward, done, _ = env.step(action)
            
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'skill': info['current_skill'],
                'skill_step': info['skill_step']
            })
            
            state = next_state
            episode_reward += reward
            step += 1
        
        total_rewards.append(episode_reward)
        
        if episode == 0:
            # Print skill execution trace for first episode
            print(f"\nEpisode 1 skill execution:")
            current_skill = None
            for t in trajectory[:50]:  # First 50 steps
                if t['skill'] != current_skill:
                    current_skill = t['skill']
                    print(f"  Step {trajectory.index(t)}: Switched to {current_skill}")
    
    print(f"\nAverage reward over 10 episodes: {np.mean(total_rewards):.2f}")
    
    # Print skill usage statistics
    metrics = hierarchical_policy.get_metrics()
    print("\nSkill usage statistics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # Print library statistics
    lib_stats = generator.skill_library.get_statistics()
    print(f"\nSkill library statistics:")
    print(f"  Total skills: {lib_stats['total_skills']}")
    print(f"  Most used: {lib_stats['most_used_skill']}")
    
    env.close()