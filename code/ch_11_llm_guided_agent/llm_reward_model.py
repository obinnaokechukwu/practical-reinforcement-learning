"""
Language-based Reward Models for RL

This module implements reward models that use language understanding
to evaluate states and actions based on natural language task descriptions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreferenceData:
    """Data structure for preference learning."""
    instruction: str
    states_a: torch.Tensor
    actions_a: torch.Tensor
    states_b: torch.Tensor
    actions_b: torch.Tensor
    preference: torch.Tensor  # 1 if A preferred, 0 if B preferred


class LanguageRewardModel(nn.Module):
    """
    A reward model that uses language understanding to evaluate state-action pairs.
    
    This model combines a pre-trained language model with learned encoders
    for states and actions to predict reward values based on task descriptions.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        state_dim: int = 10, 
        action_dim: int = 4,
        hidden_dim: int = 256,
        freeze_language_model: bool = True
    ):
        super().__init__()
        
        # Language model for instruction understanding
        self.language_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Optionally freeze language model parameters
        if freeze_language_model:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Fusion and reward prediction
        language_hidden_size = self.language_model.config.hidden_size
        fusion_input_dim = language_hidden_size + hidden_dim + hidden_dim // 2
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Single reward value
        )
        
        # Cache for language embeddings
        self._embedding_cache = {}
        
    def forward(
        self, 
        instruction: str, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward given instruction, state, and action.
        
        Args:
            instruction: String or list of strings with task description
            state: Tensor of shape (batch_size, state_dim)
            action: Tensor of shape (batch_size, action_dim) or (batch_size,) for discrete
        
        Returns:
            reward: Tensor of shape (batch_size,)
        """
        # Handle batch processing
        if isinstance(instruction, str):
            instruction = [instruction] * state.shape[0]
        
        # Get language features (with caching for efficiency)
        language_features = self._get_language_features(instruction)
        
        # Encode state
        state_features = self.state_encoder(state)
        
        # Encode action (handle discrete actions)
        if action.dim() == 1:
            # Convert discrete actions to one-hot
            action_one_hot = torch.zeros(action.shape[0], self.action_encoder[0].in_features)
            action_one_hot.scatter_(1, action.unsqueeze(1).long(), 1)
            action = action_one_hot
        
        action_features = self.action_encoder(action)
        
        # Combine all features
        combined = torch.cat([
            language_features, 
            state_features, 
            action_features
        ], dim=1)
        
        # Predict reward
        reward = self.fusion_layer(combined).squeeze(-1)
        
        return reward
    
    def _get_language_features(self, instructions: List[str]) -> torch.Tensor:
        """Get language features with caching."""
        # Create cache key
        cache_key = tuple(instructions)
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Encode instructions
        encoded = self.tokenizer(
            instructions, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to same device as model
        encoded = {k: v.to(self.state_encoder[0].weight.device) for k, v in encoded.items()}
        
        # Get language representation
        with torch.no_grad() if not self.language_model.training else torch.enable_grad():
            language_output = self.language_model(**encoded)
            # Use [CLS] token representation
            language_features = language_output.last_hidden_state[:, 0, :]
        
        # Cache if in eval mode
        if not self.training:
            self._embedding_cache[cache_key] = language_features
        
        return language_features
    
    def train_from_preferences(
        self, 
        dataset: List[PreferenceData], 
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Train the reward model from human preference comparisons.
        
        Uses the Bradley-Terry model for preference learning.
        """
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=learning_rate
        )
        
        metrics = {
            'loss': [],
            'accuracy': []
        }
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Process batches
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                
                # Compute trajectory rewards
                trajectory_rewards_a = []
                trajectory_rewards_b = []
                
                for data in batch:
                    # Sum rewards over trajectory A
                    rewards_a = []
                    for t in range(data.states_a.shape[0]):
                        r = self(
                            data.instruction, 
                            data.states_a[t].unsqueeze(0), 
                            data.actions_a[t].unsqueeze(0)
                        )
                        rewards_a.append(r)
                    trajectory_rewards_a.append(torch.sum(torch.cat(rewards_a)))
                    
                    # Sum rewards over trajectory B
                    rewards_b = []
                    for t in range(data.states_b.shape[0]):
                        r = self(
                            data.instruction, 
                            data.states_b[t].unsqueeze(0), 
                            data.actions_b[t].unsqueeze(0)
                        )
                        rewards_b.append(r)
                    trajectory_rewards_b.append(torch.sum(torch.cat(rewards_b)))
                
                # Stack rewards
                rewards_a = torch.stack(trajectory_rewards_a)
                rewards_b = torch.stack(trajectory_rewards_b)
                preferences = torch.stack([data.preference for data in batch])
                
                # Bradley-Terry model: P(A > B) = sigmoid(R_A - R_B)
                logits = rewards_a - rewards_b
                pred_probs = torch.sigmoid(logits)
                
                # Binary cross-entropy loss
                loss = -torch.mean(
                    preferences * torch.log(pred_probs + 1e-8) + 
                    (1 - preferences) * torch.log(1 - pred_probs + 1e-8)
                )
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                # Metrics
                epoch_loss += loss.item()
                predictions = (pred_probs > 0.5).float()
                correct += (predictions == preferences).sum().item()
                total += preferences.shape[0]
            
            # Record metrics
            avg_loss = epoch_loss / (len(dataset) / batch_size)
            accuracy = correct / total
            metrics['loss'].append(avg_loss)
            metrics['accuracy'].append(accuracy)
            
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2%}")
        
        return metrics


class RobustLanguageRewardModel(LanguageRewardModel):
    """
    Enhanced reward model with defenses against exploitation.
    
    Includes:
    - Ensemble predictions for uncertainty estimation
    - Adversarial detection
    - Constitutional constraints
    """
    
    def __init__(self, *args, ensemble_size: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Remove single fusion layer
        delattr(self, 'fusion_layer')
        
        # Ensemble of reward heads
        fusion_input_dim = (
            self.language_model.config.hidden_size + 
            kwargs.get('hidden_dim', 256) + 
            kwargs.get('hidden_dim', 256) // 2
        )
        
        self.reward_ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_input_dim, kwargs.get('hidden_dim', 256)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(kwargs.get('hidden_dim', 256), 1)
            ) for _ in range(ensemble_size)
        ])
        
        # Adversarial detector
        self.adversarial_detector = nn.Sequential(
            nn.Linear(fusion_input_dim, kwargs.get('hidden_dim', 256)),
            nn.ReLU(),
            nn.Linear(kwargs.get('hidden_dim', 256), kwargs.get('hidden_dim', 256) // 2),
            nn.ReLU(),
            nn.Linear(kwargs.get('hidden_dim', 256) // 2, 2)  # Binary classification
        )
        
        # Safety constraints
        self.safety_keywords = [
            "crash", "collide", "break", "damage", "hurt", "destroy"
        ]
    
    def forward_with_uncertainty(
        self, 
        instruction: str, 
        state: torch.Tensor, 
        action: torch.Tensor,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reward with uncertainty estimates.
        """
        # Get base features
        if isinstance(instruction, str):
            instruction = [instruction] * state.shape[0]
        
        language_features = self._get_language_features(instruction)
        state_features = self.state_encoder(state)
        
        # Handle discrete actions
        if action.dim() == 1:
            action_one_hot = torch.zeros(action.shape[0], self.action_encoder[0].in_features)
            action_one_hot.scatter_(1, action.unsqueeze(1).long(), 1)
            action = action_one_hot
        
        action_features = self.action_encoder(action)
        
        # Combine features
        combined = torch.cat([
            language_features, 
            state_features, 
            action_features
        ], dim=1)
        
        # Ensemble predictions
        rewards = []
        for head in self.reward_ensemble:
            rewards.append(head(combined))
        
        rewards = torch.stack(rewards, dim=0)  # (ensemble_size, batch_size, 1)
        mean_reward = rewards.mean(dim=0).squeeze(-1)
        std_reward = rewards.std(dim=0).squeeze(-1)
        
        # Adversarial detection
        adv_logits = self.adversarial_detector(combined)
        adversarial_prob = torch.softmax(adv_logits, dim=-1)[:, 1]
        
        # Apply safety constraints
        safety_penalty = self._compute_safety_penalty(instruction, state, action)
        
        # Robust reward: penalize uncertainty and adversarial patterns
        robust_reward = (
            mean_reward - 
            0.5 * std_reward - 
            2.0 * adversarial_prob -
            safety_penalty
        )
        
        result = {
            'reward': robust_reward,
            'mean_reward': mean_reward,
            'uncertainty': std_reward,
            'adversarial_score': adversarial_prob,
            'safety_penalty': safety_penalty
        }
        
        if return_all:
            result['all_rewards'] = rewards.squeeze(-1)
        
        return result
    
    def forward(
        self, 
        instruction: str, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Standard forward for compatibility."""
        return self.forward_with_uncertainty(instruction, state, action)['reward']
    
    def _compute_safety_penalty(
        self, 
        instructions: List[str], 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """Compute safety penalties based on rules."""
        batch_size = state.shape[0]
        penalties = torch.zeros(batch_size, device=state.device)
        
        # Check for safety violations in instructions
        for i, inst in enumerate(instructions):
            for keyword in self.safety_keywords:
                if keyword in inst.lower():
                    penalties[i] += 5.0
        
        # Add domain-specific safety checks here
        # Example: penalize high velocities, extreme positions, etc.
        
        return penalties
    
    def update_adversarial_detector(
        self, 
        normal_data: List[Tuple], 
        adversarial_data: List[Tuple],
        num_epochs: int = 5
    ):
        """Train the adversarial detector."""
        optimizer = optim.Adam(self.adversarial_detector.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Process normal data
            for instruction, state, action in normal_data:
                features = self._get_combined_features(instruction, state, action)
                logits = self.adversarial_detector(features)
                labels = torch.zeros(features.shape[0], dtype=torch.long)
                
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.shape[0]
            
            # Process adversarial data
            for instruction, state, action in adversarial_data:
                features = self._get_combined_features(instruction, state, action)
                logits = self.adversarial_detector(features)
                labels = torch.ones(features.shape[0], dtype=torch.long)
                
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.shape[0]
            
            accuracy = correct / total
            print(f"Adversarial Detector - Epoch {epoch + 1}: "
                  f"Loss = {total_loss:.4f}, Accuracy = {accuracy:.2%}")


# Demo and testing
if __name__ == "__main__":
    # Example: Create and test reward model
    reward_model = RobustLanguageRewardModel(
        model_name="bert-base-uncased",
        state_dim=4,  # e.g., CartPole
        action_dim=2,
        hidden_dim=128
    )
    
    # Test forward pass
    instruction = "Balance the pole without moving too fast"
    state = torch.randn(5, 4)  # Batch of 5 states
    action = torch.randint(0, 2, (5,))  # Discrete actions
    
    result = reward_model.forward_with_uncertainty(
        instruction, state, action, return_all=True
    )
    
    print("Reward Model Output:")
    print(f"Rewards: {result['reward']}")
    print(f"Uncertainty: {result['uncertainty']}")
    print(f"Adversarial Scores: {result['adversarial_score']}")
    
    # Generate synthetic preference data
    def generate_synthetic_preferences(num_samples=100):
        data = []
        for _ in range(num_samples):
            # Random trajectories
            traj_len = np.random.randint(10, 50)
            
            pref_data = PreferenceData(
                instruction="Balance the pole for as long as possible",
                states_a=torch.randn(traj_len, 4),
                actions_a=torch.randint(0, 2, (traj_len,)),
                states_b=torch.randn(traj_len, 4),
                actions_b=torch.randint(0, 2, (traj_len,)),
                preference=torch.tensor(float(np.random.random() > 0.5))
            )
            data.append(pref_data)
        
        return data
    
    # Train on synthetic data
    print("\nTraining on synthetic preferences...")
    preferences = generate_synthetic_preferences(50)
    metrics = reward_model.train_from_preferences(
        preferences, 
        num_epochs=3, 
        batch_size=10
    )
    
    print("\nTraining complete!")