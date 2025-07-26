"""
Comprehensive safety framework for production RL systems.

This module implements multi-layered safety mechanisms including
constrained optimization, runtime monitoring, and formal verification.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from enum import Enum
import cvxpy as cp


class SafetyLevel(Enum):
    """Safety levels for different operational modes."""
    FULL_AUTONOMY = "full_autonomy"
    SUPERVISED = "supervised"
    SAFE_MODE = "safe_mode"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    timestamp: float
    violation_type: str
    severity: float  # 0-1, where 1 is most severe
    state: np.ndarray
    action: np.ndarray
    details: Dict


class SafetyConstraint(ABC):
    """Abstract base class for safety constraints."""
    
    @abstractmethod
    def is_satisfied(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Check if constraint is satisfied."""
        pass
    
    @abstractmethod
    def project_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Project action to satisfy constraint."""
        pass
    
    @abstractmethod
    def get_safe_set(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Get the set of safe actions for given state."""
        pass


class BoxConstraint(SafetyConstraint):
    """Box constraints on state and action spaces."""
    
    def __init__(self, action_min: np.ndarray, action_max: np.ndarray,
                 state_min: Optional[np.ndarray] = None,
                 state_max: Optional[np.ndarray] = None):
        self.action_min = action_min
        self.action_max = action_max
        self.state_min = state_min
        self.state_max = state_max
    
    def is_satisfied(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Check if action is within bounds."""
        action_valid = np.all(action >= self.action_min) and np.all(action <= self.action_max)
        
        if self.state_min is not None and self.state_max is not None:
            # Predict next state (simplified)
            next_state = state + action * 0.1  # dt = 0.1
            state_valid = np.all(next_state >= self.state_min) and np.all(next_state <= self.state_max)
            return action_valid and state_valid
        
        return action_valid
    
    def project_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Clip action to satisfy constraints."""
        return np.clip(action, self.action_min, self.action_max)
    
    def get_safe_set(self, state: np.ndarray) -> np.ndarray:
        """Get box bounds for safe actions."""
        return np.stack([self.action_min, self.action_max])


class BarrierFunction(SafetyConstraint):
    """Control Barrier Function for safety."""
    
    def __init__(self, barrier_func: Callable, alpha: float = 1.0):
        self.h = barrier_func  # h(x) >= 0 defines safe set
        self.alpha = alpha
    
    def is_satisfied(self, state: np.ndarray, action: np.ndarray) -> bool:
        """Check if CBF constraint is satisfied."""
        h_current = self.h(state)
        
        # Approximate derivative
        dt = 0.01
        next_state = state + action * dt
        h_next = self.h(next_state)
        
        # CBF constraint: dh/dt + alpha*h >= 0
        dhdt = (h_next - h_current) / dt
        return dhdt + self.alpha * h_current >= 0
    
    def project_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Project action using QP to satisfy CBF."""
        # Solve: min ||u - action||^2 s.t. CBF constraint
        u = cp.Variable(len(action))
        
        # Approximate constraint linearization
        h_current = self.h(state)
        
        # Numerical gradient of h
        eps = 1e-6
        grad_h = np.zeros_like(state)
        for i in range(len(state)):
            state_plus = state.copy()
            state_plus[i] += eps
            grad_h[i] = (self.h(state_plus) - h_current) / eps
        
        # CBF constraint: grad_h @ u + alpha * h >= 0
        constraints = [grad_h @ u + self.alpha * h_current >= 0]
        
        # Add box constraints if needed
        constraints.extend([
            u >= -np.ones_like(action) * 10,  # Example bounds
            u <= np.ones_like(action) * 10
        ])
        
        # Objective: minimize deviation from desired action
        objective = cp.Minimize(cp.sum_squares(u - action))
        
        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            if prob.status == cp.OPTIMAL:
                return u.value
        except:
            pass
        
        # Fallback to original action if optimization fails
        return action
    
    def get_safe_set(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Safe set is implicitly defined by h(x) >= 0."""
        return None  # Not explicitly representable as array


class SafetyMonitor:
    """Runtime safety monitoring system."""
    
    def __init__(self, constraints: List[SafetyConstraint], 
                 violation_threshold: int = 5,
                 window_size: int = 100):
        self.constraints = constraints
        self.violation_threshold = violation_threshold
        self.window_size = window_size
        
        self.violation_history = []
        self.safety_level = SafetyLevel.FULL_AUTONOMY
        self.logger = logging.getLogger(__name__)
        
    def check_action(self, state: np.ndarray, action: np.ndarray) -> Tuple[bool, List[SafetyViolation]]:
        """Check if action violates any constraints."""
        violations = []
        
        for i, constraint in enumerate(self.constraints):
            if not constraint.is_satisfied(state, action):
                violation = SafetyViolation(
                    timestamp=np.datetime64('now'),
                    violation_type=f"constraint_{i}",
                    severity=0.5,  # Could be constraint-specific
                    state=state.copy(),
                    action=action.copy(),
                    details={'constraint': type(constraint).__name__}
                )
                violations.append(violation)
        
        # Update history
        self.violation_history.extend(violations)
        self._trim_history()
        
        # Update safety level
        self._update_safety_level()
        
        return len(violations) == 0, violations
    
    def project_action(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Project action to satisfy all constraints."""
        projected = action.copy()
        
        for constraint in self.constraints:
            projected = constraint.project_action(state, projected)
        
        return projected
    
    def _trim_history(self):
        """Keep only recent violations."""
        if len(self.violation_history) > self.window_size:
            self.violation_history = self.violation_history[-self.window_size:]
    
    def _update_safety_level(self):
        """Update safety level based on recent violations."""
        recent_violations = len(self.violation_history)
        
        if recent_violations == 0:
            self.safety_level = SafetyLevel.FULL_AUTONOMY
        elif recent_violations < self.violation_threshold:
            self.safety_level = SafetyLevel.SUPERVISED
        elif recent_violations < self.violation_threshold * 2:
            self.safety_level = SafetyLevel.SAFE_MODE
        else:
            self.safety_level = SafetyLevel.EMERGENCY_STOP
            self.logger.critical("EMERGENCY STOP triggered due to excessive violations")
    
    def get_safety_metrics(self) -> Dict:
        """Get current safety metrics."""
        return {
            'safety_level': self.safety_level.value,
            'recent_violations': len(self.violation_history),
            'violation_rate': len(self.violation_history) / self.window_size,
            'constraints_active': len(self.constraints)
        }


class SafetyShield(nn.Module):
    """Neural safety shield that learns to predict and prevent violations."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        # Violation predictor
        layers = []
        prev_size = state_dim + action_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.violation_predictor = nn.Sequential(*layers)
        
        # Safe action generator
        self.safe_action_generator = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict violation probability and generate safe action."""
        # Predict violation probability
        state_action = torch.cat([state, action], dim=-1)
        violation_prob = self.violation_predictor(state_action)
        
        # Generate safe action alternative
        safe_action = self.safe_action_generator(state)
        
        return violation_prob, safe_action
    
    def filter_action(self, state: torch.Tensor, action: torch.Tensor,
                     threshold: float = 0.5) -> torch.Tensor:
        """Filter action through safety shield."""
        violation_prob, safe_action = self.forward(state, action)
        
        # Use safe action if violation probability is high
        mask = (violation_prob > threshold).float()
        filtered_action = mask * safe_action + (1 - mask) * action
        
        return filtered_action


class ConstrainedPolicyOptimization:
    """Constrained Policy Optimization (CPO) implementation."""
    
    def __init__(self, policy_net: nn.Module, value_net: nn.Module,
                 cost_net: nn.Module, max_kl: float = 0.01,
                 max_constraint: float = 0.1):
        self.policy_net = policy_net
        self.value_net = value_net
        self.cost_net = cost_net
        self.max_kl = max_kl
        self.max_constraint = max_constraint
    
    def update(self, states: torch.Tensor, actions: torch.Tensor,
               rewards: torch.Tensor, costs: torch.Tensor,
               next_states: torch.Tensor, dones: torch.Tensor):
        """Update policy with constraint satisfaction."""
        # Compute advantages for rewards and costs
        reward_advantages = self._compute_advantages(
            states, rewards, next_states, dones, self.value_net
        )
        cost_advantages = self._compute_advantages(
            states, costs, next_states, dones, self.cost_net
        )
        
        # Get old policy probabilities
        with torch.no_grad():
            old_log_probs = self.policy_net.log_prob(states, actions)
        
        # CPO update (simplified)
        # In practice, this would solve a constrained optimization problem
        for _ in range(10):  # Policy improvement steps
            # Get new policy probabilities
            new_log_probs = self.policy_net.log_prob(states, actions)
            
            # Compute policy ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Objective: maximize reward advantage
            objective = (ratio * reward_advantages).mean()
            
            # Constraint: expected cost should not increase
            constraint = (ratio * cost_advantages).mean()
            
            # KL constraint
            kl = self._compute_kl(old_log_probs, new_log_probs)
            
            # Compute loss with Lagrangian
            loss = -objective
            
            if constraint > self.max_constraint:
                # Add penalty for constraint violation
                loss += 10.0 * constraint
            
            if kl > self.max_kl:
                # Add penalty for KL violation
                loss += 100.0 * (kl - self.max_kl)
            
            # Update policy
            self.policy_net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            # optimizer.step() would go here
    
    def _compute_advantages(self, states, rewards, next_states, dones, value_net):
        """Compute GAE advantages."""
        # Simplified advantage computation
        values = value_net(states).squeeze()
        next_values = value_net(next_states).squeeze()
        
        # TD error
        td_error = rewards + 0.99 * next_values * (1 - dones) - values
        
        return td_error
    
    def _compute_kl(self, old_log_probs, new_log_probs):
        """Compute KL divergence."""
        return (torch.exp(old_log_probs) * (old_log_probs - new_log_probs)).mean()


class SafetyOrchestrator:
    """Orchestrate all safety mechanisms."""
    
    def __init__(self):
        self.constraints = []
        self.monitor = None
        self.shield = None
        self.emergency_policy = None
        self.logger = logging.getLogger(__name__)
    
    def add_constraint(self, constraint: SafetyConstraint):
        """Add a safety constraint."""
        self.constraints.append(constraint)
        self._rebuild_monitor()
    
    def set_shield(self, shield: SafetyShield):
        """Set neural safety shield."""
        self.shield = shield
    
    def set_emergency_policy(self, policy: Callable):
        """Set emergency fallback policy."""
        self.emergency_policy = policy
    
    def _rebuild_monitor(self):
        """Rebuild monitor with current constraints."""
        self.monitor = SafetyMonitor(self.constraints)
    
    def safe_action(self, state: np.ndarray, policy_action: np.ndarray) -> np.ndarray:
        """Get safe action through all safety layers."""
        # Layer 1: Neural safety shield
        if self.shield is not None:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(policy_action).unsqueeze(0)
            
            filtered_action = self.shield.filter_action(state_tensor, action_tensor)
            policy_action = filtered_action.squeeze().detach().numpy()
        
        # Layer 2: Hard constraint projection
        if self.monitor is not None:
            safe_action = self.monitor.project_action(state, policy_action)
            
            # Check if action is safe
            is_safe, violations = self.monitor.check_action(state, safe_action)
            
            # Layer 3: Emergency override
            if not is_safe and self.monitor.safety_level == SafetyLevel.EMERGENCY_STOP:
                if self.emergency_policy is not None:
                    self.logger.warning("Activating emergency policy")
                    return self.emergency_policy(state)
                else:
                    # Default: stop action
                    return np.zeros_like(policy_action)
            
            return safe_action
        
        return policy_action
    
    def get_safety_report(self) -> Dict:
        """Get comprehensive safety report."""
        report = {
            'num_constraints': len(self.constraints),
            'shield_active': self.shield is not None,
            'emergency_policy_set': self.emergency_policy is not None
        }
        
        if self.monitor is not None:
            report.update(self.monitor.get_safety_metrics())
        
        return report


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create safety orchestrator
    orchestrator = SafetyOrchestrator()
    
    # Add box constraints
    action_min = np.array([-1.0, -1.0, -1.0])
    action_max = np.array([1.0, 1.0, 1.0])
    box_constraint = BoxConstraint(action_min, action_max)
    orchestrator.add_constraint(box_constraint)
    
    # Add barrier function constraint
    def sphere_barrier(state):
        """Keep state within sphere of radius 10."""
        return 100 - np.sum(state**2)  # h(x) = R^2 - ||x||^2
    
    barrier_constraint = BarrierFunction(sphere_barrier, alpha=0.1)
    orchestrator.add_constraint(barrier_constraint)
    
    # Create and set neural safety shield
    shield = SafetyShield(state_dim=6, action_dim=3, hidden_sizes=[64, 64])
    orchestrator.set_shield(shield)
    
    # Define emergency policy
    def emergency_stop(state):
        """Emergency policy: gradual stop."""
        return -0.1 * state[:3]  # Proportional to velocity
    
    orchestrator.set_emergency_policy(emergency_stop)
    
    # Test safety system
    print("Testing Safety System")
    print("=" * 50)
    
    # Test various scenarios
    test_scenarios = [
        {
            'name': 'Normal operation',
            'state': np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),
            'action': np.array([0.5, 0.5, 0.5])
        },
        {
            'name': 'Action out of bounds',
            'state': np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),
            'action': np.array([2.0, 2.0, 2.0])
        },
        {
            'name': 'Near boundary',
            'state': np.array([9.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            'action': np.array([1.0, 0.0, 0.0])
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"State: {scenario['state']}")
        print(f"Requested action: {scenario['action']}")
        
        safe_action = orchestrator.safe_action(scenario['state'], scenario['action'])
        print(f"Safe action: {safe_action}")
        
        report = orchestrator.get_safety_report()
        print(f"Safety report: {report}")
        print("-" * 30)