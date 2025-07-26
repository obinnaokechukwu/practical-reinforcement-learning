"""
Constitutional reward wrapper for principled behavior.
"""

from typing import List, Dict, Optional, Callable
from agent.constitutional import ConstitutionalPrinciples


class ConstitutionalReward:
    """Wraps any reward function with constitutional compliance checking."""
    
    def __init__(self,
                 base_reward_fn: Callable,
                 constitution: List[str],
                 constitution_weight: float = 0.3,
                 min_compliance_threshold: float = 0.5):
        """
        Initialize constitutional reward wrapper.
        
        Args:
            base_reward_fn: Base reward function to wrap
            constitution: List of constitutional principles
            constitution_weight: Weight for constitutional compliance (0-1)
            min_compliance_threshold: Minimum compliance for non-zero reward
        """
        self.base_reward_fn = base_reward_fn
        self.constitutional_principles = ConstitutionalPrinciples(
            'coding', custom_principles=constitution
        )
        self.constitution_weight = constitution_weight
        self.min_compliance_threshold = min_compliance_threshold
    
    def __call__(self, problem: str, solution: str, **kwargs) -> float:
        """
        Compute reward with constitutional compliance.
        
        Args:
            problem: Problem description
            solution: Generated solution
            **kwargs: Additional arguments for base reward function
            
        Returns:
            Combined reward score
        """
        # Get base reward
        if hasattr(self.base_reward_fn, 'evaluate'):
            # Handle reward functions that return dictionaries
            base_result = self.base_reward_fn.evaluate(problem, solution, **kwargs)
            if isinstance(base_result, dict):
                base_reward = base_result.get('total', 0.0)
            else:
                base_reward = base_result
        else:
            # Simple callable
            base_reward = self.base_reward_fn(problem, solution, **kwargs)
        
        # Get constitutional compliance
        compliance_score = self.constitutional_principles.get_weighted_compliance_score(solution)
        
        # Check minimum threshold
        if compliance_score < self.min_compliance_threshold:
            # Heavily penalize severe violations
            return base_reward * 0.1
        
        # Combine rewards
        combined_reward = (
            (1 - self.constitution_weight) * base_reward +
            self.constitution_weight * compliance_score
        )
        
        return combined_reward
    
    def get_detailed_feedback(self, problem: str, solution: str, **kwargs) -> Dict:
        """
        Get detailed feedback including constitutional compliance.
        
        Args:
            problem: Problem description
            solution: Generated solution
            **kwargs: Additional arguments
            
        Returns:
            Detailed feedback dictionary
        """
        # Get base reward details
        if hasattr(self.base_reward_fn, 'evaluate'):
            base_result = self.base_reward_fn.evaluate(problem, solution, **kwargs)
            if isinstance(base_result, dict):
                base_reward = base_result.get('total', 0.0)
                base_details = base_result
            else:
                base_reward = base_result
                base_details = {'total': base_reward}
        else:
            base_reward = self.base_reward_fn(problem, solution, **kwargs)
            base_details = {'total': base_reward}
        
        # Get constitutional compliance details
        compliance_scores = self.constitutional_principles.evaluate_compliance(solution)
        overall_compliance = self.constitutional_principles.get_weighted_compliance_score(solution)
        violations = self.constitutional_principles.get_violations(solution)
        
        # Combine feedback
        feedback = {
            'base_reward': base_reward,
            'base_details': base_details,
            'constitutional_compliance': overall_compliance,
            'compliance_details': compliance_scores,
            'violations': violations,
            'combined_reward': self(problem, solution, **kwargs),
            'constitution_weight': self.constitution_weight,
            'feedback_text': self._generate_feedback_text(
                base_reward, overall_compliance, violations
            )
        }
        
        return feedback
    
    def _generate_feedback_text(self, 
                               base_reward: float,
                               compliance: float,
                               violations: List[str]) -> str:
        """Generate human-readable feedback."""
        feedback = []
        
        # Base performance
        if base_reward >= 0.8:
            feedback.append("✓ Excellent functional performance")
        elif base_reward >= 0.6:
            feedback.append("✓ Good functional performance")
        elif base_reward >= 0.4:
            feedback.append("⚠ Adequate functional performance")
        else:
            feedback.append("✗ Poor functional performance")
        
        # Constitutional compliance
        if compliance >= 0.8:
            feedback.append("✓ Excellent constitutional compliance")
        elif compliance >= 0.6:
            feedback.append("✓ Good constitutional compliance")
        elif compliance >= 0.4:
            feedback.append("⚠ Adequate constitutional compliance")
        else:
            feedback.append("✗ Poor constitutional compliance")
        
        # Violations
        if violations:
            feedback.append("\nPrinciples needing improvement:")
            for violation in violations[:3]:  # Show top 3
                feedback.append(f"  - {violation}")
        
        return '\n'.join(feedback)
    
    def update_constitution_weight(self, new_weight: float):
        """Update the weight given to constitutional compliance."""
        self.constitution_weight = max(0.0, min(1.0, new_weight))
    
    def add_principle(self, principle: str, weight: float = 1.0):
        """Add a new constitutional principle."""
        self.constitutional_principles.add_principle(principle, weight)
    
    def remove_principle(self, principle: str):
        """Remove a constitutional principle."""
        self.constitutional_principles.remove_principle(principle)


class AdaptiveConstitutionalReward(ConstitutionalReward):
    """Constitutional reward that adapts weights based on performance."""
    
    def __init__(self,
                 base_reward_fn: Callable,
                 constitution: List[str],
                 initial_constitution_weight: float = 0.3,
                 adaptation_rate: float = 0.01):
        """
        Initialize adaptive constitutional reward.
        
        Args:
            base_reward_fn: Base reward function
            constitution: Constitutional principles
            initial_constitution_weight: Starting weight
            adaptation_rate: Rate of weight adaptation
        """
        super().__init__(base_reward_fn, constitution, initial_constitution_weight)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
    
    def __call__(self, problem: str, solution: str, **kwargs) -> float:
        """Compute reward and adapt weights."""
        # Get component rewards
        base_reward = self._get_base_reward(problem, solution, **kwargs)
        compliance_score = self.constitutional_principles.get_weighted_compliance_score(solution)
        
        # Compute combined reward
        combined_reward = super().__call__(problem, solution, **kwargs)
        
        # Track performance
        self.performance_history.append({
            'base_reward': base_reward,
            'compliance': compliance_score,
            'combined': combined_reward
        })
        
        # Adapt weights
        self._adapt_weights()
        
        return combined_reward
    
    def _get_base_reward(self, problem: str, solution: str, **kwargs) -> float:
        """Extract base reward value."""
        if hasattr(self.base_reward_fn, 'evaluate'):
            result = self.base_reward_fn.evaluate(problem, solution, **kwargs)
            return result.get('total', 0.0) if isinstance(result, dict) else result
        else:
            return self.base_reward_fn(problem, solution, **kwargs)
    
    def _adapt_weights(self):
        """Adapt constitution weight based on recent performance."""
        if len(self.performance_history) < 10:
            return  # Need more data
        
        # Look at recent performance
        recent = self.performance_history[-10:]
        
        # Calculate correlations
        avg_base = sum(p['base_reward'] for p in recent) / len(recent)
        avg_compliance = sum(p['compliance'] for p in recent) / len(recent)
        
        # If base performance is low but compliance is high,
        # reduce constitution weight
        if avg_base < 0.5 and avg_compliance > 0.7:
            self.constitution_weight *= (1 - self.adaptation_rate)
        
        # If base performance is high but compliance is low,
        # increase constitution weight
        elif avg_base > 0.7 and avg_compliance < 0.5:
            self.constitution_weight *= (1 + self.adaptation_rate)
        
        # Keep weight in reasonable range
        self.constitution_weight = max(0.1, min(0.5, self.constitution_weight))