"""
Dynamic Programming Solver for GridWorld Environments

This module implements the core DP algorithms:
- Policy Iteration
- Value Iteration  
- Modified Policy Iteration
- Prioritized Sweeping

Each algorithm demonstrates different aspects of planning with perfect models.
"""

import numpy as np
import heapq
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class DPSolver:
    """
    Dynamic Programming solver implementing multiple algorithms.
    
    This class showcases the relationships between different DP algorithms
    and provides instrumentation for understanding their behavior.
    """
    
    def __init__(self, env, gamma=0.99):
        """
        Initialize the solver.
        
        Args:
            env: GridWorld environment instance
            gamma: Discount factor for future rewards
        """
        self.env = env
        self.gamma = gamma
        self.iteration_history = []
        
    def policy_evaluation(self, policy, theta=1e-6, max_iterations=1000):
        """
        Evaluate a given policy using iterative policy evaluation.
        
        This solves the prediction problem: given π, compute V^π.
        The algorithm repeatedly applies the Bellman expectation equation
        until the value function converges.
        
        Args:
            policy: Policy to evaluate (nS x nA array or nS array)
            theta: Convergence threshold
            max_iterations: Maximum iterations before stopping
            
        Returns:
            V: Value function under the policy
            iteration_count: Number of iterations until convergence
        """
        V = np.zeros(self.env.nS)
        iteration_count = 0
        
        for i in range(max_iterations):
            delta = 0
            v_old = V.copy()
            
            for s in range(self.env.nS):
                # Store old value for convergence check
                v = V[s]
                
                # Apply Bellman expectation equation
                # V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[R + γV^π(s')]
                new_value = 0
                for a in range(self.env.nA):
                    # Get probability of taking action a in state s under policy π
                    if policy.ndim == 2:  # Stochastic policy
                        action_prob = policy[s, a]
                    else:  # Deterministic policy
                        action_prob = 1.0 if policy[s] == a else 0.0
                    
                    # Expected value of taking action a
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        # Note: (1 - done) ensures terminal states have V=0
                        new_value += action_prob * prob * (
                            reward + self.gamma * V[next_state] * (1 - done)
                        )
                
                V[s] = new_value
                delta = max(delta, abs(v - V[s]))
            
            iteration_count += 1
            
            # Track convergence history for analysis
            self.iteration_history.append({
                'iteration': i,
                'delta': delta,
                'V': V.copy(),
                'mean_value': np.mean(V),
                'max_value': np.max(V),
                'min_value': np.min(V)
            })
            
            if delta < theta:
                break
        
        return V, iteration_count
    
    def policy_improvement(self, V):
        """
        Improve policy given value function.
        
        This performs the control step: given V^π, find π' where V^π' ≥ V^π.
        The improved policy acts greedily with respect to the value function.
        
        Args:
            V: Value function to improve upon
            
        Returns:
            policy: Improved policy (deterministic)
        """
        policy = np.zeros((self.env.nS, self.env.nA))
        
        for s in range(self.env.nS):
            # Compute Q^π(s,a) for all actions
            Q_values = np.zeros(self.env.nA)
            
            for a in range(self.env.nA):
                # Q(s,a) = Σ_s' P(s'|s,a)[R + γV(s')]
                for prob, next_state, reward, done in self.env.P[s][a]:
                    Q_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))
            
            # Choose action that maximizes Q-value
            # Using argmax ensures deterministic tie-breaking
            best_action = np.argmax(Q_values)
            policy[s, best_action] = 1.0
        
        return policy
    
    def policy_iteration(self, theta=1e-6):
        """
        Find optimal policy using Policy Iteration.
        
        Policy Iteration alternates between:
        1. Policy Evaluation: Compute V^π for current policy
        2. Policy Improvement: Find better policy using V^π
        
        This continues until the policy no longer changes.
        
        Args:
            theta: Value convergence threshold for policy evaluation
            
        Returns:
            policy: Optimal policy
            V: Optimal value function
            iterations: Number of policy improvement steps
        """
        # Initialize with uniform random policy
        policy = np.ones((self.env.nS, self.env.nA)) / self.env.nA
        
        iteration = 0
        policy_stable = False
        
        while not policy_stable:
            # Policy Evaluation: Find value of current policy
            V, eval_iterations = self.policy_evaluation(policy, theta)
            
            # Policy Improvement: Find better policy
            new_policy = self.policy_improvement(V)
            
            # Check if policy has changed
            policy_stable = np.array_equal(policy, new_policy)
            policy = new_policy
            
            iteration += 1
            print(f"Policy Iteration {iteration}: Evaluation took {eval_iterations} iterations")
            
            if iteration > 100:  # Safety check
                print("Warning: Policy iteration did not converge in 100 iterations")
                break
        
        return policy, V, iteration
    
    def value_iteration(self, theta=1e-6, max_iterations=1000):
        """
        Find optimal policy using Value Iteration.
        
        Value Iteration directly computes V* by repeatedly applying
        the Bellman optimality equation:
        V*(s) = max_a Σ_s' P(s'|s,a)[R + γV*(s')]
        
        This combines policy evaluation and improvement into one step.
        
        Args:
            theta: Convergence threshold
            max_iterations: Maximum iterations
            
        Returns:
            policy: Optimal policy
            V: Optimal value function
            iterations: Number of iterations until convergence
        """
        V = np.zeros(self.env.nS)
        iteration_count = 0
        
        for i in range(max_iterations):
            delta = 0
            
            for s in range(self.env.nS):
                v = V[s]
                
                # Apply Bellman optimality equation
                # V*(s) = max_a Σ_s' P(s'|s,a)[R + γV*(s')]
                Q_values = np.zeros(self.env.nA)
                
                for a in range(self.env.nA):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        Q_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))
                
                V[s] = np.max(Q_values)
                delta = max(delta, abs(v - V[s]))
            
            iteration_count += 1
            
            # Track progress
            if i % 10 == 0:
                print(f"Value Iteration {i}: delta = {delta:.6f}")
            
            if delta < theta:
                break
        
        # Extract optimal policy from optimal values
        policy = self.policy_improvement(V)
        
        return policy, V, iteration_count
    
    def modified_policy_iteration(self, k=5, theta=1e-6):
        """
        Modified Policy Iteration: interpolates between VI and PI.
        
        Instead of fully evaluating each policy, we do k steps
        of value iteration for evaluation. This provides a smooth
        tradeoff between the two algorithms:
        - k=1: Equivalent to Value Iteration
        - k=∞: Equivalent to Policy Iteration
        - 1<k<∞: Hybrid approach
        
        Args:
            k: Number of evaluation steps per policy
            theta: Convergence threshold
            
        Returns:
            policy: Optimal policy
            V: Optimal value function
            iterations: Number of policy updates
        """
        policy = np.ones((self.env.nS, self.env.nA)) / self.env.nA
        V = np.zeros(self.env.nS)
        
        iteration = 0
        policy_stable = False
        
        while not policy_stable:
            # Partial Policy Evaluation: k steps of value iteration
            for _ in range(k):
                for s in range(self.env.nS):
                    new_value = 0
                    for a in range(self.env.nA):
                        action_prob = policy[s, a]
                        for prob, next_state, reward, done in self.env.P[s][a]:
                            new_value += action_prob * prob * (
                                reward + self.gamma * V[next_state] * (1 - done)
                            )
                    V[s] = new_value
            
            # Policy Improvement
            new_policy = self.policy_improvement(V)
            
            # Check convergence
            policy_stable = np.array_equal(policy, new_policy)
            policy = new_policy
            
            iteration += 1
            
            if iteration > 100:
                break
        
        return policy, V, iteration
    
    def prioritized_sweeping_value_iteration(self, theta=1e-6, priority_theta=1e-5):
        """
        Value Iteration with prioritized sweeping.
        
        Instead of updating all states equally, we focus computation
        on states with large Bellman errors. This dramatically improves
        efficiency for large state spaces where only some states matter.
        
        The algorithm maintains a priority queue of states sorted by
        their Bellman error, updating high-error states first.
        
        Args:
            theta: Convergence threshold
            priority_theta: Minimum error to add state to priority queue
            
        Returns:
            policy: Optimal policy
            V: Optimal value function
            iterations: Number of state updates performed
        """
        V = np.zeros(self.env.nS)
        
        # Priority queue stores (-priority, state) for max-heap behavior
        pq = []
        
        # Initialize priorities based on Bellman error
        for s in range(self.env.nS):
            # Compute initial Bellman error
            Q_values = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    Q_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))
            
            error = abs(np.max(Q_values) - V[s])
            if error > priority_theta:
                heapq.heappush(pq, (-error, s))
        
        # Build predecessor graph for efficient updates
        # predecessors[s] = set of states that can transition to s
        predecessors = {s: set() for s in range(self.env.nS)}
        for s in range(self.env.nS):
            for a in range(self.env.nA):
                for prob, next_state, _, _ in self.env.P[s][a]:
                    if prob > 0 and next_state != s:
                        predecessors[next_state].add(s)
        
        iteration = 0
        while pq and iteration < 10000:
            # Pop state with highest priority (largest error)
            _, s = heapq.heappop(pq)
            
            # Update state value
            old_v = V[s]
            Q_values = np.zeros(self.env.nA)
            for a in range(self.env.nA):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    Q_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))
            V[s] = np.max(Q_values)
            
            # If value changed significantly, update predecessors
            if abs(V[s] - old_v) > priority_theta:
                for pred in predecessors[s]:
                    # Compute new priority for predecessor
                    Q_values = np.zeros(self.env.nA)
                    for a in range(self.env.nA):
                        for prob, next_state, reward, done in self.env.P[pred][a]:
                            Q_values[a] += prob * (reward + self.gamma * V[next_state] * (1 - done))
                    
                    error = abs(np.max(Q_values) - V[pred])
                    if error > priority_theta:
                        heapq.heappush(pq, (-error, pred))
            
            iteration += 1
        
        # Extract policy from final values
        policy = self.policy_improvement(V)
        
        return policy, V, iteration
    
    def analyze_algorithm_performance(self, algorithms=['PI', 'VI', 'MPI']):
        """
        Compare performance of different DP algorithms.
        
        This method runs multiple algorithms on the same environment
        and collects statistics for comparison.
        
        Args:
            algorithms: List of algorithm names to compare
            
        Returns:
            results: Dict mapping algorithm names to performance metrics
        """
        results = {}
        
        for alg in algorithms:
            if alg == 'PI':
                policy, V, iters = self.policy_iteration()
                results['Policy Iteration'] = {
                    'policy': policy,
                    'V': V,
                    'iterations': iters,
                    'mean_value': np.mean(V)
                }
            elif alg == 'VI':
                policy, V, iters = self.value_iteration()
                results['Value Iteration'] = {
                    'policy': policy,
                    'V': V,
                    'iterations': iters,
                    'mean_value': np.mean(V)
                }
            elif alg == 'MPI':
                policy, V, iters = self.modified_policy_iteration(k=5)
                results['Modified PI (k=5)'] = {
                    'policy': policy,
                    'V': V,
                    'iterations': iters,
                    'mean_value': np.mean(V)
                }
            elif alg == 'PS':
                policy, V, iters = self.prioritized_sweeping_value_iteration()
                results['Prioritized Sweeping'] = {
                    'policy': policy,
                    'V': V,
                    'iterations': iters,
                    'mean_value': np.mean(V)
                }
        
        return results


def verify_bellman_consistency(env, policy, V, gamma=0.99):
    """
    Verify that a policy and value function satisfy the Bellman equations.
    
    This is useful for debugging DP implementations.
    
    Args:
        env: GridWorld environment
        policy: Policy to verify
        V: Value function to verify
        gamma: Discount factor
        
    Returns:
        is_consistent: Whether the solution satisfies Bellman equations
        max_error: Maximum Bellman error across all states
    """
    max_error = 0
    
    for s in range(env.nS):
        # Compute expected value under policy
        v_pi = 0
        for a in range(env.nA):
            if policy.ndim == 2:
                action_prob = policy[s, a]
            else:
                action_prob = 1.0 if policy[s] == a else 0.0
            
            for prob, next_state, reward, done in env.P[s][a]:
                v_pi += action_prob * prob * (
                    reward + gamma * V[next_state] * (1 - done)
                )
        
        error = abs(V[s] - v_pi)
        max_error = max(max_error, error)
    
    is_consistent = max_error < 1e-6
    
    if is_consistent:
        print("✓ Solution satisfies Bellman consistency")
    else:
        print(f"✗ Maximum Bellman error: {max_error}")
    
    return is_consistent, max_error