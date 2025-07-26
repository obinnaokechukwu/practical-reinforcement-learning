import numpy as np
from typing import Tuple, Dict, List
import time

class DynamicProgramming:
    """Implementation of core Dynamic Programming algorithms for solving MDPs."""
    
    def __init__(self, env, gamma: float = 0.99):
        """
        Initialize DP solver.
        
        Args:
            env: Environment with OpenAI Gym-like interface
            gamma: Discount factor
        """
        self.env = env
        self.gamma = gamma
        self.nS = env.nS
        self.nA = env.nA
        
    def policy_evaluation(self, policy: np.ndarray, theta: float = 1e-6) -> np.ndarray:
        """
        Evaluate a given policy using iterative policy evaluation.
        
        Args:
            policy: Policy to evaluate [nS x nA] stochastic or [nS] deterministic
            theta: Convergence threshold
            
        Returns:
            V: Value function for the policy
        """
        V = np.zeros(self.nS)
        
        # Convert deterministic policy to stochastic format
        if policy.ndim == 1:
            policy_stoch = np.zeros((self.nS, self.nA))
            for s in range(self.nS):
                policy_stoch[s, policy[s]] = 1.0
            policy = policy_stoch
        
        iteration = 0
        history = []
        
        while True:
            delta = 0
            old_V = V.copy()
            
            for s in range(self.nS):
                v = V[s]
                # Bellman expectation update
                V[s] = sum(policy[s][a] * 
                          sum(p * (r + self.gamma * V[s_])
                              for p, s_, r, _ in self.env.P[s][a])
                          for a in range(self.nA))
                
                delta = max(delta, abs(v - V[s]))
            
            iteration += 1
            history.append({
                'iteration': iteration,
                'delta': delta,
                'V': V.copy()
            })
            
            if delta < theta:
                break
                
        return V, history
    
    def policy_iteration(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find optimal policy using policy iteration.
        
        Args:
            theta: Convergence threshold for policy evaluation
            
        Returns:
            policy: Optimal policy [nS x nA]
            V: Optimal value function [nS]
            info: Dictionary with convergence information
        """
        # Initialize with uniform random policy
        policy = np.ones((self.nS, self.nA)) / self.nA
        
        iteration = 0
        history = []
        start_time = time.time()
        
        while True:
            # Policy Evaluation
            V, eval_history = self.policy_evaluation(policy, theta)
            
            # Policy Improvement
            policy_stable = True
            new_policy = np.zeros_like(policy)
            
            for s in range(self.nS):
                # Find best action
                q_values = np.zeros(self.nA)
                for a in range(self.nA):
                    q_values[a] = sum(p * (r + self.gamma * V[s_])
                                      for p, s_, r, _ in self.env.P[s][a])
                
                best_action = np.argmax(q_values)
                new_policy[s, best_action] = 1.0
                
                # Check if policy changed
                if not np.array_equal(policy[s], new_policy[s]):
                    policy_stable = False
            
            iteration += 1
            history.append({
                'iteration': iteration,
                'V': V.copy(),
                'policy': new_policy.copy(),
                'policy_stable': policy_stable,
                'eval_iterations': len(eval_history)
            })
            
            policy = new_policy
            
            if policy_stable:
                break
        
        elapsed_time = time.time() - start_time
        
        info = {
            'iterations': iteration,
            'time': elapsed_time,
            'history': history
        }
        
        return policy, V, info
    
    def value_iteration(self, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find optimal policy using value iteration.
        
        Args:
            theta: Convergence threshold
            
        Returns:
            policy: Optimal policy [nS x nA]
            V: Optimal value function [nS]
            info: Dictionary with convergence information
        """
        V = np.zeros(self.nS)
        
        iteration = 0
        history = []
        start_time = time.time()
        
        while True:
            delta = 0
            old_V = V.copy()
            
            for s in range(self.nS):
                v = V[s]
                # Bellman optimality update
                V[s] = max(sum(p * (r + self.gamma * V[s_])
                              for p, s_, r, _ in self.env.P[s][a])
                          for a in range(self.nA))
                
                delta = max(delta, abs(v - V[s]))
            
            iteration += 1
            history.append({
                'iteration': iteration,
                'delta': delta,
                'V': V.copy()
            })
            
            if delta < theta:
                break
        
        # Extract optimal policy
        policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            q_values = np.array([sum(p * (r + self.gamma * V[s_])
                                     for p, s_, r, _ in self.env.P[s][a])
                                 for a in range(self.nA)])
            best_action = np.argmax(q_values)
            policy[s, best_action] = 1.0
        
        elapsed_time = time.time() - start_time
        
        info = {
            'iterations': iteration,
            'time': elapsed_time,
            'history': history
        }
        
        return policy, V, info
    
    def modified_policy_iteration(self, k: int = 5, theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find optimal policy using modified policy iteration.
        
        Args:
            k: Number of evaluation steps per iteration
            theta: Convergence threshold
            
        Returns:
            policy: Optimal policy [nS x nA]
            V: Optimal value function [nS]
            info: Dictionary with convergence information
        """
        V = np.zeros(self.nS)
        policy = np.ones((self.nS, self.nA)) / self.nA
        
        iteration = 0
        history = []
        start_time = time.time()
        
        while True:
            # Partial Policy Evaluation (k steps)
            for _ in range(k):
                new_V = np.zeros_like(V)
                for s in range(self.nS):
                    new_V[s] = sum(policy[s][a] * 
                                  sum(p * (r + self.gamma * V[s_])
                                      for p, s_, r, _ in self.env.P[s][a])
                                  for a in range(self.nA))
                V = new_V
            
            # Policy Improvement
            policy_stable = True
            new_policy = np.zeros_like(policy)
            
            for s in range(self.nS):
                q_values = np.zeros(self.nA)
                for a in range(self.nA):
                    q_values[a] = sum(p * (r + self.gamma * V[s_])
                                      for p, s_, r, _ in self.env.P[s][a])
                
                best_action = np.argmax(q_values)
                new_policy[s, best_action] = 1.0
                
                if not np.array_equal(policy[s], new_policy[s]):
                    policy_stable = False
            
            iteration += 1
            history.append({
                'iteration': iteration,
                'V': V.copy(),
                'policy': new_policy.copy(),
                'policy_stable': policy_stable
            })
            
            policy = new_policy
            
            if policy_stable:
                break
        
        elapsed_time = time.time() - start_time
        
        info = {
            'iterations': iteration,
            'time': elapsed_time,
            'history': history,
            'k': k
        }
        
        return policy, V, info
    
    def prioritized_sweeping(self, theta: float = 1e-6, priority_threshold: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Find optimal policy using prioritized sweeping value iteration.
        
        Args:
            theta: Convergence threshold
            priority_threshold: Minimum priority to add state to queue
            
        Returns:
            policy: Optimal policy [nS x nA]
            V: Optimal value function [nS]
            info: Dictionary with convergence information
        """
        V = np.zeros(self.nS)
        
        # Build predecessor dictionary for efficiency
        predecessors = {s: set() for s in range(self.nS)}
        for s in range(self.nS):
            for a in range(self.nA):
                for p, s_, _, _ in self.env.P[s][a]:
                    if p > 0:
                        predecessors[s_].add((s, a))
        
        # Priority queue (using list for simplicity, could use heapq)
        priority_queue = []
        
        # Initialize with all states
        for s in range(self.nS):
            priority_queue.append(s)
        
        iteration = 0
        updates = 0
        history = []
        start_time = time.time()
        
        while priority_queue and iteration < 10000:  # Safety limit
            # Pop state with highest priority
            s = priority_queue.pop(0)
            
            # Bellman update
            old_v = V[s]
            V[s] = max(sum(p * (r + self.gamma * V[s_])
                          for p, s_, r, _ in self.env.P[s][a])
                      for a in range(self.nA))
            
            updates += 1
            error = abs(V[s] - old_v)
            
            # Update predecessors if error is significant
            if error > priority_threshold:
                for pred_s, pred_a in predecessors[s]:
                    # Calculate priority (Bellman error)
                    pred_v = V[pred_s]
                    pred_v_new = sum(p * (r + self.gamma * V[s_])
                                     for p, s_, r, _ in self.env.P[pred_s][pred_a])
                    pred_error = abs(pred_v_new - pred_v)
                    
                    if pred_error > priority_threshold and pred_s not in priority_queue:
                        priority_queue.append(pred_s)
            
            iteration += 1
            
            if iteration % 100 == 0:
                history.append({
                    'iteration': iteration,
                    'updates': updates,
                    'queue_size': len(priority_queue),
                    'V': V.copy()
                })
        
        # Extract optimal policy
        policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            q_values = np.array([sum(p * (r + self.gamma * V[s_])
                                     for p, s_, r, _ in self.env.P[s][a])
                                 for a in range(self.nA)])
            best_action = np.argmax(q_values)
            policy[s, best_action] = 1.0
        
        elapsed_time = time.time() - start_time
        
        info = {
            'iterations': iteration,
            'updates': updates,
            'time': elapsed_time,
            'history': history
        }
        
        return policy, V, info