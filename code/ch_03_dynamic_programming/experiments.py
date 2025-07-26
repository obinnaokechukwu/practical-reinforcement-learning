"""
Educational Experiments for Understanding Dynamic Programming

This module contains experiments designed to build intuition about:
- How discount factors affect optimal policies
- Convergence properties of different algorithms
- The relationship between values and policies
- Practical considerations for DP implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld, create_example_environments
from dp_solver import DPSolver, verify_bellman_consistency


def experiment_discount_factor_effects():
    """
    Experiment 1: How Discount Factor Affects Optimal Policy
    
    This experiment shows how γ influences the agent's planning horizon.
    Low γ values make the agent myopic (focusing on immediate rewards),
    while high γ values encourage long-term planning.
    """
    print("Experiment 1: Effect of Discount Factor on Optimal Policy")
    print("-" * 60)
    
    # Create environment with both immediate and delayed rewards
    env = GridWorld(
        height=4, width=4,
        start=(0, 0),
        terminals=[(3, 3)],
        rewards={(3, 3): 10, (1, 2): 5}  # Goal and intermediate reward
    )
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    gammas = [0.1, 0.5, 0.9, 0.95, 0.99, 0.999]
    
    for idx, gamma in enumerate(gammas):
        solver = DPSolver(env, gamma=gamma)
        policy, V, _ = solver.value_iteration()
        
        ax = axes[idx]
        
        # Create value grid for visualization
        value_grid = np.full((env.height, env.width), np.nan)
        for i, state in enumerate(env.states):
            value_grid[state] = V[i]
        
        # Plot value heatmap
        im = ax.imshow(value_grid, cmap='RdYlGn', aspect='equal')
        ax.set_title(f'γ = {gamma}', fontsize=14, fontweight='bold')
        
        # Add policy arrows
        for i, state in enumerate(env.states):
            if state not in env.terminals:
                action = np.argmax(policy[i])
                dy, dx = env.actions[action]
                y, x = state
                ax.arrow(x, y, dx*0.3, dy*0.3, head_width=0.1, 
                        head_length=0.05, fc='black', ec='black')
        
        # Mark special states
        ax.plot(0, 0, 'bo', markersize=10, label='Start')  # Start
        ax.plot(3, 3, 'r*', markersize=15, label='Goal')   # Goal
        ax.plot(2, 1, 'g^', markersize=10, label='Bonus')  # Intermediate reward
        
        # Display values in cells
        for i, state in enumerate(env.states):
            y, x = state
            ax.text(x, y, f'{V[i]:.1f}', ha='center', va='center',
                   fontsize=9, color='white', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, linewidth=1, alpha=0.3)
    
    # Add legend to first subplot
    axes[0].legend(loc='upper right', fontsize=8)
    
    plt.suptitle('How Discount Factor Affects Policy and Values', fontsize=16)
    plt.tight_layout()
    plt.savefig('discount_factor_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\nKey Observations:")
    print("- Low γ (0.1): Agent is myopic, barely sees beyond immediate rewards")
    print("- Medium γ (0.5-0.9): Agent balances immediate and future rewards")
    print("- High γ (0.99+): Agent plans for long-term, values compound over distance")
    print("- Policy changes: Notice how the path to goal changes with γ!")


def experiment_algorithm_convergence():
    """
    Experiment 2: Convergence Properties of DP Algorithms
    
    This experiment compares how quickly different algorithms converge
    and visualizes their convergence patterns.
    """
    print("\nExperiment 2: Algorithm Convergence Comparison")
    print("-" * 60)
    
    # Use a larger environment to see convergence differences
    env = GridWorld(height=10, width=10, start=(0, 0), terminals=[(9, 9)])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Value Iteration convergence
    print("Running Value Iteration...")
    V = np.zeros(env.nS)
    vi_deltas = []
    vi_values = []
    
    for i in range(200):
        V_old = V.copy()
        for s in range(env.nS):
            Q_values = []
            for a in range(env.nA):
                q_val = sum(p * (r + 0.99 * V[s_] * (1-d))
                           for p, s_, r, d in env.P[s][a])
                Q_values.append(q_val)
            V[s] = max(Q_values)
        
        delta = np.max(np.abs(V - V_old))
        vi_deltas.append(delta)
        vi_values.append(V.copy())
        
        if delta < 1e-6:
            break
    
    # Plot VI convergence
    ax1.semilogy(vi_deltas, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max Value Change (log scale)')
    ax1.set_title('Value Iteration Convergence')
    ax1.grid(True, alpha=0.3)
    
    # Track value evolution for specific states
    tracked_states = [0, env.nS // 2, env.nS - 1]
    colors = ['red', 'green', 'blue']
    
    for idx, state in enumerate(tracked_states):
        values = [v[state] for v in vi_values[:50]]  # First 50 iterations
        ax2.plot(values, color=colors[idx], linewidth=2,
                label=f'State {state}')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('State Value')
    ax2.set_title('Value Evolution During VI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Policy Iteration comparison
    print("Running Policy Iteration...")
    solver = DPSolver(env, gamma=0.99)
    
    # Track evaluation iterations for each policy improvement
    pi_eval_iters = []
    
    # Custom PI to track iterations
    policy = np.ones((env.nS, env.nA)) / env.nA
    for i in range(20):
        V, eval_iters = solver.policy_evaluation(policy)
        pi_eval_iters.append(eval_iters)
        
        new_policy = solver.policy_improvement(V)
        if np.array_equal(policy, new_policy):
            break
        policy = new_policy
    
    # Plot PI evaluation iterations
    ax3.bar(range(len(pi_eval_iters)), pi_eval_iters, color='green', alpha=0.7)
    ax3.set_xlabel('Policy Improvement Step')
    ax3.set_ylabel('Evaluation Iterations')
    ax3.set_title('Policy Iteration: Evaluation Cost per Step')
    ax3.grid(True, alpha=0.3)
    
    # Algorithm comparison
    results = {
        'Value Iteration': len(vi_deltas),
        'Policy Iteration': len(pi_eval_iters),
        'Modified PI (k=5)': 0,  # Placeholder
        'Modified PI (k=10)': 0   # Placeholder
    }
    
    # Run Modified PI variants
    for k in [5, 10]:
        policy, V, iters = solver.modified_policy_iteration(k=k)
        results[f'Modified PI (k={k})'] = iters
    
    # Bar chart comparison
    ax4.bar(range(len(results)), list(results.values()), 
            tick_label=list(results.keys()), color=['blue', 'green', 'orange', 'red'])
    ax4.set_ylabel('Iterations to Convergence')
    ax4.set_title('Algorithm Efficiency Comparison')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Convergence Analysis of DP Algorithms', fontsize=16)
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nKey Observations:")
    print("- Value Iteration: Many iterations, but each is cheap")
    print("- Policy Iteration: Few iterations, but each is expensive")
    print("- Modified PI: Provides a tunable middle ground")
    print("- Convergence is geometric (exponential decay)")


def experiment_stochastic_environments():
    """
    Experiment 3: Planning in Stochastic Environments
    
    This experiment shows how optimal policies change when
    transitions are uncertain (slippery gridworld).
    """
    print("\nExperiment 3: Deterministic vs Stochastic Planning")
    print("-" * 60)
    
    # Create two environments: deterministic and stochastic
    env_det = GridWorld(
        height=4, width=6,
        start=(3, 0),
        terminals=[(3, 5)],
        rewards={(3, 5): 10, (2, 1): -10, (2, 2): -10, (2, 3): -10, (2, 4): -10}
    )
    
    env_stoch = GridWorld(
        height=4, width=6,
        start=(3, 0),
        terminals=[(3, 5)],
        slip_prob=0.3,
        rewards={(3, 5): 10, (2, 1): -10, (2, 2): -10, (2, 3): -10, (2, 4): -10}
    )
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Solve both environments
    solver_det = DPSolver(env_det, gamma=0.99)
    policy_det, V_det, _ = solver_det.value_iteration()
    
    solver_stoch = DPSolver(env_stoch, gamma=0.99)
    policy_stoch, V_stoch, _ = solver_stoch.value_iteration()
    
    # Visualize deterministic case
    value_grid_det = np.full((env_det.height, env_det.width), np.nan)
    for idx, state in enumerate(env_det.states):
        value_grid_det[state] = V_det[idx]
    
    im1 = ax1.imshow(value_grid_det, cmap='RdYlGn', aspect='equal')
    ax1.set_title('Deterministic: Values', fontsize=14)
    
    # Add policy arrows and values
    for idx, state in enumerate(env_det.states):
        y, x = state
        ax1.text(x, y, f'{V_det[idx]:.1f}', ha='center', va='center',
                fontsize=10, fontweight='bold')
        
        if state not in env_det.terminals:
            action = np.argmax(policy_det[idx])
            dy, dx = env_det.actions[action]
            ax2.arrow(x, y, dx*0.3, dy*0.3, head_width=0.15, 
                     head_length=0.1, fc='blue', ec='blue', linewidth=2)
    
    # Mark hazards
    for x in range(1, 5):
        ax1.add_patch(plt.Rectangle((x-0.5, 1.5), 1, 1, 
                                   facecolor='red', alpha=0.3))
        ax2.add_patch(plt.Rectangle((x-0.5, 1.5), 1, 1, 
                                   facecolor='red', alpha=0.3))
        ax2.text(x, 2, 'CLIFF', ha='center', va='center',
                fontsize=8, fontweight='bold', color='red')
    
    ax2.set_title('Deterministic: Policy', fontsize=14)
    
    # Visualize stochastic case
    value_grid_stoch = np.full((env_stoch.height, env_stoch.width), np.nan)
    for idx, state in enumerate(env_stoch.states):
        value_grid_stoch[state] = V_stoch[idx]
    
    im3 = ax3.imshow(value_grid_stoch, cmap='RdYlGn', aspect='equal')
    ax3.set_title('Stochastic (30% slip): Values', fontsize=14)
    
    # Add policy arrows and values
    for idx, state in enumerate(env_stoch.states):
        y, x = state
        ax3.text(x, y, f'{V_stoch[idx]:.1f}', ha='center', va='center',
                fontsize=10, fontweight='bold')
        
        if state not in env_stoch.terminals:
            action = np.argmax(policy_stoch[idx])
            dy, dx = env_stoch.actions[action]
            ax4.arrow(x, y, dx*0.3, dy*0.3, head_width=0.15, 
                     head_length=0.1, fc='blue', ec='blue', linewidth=2)
    
    # Mark hazards
    for x in range(1, 5):
        ax3.add_patch(plt.Rectangle((x-0.5, 1.5), 1, 1, 
                                   facecolor='red', alpha=0.3))
        ax4.add_patch(plt.Rectangle((x-0.5, 1.5), 1, 1, 
                                   facecolor='red', alpha=0.3))
        ax4.text(x, 2, 'CLIFF', ha='center', va='center',
                fontsize=8, fontweight='bold', color='red')
    
    ax4.set_title('Stochastic (30% slip): Policy', fontsize=14)
    
    # Configure all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(3.5, -0.5)
        ax.set_xticks(range(6))
        ax.set_yticks(range(4))
        ax.grid(True, linewidth=1, alpha=0.3)
        
        # Mark start and goal
        ax.plot(0, 3, 'go', markersize=12)
        ax.plot(5, 3, 'r*', markersize=15)
    
    plt.suptitle('Optimal Policies: Deterministic vs Stochastic', fontsize=16)
    plt.tight_layout()
    plt.savefig('stochastic_planning.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nKey Observations:")
    print("- Deterministic: Takes shortest path along the cliff edge")
    print("- Stochastic: Takes safer path away from cliff")
    print("- Values are lower in stochastic case (risk of falling)")
    print("- This demonstrates risk-aware planning under uncertainty")


def experiment_prioritized_sweeping():
    """
    Experiment 4: Efficiency of Prioritized Sweeping
    
    This experiment demonstrates how prioritized sweeping can
    dramatically reduce computation by focusing on important states.
    """
    print("\nExperiment 4: Prioritized Sweeping Efficiency")
    print("-" * 60)
    
    # Create a maze environment where most states don't matter initially
    env = GridWorld(
        height=8, width=12,
        start=(7, 0),
        terminals=[(0, 11)],
        obstacles=[
            # Create maze walls
            (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
            (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
            (1, 6), (2, 6), (3, 6), (4, 6),
            (3, 8), (4, 8), (5, 8), (6, 8),
            (1, 10), (2, 10), (3, 10), (4, 10), (5, 10)
        ],
        rewards={(0, 11): 100}
    )
    
    # Track which states get updated
    update_counts_vi = np.zeros(env.nS)
    update_counts_ps = np.zeros(env.nS)
    
    # Standard Value Iteration
    print("Running standard Value Iteration...")
    V_vi = np.zeros(env.nS)
    for iteration in range(100):
        for s in range(env.nS):
            old_v = V_vi[s]
            Q_values = []
            for a in range(env.nA):
                q_val = sum(p * (r + 0.99 * V_vi[s_] * (1-d))
                           for p, s_, r, d in env.P[s][a])
                Q_values.append(q_val)
            V_vi[s] = max(Q_values)
            if abs(V_vi[s] - old_v) > 1e-10:
                update_counts_vi[s] += 1
    
    # Prioritized Sweeping
    print("Running Prioritized Sweeping...")
    solver = DPSolver(env, gamma=0.99)
    
    # Custom implementation to track updates
    V_ps = np.zeros(env.nS)
    pq = []
    
    # Initialize with high-value states
    for s in range(env.nS):
        if env.states[s] in env.terminals:
            for a in range(env.nA):
                for p, s_next, r, d in env.P[s][a]:
                    if p > 0 and r > 0:
                        import heapq
                        heapq.heappush(pq, (-abs(r), s))
                        break
    
    # Run prioritized updates
    iteration = 0
    while pq and iteration < 1000:
        _, s = heapq.heappop(pq)
        old_v = V_ps[s]
        
        Q_values = []
        for a in range(env.nA):
            q_val = sum(p * (r + 0.99 * V_ps[s_] * (1-d))
                       for p, s_, r, d in env.P[s][a])
            Q_values.append(q_val)
        V_ps[s] = max(Q_values)
        
        if abs(V_ps[s] - old_v) > 1e-6:
            update_counts_ps[s] += 1
            
            # Add predecessors
            for s_pred in range(env.nS):
                for a in range(env.nA):
                    for p, s_next, r, d in env.P[s_pred][a]:
                        if s_next == s and p > 0:
                            error = abs(max(sum(p2 * (r2 + 0.99 * V_ps[s2] * (1-d2))
                                              for p2, s2, r2, d2 in env.P[s_pred][a2])
                                          for a2 in range(env.nA)) - V_ps[s_pred])
                            if error > 1e-6:
                                heapq.heappush(pq, (-error, s_pred))
        iteration += 1
    
    # Visualize update patterns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create update heatmaps
    update_grid_vi = np.zeros((env.height, env.width))
    update_grid_ps = np.zeros((env.height, env.width))
    value_grid = np.full((env.height, env.width), np.nan)
    
    for idx, state in enumerate(env.states):
        update_grid_vi[state] = update_counts_vi[idx]
        update_grid_ps[state] = update_counts_ps[idx]
        value_grid[state] = V_ps[idx]
    
    # Plot VI updates
    im1 = ax1.imshow(update_grid_vi, cmap='hot', aspect='equal')
    ax1.set_title('Value Iteration: Update Counts', fontsize=14)
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Plot PS updates
    im2 = ax2.imshow(update_grid_ps, cmap='hot', aspect='equal')
    ax2.set_title('Prioritized Sweeping: Update Counts', fontsize=14)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # Plot final values
    masked_values = np.ma.masked_invalid(value_grid)
    im3 = ax3.imshow(masked_values, cmap='RdYlGn', aspect='equal')
    ax3.set_title('Final Values', fontsize=14)
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # Mark obstacles
    for ax in [ax1, ax2, ax3]:
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1,
                                     facecolor='black'))
        # Mark start and goal
        ax.plot(0, 7, 'bo', markersize=10)
        ax.plot(11, 0, 'r*', markersize=15)
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(7.5, -0.5)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Prioritized Sweeping vs Standard Value Iteration', fontsize=16)
    plt.tight_layout()
    plt.savefig('prioritized_sweeping.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    total_updates_vi = np.sum(update_counts_vi)
    total_updates_ps = np.sum(update_counts_ps)
    
    print(f"\nTotal updates:")
    print(f"- Value Iteration: {total_updates_vi:.0f}")
    print(f"- Prioritized Sweeping: {total_updates_ps:.0f}")
    print(f"- Speedup: {total_updates_vi/total_updates_ps:.1f}x")
    print("\nKey Observations:")
    print("- PS focuses updates near goal and along optimal paths")
    print("- VI wastes computation updating irrelevant states")
    print("- PS converges faster by propagating information efficiently")


def run_all_experiments():
    """Run all educational experiments."""
    experiments = [
        experiment_discount_factor_effects,
        experiment_algorithm_convergence,
        experiment_stochastic_environments,
        experiment_prioritized_sweeping
    ]
    
    for exp in experiments:
        exp()
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run experiments
    run_all_experiments()
    
    # Test Bellman consistency
    print("Verifying Implementation Correctness...")
    env = create_example_environments()['simple']
    solver = DPSolver(env)
    policy, V, _ = solver.value_iteration()
    verify_bellman_consistency(env, policy, V)