"""
Main demonstration script for Dynamic Programming algorithms.

This script shows how to use the gridworld environment and DP solvers
to find optimal policies for various scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld, create_example_environments
from dp_solver import DPSolver, verify_bellman_consistency


def demonstrate_simple_gridworld():
    """Basic demonstration of solving a simple gridworld."""
    print("="*60)
    print("Simple GridWorld Demonstration")
    print("="*60)
    
    # Create a basic 4x4 gridworld
    env = GridWorld(
        height=4, 
        width=4,
        start=(0, 0),
        terminals=[(3, 3)],
        rewards={(3, 3): 10}
    )
    
    # Create solver
    solver = DPSolver(env, gamma=0.99)
    
    # Solve using different algorithms
    print("\n1. Policy Iteration:")
    pi_policy, pi_V, pi_iters = solver.policy_iteration()
    print(f"   Converged in {pi_iters} iterations")
    print(f"   Optimal value at start: {pi_V[0]:.2f}")
    
    print("\n2. Value Iteration:")
    vi_policy, vi_V, vi_iters = solver.value_iteration()
    print(f"   Converged in {vi_iters} iterations")
    print(f"   Optimal value at start: {vi_V[0]:.2f}")
    
    # Verify both algorithms found same solution
    if np.allclose(pi_V, vi_V):
        print("\n✓ Both algorithms found the same optimal values!")
    
    # Visualize the solution
    fig1, ax1 = env.render_values(vi_V, title="Optimal State Values")
    fig2, ax2 = env.render_policy(vi_policy, vi_V, title="Optimal Policy")
    
    plt.show()


def demonstrate_all_environments():
    """Show solutions for all example environments."""
    print("\n" + "="*60)
    print("Solving All Example Environments")
    print("="*60)
    
    environments = create_example_environments()
    
    # Create figure for all environments
    fig = plt.figure(figsize=(20, 12))
    
    for idx, (name, env) in enumerate(environments.items()):
        print(f"\nSolving {name} environment...")
        
        # Solve environment
        solver = DPSolver(env, gamma=0.99)
        policy, V, iters = solver.value_iteration()
        print(f"  States: {env.nS}, Actions: {env.nA}")
        print(f"  Converged in {iters} iterations")
        print(f"  Value range: [{np.min(V):.2f}, {np.max(V):.2f}]")
        
        # Create value grid for visualization
        value_grid = np.full((env.height, env.width), np.nan)
        for i, state in enumerate(env.states):
            value_grid[state] = V[i]
        
        # Plot values
        ax = fig.add_subplot(2, 3, idx + 1)
        masked_grid = np.ma.masked_invalid(value_grid)
        im = ax.imshow(masked_grid, cmap='RdYlGn', aspect='equal')
        
        # Add policy arrows
        for i, state in enumerate(env.states):
            if state not in env.terminals:
                action = np.argmax(policy[i])
                dy, dx = env.actions[action]
                y, x = state
                ax.arrow(x, y, dx*0.3, dy*0.3, head_width=0.1, 
                        head_length=0.05, fc='darkblue', ec='darkblue')
        
        # Mark special cells
        for obs in env.obstacles:
            ax.add_patch(plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1,
                                     facecolor='black'))
        
        # Mark start and terminals
        y_start, x_start = env.start
        ax.plot(x_start, y_start, 'go', markersize=8)
        
        for term in env.terminals:
            y_term, x_term = term
            ax.plot(x_term, y_term, 'r*', markersize=12)
        
        ax.set_title(f'{name.title()} Environment', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, env.width - 0.5)
        ax.set_ylim(env.height - 0.5, -0.5)
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Optimal Policies for Different Environments', fontsize=16)
    plt.tight_layout()
    plt.show()


def interactive_parameter_exploration():
    """Interactive exploration of algorithm parameters."""
    print("\n" + "="*60)
    print("Parameter Exploration")
    print("="*60)
    
    # Create test environment
    env = GridWorld(
        height=5,
        width=5,
        start=(0, 0),
        terminals=[(4, 4)],
        obstacles=[(2, 2)],
        rewards={(4, 4): 10, (1, 3): 5}
    )
    
    # Test different discount factors
    gammas = [0.1, 0.5, 0.9, 0.99]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, gamma in enumerate(gammas):
        solver = DPSolver(env, gamma=gamma)
        policy, V, _ = solver.value_iteration()
        
        # Create value grid
        value_grid = np.full((env.height, env.width), np.nan)
        for i, state in enumerate(env.states):
            value_grid[state] = V[i]
        
        ax = axes[idx]
        masked_grid = np.ma.masked_invalid(value_grid)
        im = ax.imshow(masked_grid, cmap='RdYlGn', aspect='equal')
        
        # Add values and arrows
        for i, state in enumerate(env.states):
            y, x = state
            ax.text(x, y, f'{V[i]:.1f}', ha='center', va='center',
                   fontsize=11, fontweight='bold')
            
            if state not in env.terminals:
                action = np.argmax(policy[i])
                dy, dx = env.actions[action]
                ax.arrow(x, y, dx*0.25, dy*0.25, head_width=0.08,
                        head_length=0.05, fc='darkblue', ec='darkblue',
                        alpha=0.7)
        
        # Mark special states
        ax.plot(0, 0, 'go', markersize=10)
        ax.plot(4, 4, 'r*', markersize=15)
        ax.plot(3, 1, 'y^', markersize=10)  # Bonus reward
        ax.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, facecolor='black'))
        
        ax.set_title(f'γ = {gamma}', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(4.5, -0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
    
    plt.suptitle('Effect of Discount Factor on Optimal Policy', fontsize=16)
    plt.tight_layout()
    plt.show()


def benchmark_algorithms():
    """Benchmark different DP algorithms."""
    print("\n" + "="*60)
    print("Algorithm Benchmarking")
    print("="*60)
    
    # Create environments of different sizes
    sizes = [5, 10, 15, 20]
    results = {
        'Policy Iteration': [],
        'Value Iteration': [],
        'Modified PI (k=5)': [],
        'Prioritized Sweeping': []
    }
    
    for size in sizes:
        print(f"\nGrid size: {size}x{size}")
        env = GridWorld(
            height=size,
            width=size,
            start=(0, 0),
            terminals=[(size-1, size-1)]
        )
        
        solver = DPSolver(env, gamma=0.99)
        
        # Time each algorithm
        import time
        
        # Policy Iteration
        start = time.time()
        _, _, pi_iters = solver.policy_iteration()
        pi_time = time.time() - start
        results['Policy Iteration'].append(pi_time)
        print(f"  Policy Iteration: {pi_iters} iters, {pi_time:.3f}s")
        
        # Value Iteration
        start = time.time()
        _, _, vi_iters = solver.value_iteration()
        vi_time = time.time() - start
        results['Value Iteration'].append(vi_time)
        print(f"  Value Iteration: {vi_iters} iters, {vi_time:.3f}s")
        
        # Modified Policy Iteration
        start = time.time()
        _, _, mpi_iters = solver.modified_policy_iteration(k=5)
        mpi_time = time.time() - start
        results['Modified PI (k=5)'].append(mpi_time)
        print(f"  Modified PI: {mpi_iters} iters, {mpi_time:.3f}s")
        
        # Prioritized Sweeping
        start = time.time()
        _, _, ps_iters = solver.prioritized_sweeping_value_iteration()
        ps_time = time.time() - start
        results['Prioritized Sweeping'].append(ps_time)
        print(f"  Prioritized Sweeping: {ps_iters} updates, {ps_time:.3f}s")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for alg, times in results.items():
        plt.plot(sizes, times, 'o-', linewidth=2, markersize=8, label=alg)
    
    plt.xlabel('Grid Size', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Algorithm Performance vs Problem Size', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_simple_gridworld()
    demonstrate_all_environments()
    interactive_parameter_exploration()
    benchmark_algorithms()
    
    print("\n" + "="*60)
    print("All demonstrations completed!")
    print("="*60)