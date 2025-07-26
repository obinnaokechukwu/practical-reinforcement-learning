import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from dynamic_programming import DynamicProgramming
import seaborn as sns

def experiment_basic_gridworld():
    """Basic gridworld with simple obstacles."""
    print("=== Experiment 1: Basic Gridworld ===")
    
    # Create environment
    env = GridWorld(
        height=4, 
        width=4,
        start=(0, 0),
        terminals=[(3, 3)],
        obstacles=[(1, 1), (2, 1)]
    )
    
    # Solve with different algorithms
    dp = DynamicProgramming(env, gamma=0.9)
    
    # Value Iteration
    print("\nRunning Value Iteration...")
    vi_policy, vi_V, vi_info = dp.value_iteration(theta=1e-6)
    print(f"Converged in {vi_info['iterations']} iterations ({vi_info['time']:.3f}s)")
    
    # Policy Iteration
    print("\nRunning Policy Iteration...")
    pi_policy, pi_V, pi_info = dp.policy_iteration(theta=1e-6)
    print(f"Converged in {pi_info['iterations']} iterations ({pi_info['time']:.3f}s)")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.sca(axes[0])
    env.render_values(vi_V, title="Value Function (Optimal)")
    plt.sca(axes[1])
    env.render_policy(vi_policy, vi_V, title="Optimal Policy")
    plt.tight_layout()
    plt.savefig('basic_gridworld_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return env, vi_policy, vi_V

def experiment_stochastic_gridworld():
    """Gridworld with slippery floor (stochastic transitions)."""
    print("\n=== Experiment 2: Stochastic Gridworld (Slippery Floor) ===")
    
    # Create environment with slip probability
    env = GridWorld(
        height=4,
        width=4, 
        start=(0, 0),
        terminals=[(3, 3)],
        obstacles=[(1, 1), (2, 1)],
        slip_prob=0.2  # 20% chance of slipping perpendicular to intended direction
    )
    
    dp = DynamicProgramming(env, gamma=0.9)
    
    # Compare deterministic vs stochastic
    det_env = GridWorld(
        height=4,
        width=4,
        start=(0, 0), 
        terminals=[(3, 3)],
        obstacles=[(1, 1), (2, 1)],
        slip_prob=0.0
    )
    dp_det = DynamicProgramming(det_env, gamma=0.9)
    
    # Solve both
    _, det_V, _ = dp_det.value_iteration()
    _, stoch_V, _ = dp.value_iteration()
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.sca(axes[0])
    det_env.render_values(det_V, title="Deterministic Environment")
    plt.sca(axes[1])
    env.render_values(stoch_V, title="Stochastic Environment (20% slip)")
    plt.tight_layout()
    plt.savefig('stochastic_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Average value difference: {np.mean(np.abs(det_V - stoch_V)):.3f}")

def experiment_reward_shaping():
    """Effect of different reward structures."""
    print("\n=== Experiment 3: Reward Shaping ===")
    
    # Create environments with different reward structures
    configs = [
        {
            'name': 'Sparse Rewards',
            'rewards': {},  # Only -1 per step
            'terminals': [(3, 3)]
        },
        {
            'name': 'Positive Goal Reward',
            'rewards': {(3, 3): 10},
            'terminals': [(3, 3)]
        },
        {
            'name': 'Negative Hazard',
            'rewards': {(2, 2): -10},
            'terminals': [(3, 3)]
        },
        {
            'name': 'Breadcrumbs',
            'rewards': {(1, 0): -0.5, (2, 0): -0.5, (3, 0): -0.5, 
                       (3, 1): -0.5, (3, 2): -0.5},
            'terminals': [(3, 3)]
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, config in enumerate(configs):
        env = GridWorld(
            height=4,
            width=4,
            start=(0, 0),
            terminals=config['terminals'],
            obstacles=[(1, 1)],
            rewards=config['rewards']
        )
        
        dp = DynamicProgramming(env, gamma=0.9)
        policy, V, _ = dp.value_iteration()
        
        plt.sca(axes[idx])
        env.render_policy(policy, V, title=config['name'])
    
    plt.tight_layout()
    plt.savefig('reward_shaping_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def experiment_algorithm_comparison():
    """Compare convergence of different DP algorithms."""
    print("\n=== Experiment 4: Algorithm Convergence Comparison ===")
    
    # Create a larger environment
    env = GridWorld(
        height=10,
        width=10,
        start=(0, 0),
        terminals=[(9, 9)],
        obstacles=[(i, 5) for i in range(8)] + [(5, j) for j in range(8)]
    )
    
    dp = DynamicProgramming(env, gamma=0.95)
    
    # Run all algorithms
    algorithms = {
        'Value Iteration': dp.value_iteration,
        'Policy Iteration': dp.policy_iteration,
        'Modified PI (k=3)': lambda: dp.modified_policy_iteration(k=3),
        'Modified PI (k=10)': lambda: dp.modified_policy_iteration(k=10),
    }
    
    results = {}
    for name, algo in algorithms.items():
        print(f"\nRunning {name}...")
        policy, V, info = algo()
        results[name] = info
        print(f"  Iterations: {info['iterations']}")
        print(f"  Time: {info['time']:.3f}s")
    
    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Iterations vs Time
    names = list(results.keys())
    iterations = [results[name]['iterations'] for name in names]
    times = [results[name]['time'] for name in names]
    
    ax1.bar(names, iterations, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_ylabel('Iterations to Convergence', fontsize=12)
    ax1.set_title('Algorithm Iterations', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(names, times, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax2.set_ylabel('Time to Convergence (s)', fontsize=12)
    ax2.set_title('Algorithm Runtime', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def experiment_discount_factor():
    """Effect of discount factor on optimal policy."""
    print("\n=== Experiment 5: Discount Factor Analysis ===")
    
    # Create environment with interesting structure
    env = GridWorld(
        height=5,
        width=5,
        start=(2, 0),
        terminals=[(0, 4), (4, 4)],
        rewards={(0, 4): 10, (4, 4): 1},  # Different rewards for different goals
        obstacles=[(2, 2)]
    )
    
    gammas = [0.1, 0.5, 0.9, 0.99]
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, gamma in enumerate(gammas):
        dp = DynamicProgramming(env, gamma=gamma)
        policy, V, _ = dp.value_iteration()
        
        plt.sca(axes[idx])
        env.render_policy(policy, V, title=f'Discount Factor γ = {gamma}')
        
        # Calculate which goal is preferred
        start_idx = env.state_to_idx[env.start]
        print(f"\nγ = {gamma}:")
        print(f"  Start state value: {V[start_idx]:.2f}")
    
    plt.tight_layout()
    plt.savefig('discount_factor_effect.png', dpi=150, bbox_inches='tight')
    plt.show()

def experiment_prioritized_sweeping():
    """Compare standard value iteration with prioritized sweeping."""
    print("\n=== Experiment 6: Prioritized Sweeping ===")
    
    # Create sparse reward environment where prioritized sweeping shines
    env = GridWorld(
        height=20,
        width=20,
        start=(0, 0),
        terminals=[(19, 19)],
        rewards={(19, 19): 100},
        obstacles=[(i, 10) for i in range(15)] + [(10, j) for j in range(15)]
    )
    
    dp = DynamicProgramming(env, gamma=0.95)
    
    # Standard Value Iteration
    print("\nRunning Standard Value Iteration...")
    _, _, vi_info = dp.value_iteration()
    
    # Prioritized Sweeping
    print("\nRunning Prioritized Sweeping...")
    _, _, ps_info = dp.prioritized_sweeping()
    
    print(f"\nStandard VI: {vi_info['iterations']} iterations, {vi_info['time']:.3f}s")
    print(f"Prioritized: {ps_info['updates']} updates, {ps_info['time']:.3f}s")
    print(f"Speedup: {vi_info['time'] / ps_info['time']:.2f}x")

def visualize_value_iteration_progress():
    """Visualize how values propagate during value iteration."""
    print("\n=== Visualization: Value Propagation ===")
    
    # Simple environment to clearly show propagation
    env = GridWorld(
        height=5,
        width=5,
        start=(0, 0),
        terminals=[(4, 4)],
        rewards={(4, 4): 10}
    )
    
    dp = DynamicProgramming(env, gamma=0.9)
    _, _, info = dp.value_iteration()
    
    # Select iterations to visualize
    iterations_to_show = [0, 1, 2, 3, 5, 10, 20, info['iterations']-1]
    iterations_to_show = [i for i in iterations_to_show if i < len(info['history'])]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, iteration in enumerate(iterations_to_show):
        if idx < len(axes):
            plt.sca(axes[idx])
            V = info['history'][iteration]['V']
            env.render_values(V, title=f'Iteration {iteration}')
    
    plt.tight_layout()
    plt.savefig('value_propagation.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run all experiments
    print("Running Dynamic Programming Experiments\n" + "="*50)
    
    # Basic experiments
    env, policy, V = experiment_basic_gridworld()
    experiment_stochastic_gridworld()
    experiment_reward_shaping()
    
    # Advanced experiments
    experiment_algorithm_comparison()
    experiment_discount_factor()
    experiment_prioritized_sweeping()
    
    # Visualization
    visualize_value_iteration_progress()
    
    print("\n" + "="*50)
    print("All experiments completed! Check the generated PNG files for visualizations.")