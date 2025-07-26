"""
GridWorld Environment for Dynamic Programming Experiments

This module implements a flexible gridworld environment that supports:
- Configurable layouts with obstacles and rewards
- Stochastic transitions (wind, slippery surfaces)
- Rich visualization capabilities
- Standard MDP interface compatible with OpenAI Gym conventions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches


class GridWorld:
    """
    A customizable gridworld environment for Dynamic Programming experiments.
    
    The environment follows the standard MDP formulation with:
    - States: Grid positions (excluding obstacles)
    - Actions: UP, RIGHT, DOWN, LEFT
    - Transitions: Can be deterministic or stochastic
    - Rewards: Configurable per state, with default step cost
    """
    
    def __init__(self, height=4, width=4, start=(0, 0), terminals=None, 
                 obstacles=None, rewards=None, wind=None, slip_prob=0.0):
        """
        Initialize the gridworld environment.
        
        Args:
            height: Number of rows in the grid
            width: Number of columns in the grid
            start: Starting position (row, col)
            terminals: List of terminal state positions
            obstacles: List of obstacle positions (impassable cells)
            rewards: Dict mapping positions to reward values
            wind: Dict mapping positions to wind vectors (dy, dx)
            slip_prob: Probability of slipping perpendicular to intended direction
        """
        self.height = height
        self.width = width
        self.start = start
        self.terminals = terminals or [(height-1, width-1)]
        self.obstacles = obstacles or []
        self.rewards = rewards or {}
        self.wind = wind or {}
        self.slip_prob = slip_prob
        
        # Define action space
        self.actions = {
            0: (-1, 0),  # UP
            1: (0, 1),   # RIGHT
            2: (1, 0),   # DOWN
            3: (0, -1)   # LEFT
        }
        self.action_names = ['↑', '→', '↓', '←']
        
        # Create state space (excluding obstacles)
        self.states = []
        self.state_to_idx = {}
        idx = 0
        for i in range(height):
            for j in range(width):
                if (i, j) not in self.obstacles:
                    self.states.append((i, j))
                    self.state_to_idx[(i, j)] = idx
                    idx += 1
        
        self.nS = len(self.states)
        self.nA = len(self.actions)
        
        # Build transition and reward models
        self.P = self._build_transitions()
        
    def _build_transitions(self):
        """
        Build the transition probability matrix.
        
        Returns:
            P: Dict where P[s][a] = [(prob, next_state, reward, done), ...]
        """
        P = {}
        
        for idx, state in enumerate(self.states):
            P[idx] = {}
            
            for action in range(self.nA):
                transitions = []
                
                if state in self.terminals:
                    # Terminal states self-loop with zero reward
                    transitions.append((1.0, idx, 0.0, True))
                else:
                    # Calculate intended next state
                    intended_next = self._get_next_state(state, action)
                    
                    # Handle stochasticity
                    if self.slip_prob > 0:
                        # With slip_prob, go perpendicular to intended direction
                        for a in range(self.nA):
                            next_state = self._get_next_state(state, a)
                            next_idx = self.state_to_idx.get(next_state, idx)
                            
                            if a == action:
                                prob = 1 - self.slip_prob
                            elif (a - action) % 4 in [1, 3]:  # Perpendicular actions
                                prob = self.slip_prob / 2
                            else:
                                prob = 0
                            
                            if prob > 0:
                                reward = self._get_reward(state, a, next_state)
                                done = next_state in self.terminals
                                transitions.append((prob, next_idx, reward, done))
                    else:
                        # Deterministic transition
                        next_idx = self.state_to_idx.get(intended_next, idx)
                        reward = self._get_reward(state, action, intended_next)
                        done = intended_next in self.terminals
                        transitions.append((1.0, next_idx, reward, done))
                
                P[idx][action] = transitions
        
        return P
    
    def _get_next_state(self, state, action):
        """
        Get next state given current state and action, considering wind.
        
        Args:
            state: Current (row, col) position
            action: Action index
            
        Returns:
            Next (row, col) position
        """
        row, col = state
        drow, dcol = self.actions[action]
        
        # Apply action
        next_row = row + drow
        next_col = col + dcol
        
        # Apply wind if present
        if state in self.wind:
            wind_row, wind_col = self.wind[state]
            next_row += wind_row
            next_col += wind_col
        
        # Check boundaries and obstacles
        if (0 <= next_row < self.height and 
            0 <= next_col < self.width and
            (next_row, next_col) not in self.obstacles):
            return (next_row, next_col)
        else:
            return state  # Stay in place if hit wall/obstacle
    
    def _get_reward(self, state, action, next_state):
        """
        Get reward for transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Reward value
        """
        # Check for special rewards
        if next_state in self.rewards:
            return self.rewards[next_state]
        elif next_state in self.terminals:
            return 0.0  # Default terminal reward
        else:
            return -1.0  # Default step cost
    
    def render_values(self, V, title="State Values"):
        """
        Visualize value function as a heatmap with values displayed.
        
        Args:
            V: Value function (array of size nS)
            title: Plot title
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create value grid
        value_grid = np.full((self.height, self.width), np.nan)
        for idx, state in enumerate(self.states):
            value_grid[state] = V[idx]
        
        # Plot heatmap
        masked_grid = np.ma.masked_invalid(value_grid)
        im = ax.imshow(masked_grid, cmap='RdYlGn', aspect='equal', 
                       vmin=np.min(V), vmax=np.max(V))
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Draw grid and annotations
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.obstacles:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                         facecolor='black'))
                elif (i, j) in self.states:
                    idx = self.state_to_idx[(i, j)]
                    ax.text(j, i, f'{V[idx]:.2f}', ha='center', va='center',
                           fontsize=12, fontweight='bold')
                    
                    if (i, j) in self.terminals:
                        ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                             facecolor='none', 
                                             edgecolor='gold', linewidth=3))
                    
                    if (i, j) == self.start:
                        ax.add_patch(plt.Circle((j, i), 0.3, 
                                              facecolor='none', 
                                              edgecolor='blue', linewidth=3))
        
        # Add rewards
        for state, reward in self.rewards.items():
            if reward != -1:
                i, j = state
                ax.text(j, i-0.35, f'R={reward}', ha='center', va='center',
                       fontsize=10, style='italic', color='darkred')
        
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, linewidth=2, color='black')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig, ax
    
    def render_policy(self, policy, V=None, title="Optimal Policy"):
        """
        Visualize policy with arrows, optionally overlaid on value heatmap.
        
        Args:
            policy: Policy array (either nS x nA stochastic or nS deterministic)
            V: Optional value function for background heatmap
            title: Plot title
            
        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot value heatmap if provided
        if V is not None:
            value_grid = np.full((self.height, self.width), np.nan)
            for idx, state in enumerate(self.states):
                value_grid[state] = V[idx]
            masked_grid = np.ma.masked_invalid(value_grid)
            ax.imshow(masked_grid, cmap='RdYlGn', aspect='equal', 
                     alpha=0.3, vmin=np.min(V), vmax=np.max(V))
        
        # Draw grid and policy arrows
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) in self.obstacles:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, 
                                         facecolor='black'))
                elif (i, j) in self.states:
                    idx = self.state_to_idx[(i, j)]
                    
                    if (i, j) not in self.terminals:
                        # Get action from policy
                        if policy.ndim == 2:  # Stochastic policy
                            action = np.argmax(policy[idx])
                        else:  # Deterministic policy
                            action = policy[idx]
                        
                        # Draw arrow
                        di, dj = self.actions[action]
                        ax.arrow(j, i, dj*0.3, di*0.3, 
                                head_width=0.15, head_length=0.1, 
                                fc='darkblue', ec='darkblue', linewidth=2)
                    
                    # Mark special states
                    if (i, j) in self.terminals:
                        ax.add_patch(FancyBboxPatch((j-0.4, i-0.4), 0.8, 0.8,
                                                   boxstyle="round,pad=0.1",
                                                   facecolor='gold', 
                                                   edgecolor='darkorange',
                                                   linewidth=2))
                        ax.text(j, i, 'GOAL', ha='center', va='center',
                               fontsize=10, fontweight='bold')
                    
                    if (i, j) == self.start:
                        ax.add_patch(plt.Circle((j, i), 0.35, 
                                              facecolor='lightblue', 
                                              edgecolor='darkblue', 
                                              linewidth=2, alpha=0.7))
                        ax.text(j, i, 'START', ha='center', va='center',
                               fontsize=8, fontweight='bold')
        
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, linewidth=2, color='black')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='gold', label='Goal'),
            mpatches.Patch(color='lightblue', label='Start'),
            mpatches.Patch(color='black', label='Obstacle')
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.05, 1), fontsize=12)
        
        plt.tight_layout()
        return fig, ax


def create_example_environments():
    """
    Create a collection of gridworld environments showcasing different challenges.
    
    Returns:
        Dict of environment name -> GridWorld instance
    """
    environments = {}
    
    # Simple 4x4 gridworld - baseline environment
    environments['simple'] = GridWorld(
        height=4, width=4,
        start=(0, 0),
        terminals=[(3, 3)],
        rewards={(3, 3): 10}
    )
    
    # Maze-like environment with obstacles and penalty
    environments['maze'] = GridWorld(
        height=6, width=8,
        start=(0, 0),
        terminals=[(5, 7)],
        obstacles=[(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (3, 5), 
                   (4, 1), (4, 2)],
        rewards={(5, 7): 10, (2, 6): -10}  # Goal and penalty
    )
    
    # Windy gridworld - demonstrates planning under deterministic dynamics
    wind_field = {
        (0, 3): (-1, 0), (1, 3): (-1, 0), (2, 3): (-1, 0),
        (0, 4): (-1, 0), (1, 4): (-1, 0), (2, 4): (-1, 0),
        (0, 5): (-2, 0), (1, 5): (-2, 0), (2, 5): (-2, 0)
    }
    environments['windy'] = GridWorld(
        height=7, width=10,
        start=(3, 0),
        terminals=[(3, 7)],
        wind=wind_field,
        rewards={(3, 7): 0}  # No reward, just reach goal quickly
    )
    
    # Slippery gridworld - stochastic transitions
    environments['slippery'] = GridWorld(
        height=4, width=4,
        start=(0, 0),
        terminals=[(3, 3)],
        slip_prob=0.3,  # 30% chance of slipping
        rewards={(3, 3): 10, (3, 0): -10}  # Goal and cliff
    )
    
    # Cliff walking - risk vs reward tradeoff
    cliff_positions = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]
    cliff_rewards = {pos: -100 for pos in cliff_positions}
    cliff_rewards[(3, 9)] = 10  # Goal
    
    environments['cliff'] = GridWorld(
        height=4, width=10,
        start=(3, 0),
        terminals=[(3, 9)],
        rewards=cliff_rewards
    )
    
    return environments


if __name__ == "__main__":
    # Test the environment with a simple example
    env = create_example_environments()['simple']
    
    # Initialize random values for visualization
    V = np.random.randn(env.nS) * 5
    
    # Visualize
    fig1, ax1 = env.render_values(V, title="Random Value Function")
    
    # Create random policy
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        policy[s, np.random.randint(env.nA)] = 1.0
    
    fig2, ax2 = env.render_policy(policy, V, title="Random Policy")
    
    plt.show()