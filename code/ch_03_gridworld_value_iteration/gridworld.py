import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

class GridWorld:
    """
    A customizable gridworld environment for Dynamic Programming experiments.
    
    Features:
    - Configurable grid size and layout
    - Obstacles, rewards, and terminal states  
    - Stochastic transitions (wind, slippery surfaces)
    - Rich visualization capabilities
    """
    
    def __init__(self, height=4, width=4, start=(0, 0), terminals=None, 
                 obstacles=None, rewards=None, wind=None, slip_prob=0.0):
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
        
        # Create state space
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
        """Build transition probability matrix P[s][a] = [(prob, next_state, reward, done), ...]"""
        P = {}
        
        for idx, state in enumerate(self.states):
            P[idx] = {}
            
            for action in range(self.nA):
                transitions = []
                
                if state in self.terminals:
                    # Terminal states self-loop
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
                            elif (a - action) % 4 in [1, 3]:  # Perpendicular
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
        """Get next state given current state and action, considering wind."""
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
        """Get reward for transition."""
        # Check for special rewards
        if next_state in self.rewards:
            return self.rewards[next_state]
        elif next_state in self.terminals:
            return 0.0  # Default terminal reward
        else:
            return -1.0  # Default step cost
    
    def render_values(self, V, title="State Values"):
        """Visualize value function."""
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
        """Visualize policy with optional value backdrop."""
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