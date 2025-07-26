"""
Treasure Hunt Environment for Q-Learning Experiments

A grid world environment where an agent must collect treasures while avoiding traps.
This environment demonstrates key challenges for Q-learning:
- Sparse rewards (treasures are scattered)
- Negative rewards (traps must be avoided)
- Time pressure (limited battery)
- Hierarchical tasks (keys unlock doors)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches
from collections import defaultdict
import random


class TreasureHuntEnv:
    """
    A grid world where an agent hunts for treasure while avoiding traps.
    """
    
    def __init__(self, width=10, height=10, num_treasures=3, num_traps=5):
        self.width = width
        self.height = height
        
        # Core components
        self._setup_grid()
        self._place_objects(num_treasures, num_traps)
        self._setup_action_space()
        
        # Episode tracking
        self.reset()
    
    def _setup_grid(self):
        """Initialize the grid structure."""
        self.grid = np.zeros((self.height, self.width))
        self.start_pos = (0, 0)
        
    def _place_objects(self, num_treasures, num_traps):
        """Place treasures, traps, and other objects on the grid."""
        # Treasures: positive rewards of varying values
        self.treasures = {}
        treasure_positions = self._random_positions(num_treasures, exclude=[self.start_pos])
        for i, pos in enumerate(treasure_positions):
            value = (i + 1) * 10  # 10, 20, 30 points
            self.treasures[pos] = value
            self.grid[pos] = 1  # Mark on grid
        
        # Traps: negative rewards that end the episode
        self.traps = set()
        trap_positions = self._random_positions(
            num_traps, 
            exclude=[self.start_pos] + list(self.treasures.keys())
        )
        for pos in trap_positions:
            self.traps.add(pos)
            self.grid[pos] = -1  # Mark on grid
        
        # Advanced feature: keys and doors
        # This creates a hierarchical task structure
        self.doors = {(self.height//2, self.width//2)}  # Center door
        self.keys = {(2, 2): (self.height//2, self.width//2)}  # Key location -> door location
        
    def _setup_action_space(self):
        """Define the action space."""
        # Four discrete actions: up, right, down, left
        self.actions = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }
        
        # Create a simple action space object compatible with standard RL interfaces
        self.action_space = type('ActionSpace', (), {
            'n': 4,
            'sample': lambda: np.random.randint(4)
        })()
        
    def reset(self):
        """Reset the environment to start a new episode."""
        self.agent_pos = self.start_pos
        self.collected_treasures = set()
        self.collected_keys = set()
        self.total_reward = 0
        
        # Battery: adds time pressure
        self.max_battery = 100
        self.battery = self.max_battery
        
        return self._get_state()
        
    def _random_positions(self, n, exclude):
        """Generate n random positions excluding certain cells."""
        positions = []
        while len(positions) < n:
            pos = (np.random.randint(self.height), np.random.randint(self.width))
            if pos not in exclude and pos not in positions:
                positions.append(pos)
        return positions
    
    def _get_state(self):
        """
        Encode the current state as a hashable tuple.
        
        State representation is crucial for Q-learning. We need to capture
        all information that affects future rewards:
        - Current position
        - What we've collected (affects available rewards)
        - What keys we have (affects accessible areas)
        - Time remaining (affects optimal policy)
        """
        return (
            self.agent_pos,
            frozenset(self.collected_treasures),
            frozenset(self.collected_keys),
            self.battery // 10  # Discretize battery to reduce state space
        )
    
    def step(self, action):
        """
        Execute an action and return the result.
        
        Returns:
            tuple: (next_state, reward, done)
        """
        # 1. Movement dynamics
        dy, dx = self.actions[action]
        new_pos = (self.agent_pos[0] + dy, self.agent_pos[1] + dx)
        
        # Check if move is valid
        if self._is_valid_position(new_pos):
            # Check for locked doors
            if new_pos in self.doors and new_pos not in self.collected_keys:
                new_pos = self.agent_pos  # Can't pass through
            else:
                self.agent_pos = new_pos  # Move successful
        else:
            new_pos = self.agent_pos  # Hit wall, stay put
        
        # 2. Time dynamics
        self.battery -= 1
        
        # 3. Reward calculation
        reward = self._calculate_reward()
        
        # 4. Check termination conditions
        done = self._check_termination()
        
        # 5. Update total score
        self.total_reward += reward
        
        return self._get_state(), reward, done
    
    def _is_valid_position(self, pos):
        """Check if a position is within grid boundaries."""
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width
    
    def _calculate_reward(self):
        """
        Calculate reward for current position.
        
        Reward structure teaches the agent:
        - Move efficiently (small step penalty)
        - Collect treasures (large positive rewards)
        - Avoid traps (large negative rewards)
        - Collect keys (small positive rewards)
        """
        reward = -0.1  # Base step penalty encourages efficiency
        
        # Treasure collection
        if self.agent_pos in self.treasures and self.agent_pos not in self.collected_treasures:
            treasure_value = self.treasures[self.agent_pos]
            reward += treasure_value
            self.collected_treasures.add(self.agent_pos)
            
            # Bonus for collecting all treasures
            if len(self.collected_treasures) == len(self.treasures):
                reward += 50  # Completion bonus
        
        # Trap penalty
        if self.agent_pos in self.traps:
            reward = -50  # Override other rewards
        
        # Key collection
        if self.agent_pos in self.keys:
            door_pos = self.keys[self.agent_pos]
            if door_pos not in self.collected_keys:
                self.collected_keys.add(door_pos)
                reward += 5  # Small reward for progress
        
        return reward
    
    def _check_termination(self):
        """Check if episode should end."""
        # Victory: collected all treasures
        if len(self.collected_treasures) == len(self.treasures):
            return True
        
        # Defeat: stepped on trap
        if self.agent_pos in self.traps:
            return True
        
        # Timeout: battery depleted
        if self.battery <= 0:
            return True
        
        return False
    
    def render(self, Q=None, save_path=None):
        """
        Visualize the environment and optionally Q-values.
        
        Args:
            Q: Q-table for visualizing action values
            save_path: If provided, save figure to file
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        for i in range(self.height + 1):
            ax.axhline(i, color='black', linewidth=1)
        for j in range(self.width + 1):
            ax.axvline(j, color='black', linewidth=1)
        
        # Draw treasures
        for pos, value in self.treasures.items():
            if pos not in self.collected_treasures:
                y, x = pos
                circle = Circle((x + 0.5, y + 0.5), 0.3, 
                               color='gold', ec='orange', linewidth=2)
                ax.add_patch(circle)
                ax.text(x + 0.5, y + 0.5, f'${value}', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw traps
        for pos in self.traps:
            y, x = pos
            rect = Rectangle((x + 0.1, y + 0.1), 0.8, 0.8,
                           facecolor='red', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.5, '⚠', 
                   ha='center', va='center', fontsize=20)
        
        # Draw doors and keys
        for door_pos in self.doors:
            y, x = door_pos
            if door_pos not in self.collected_keys:
                rect = Rectangle((x + 0.1, y + 0.1), 0.8, 0.8,
                               facecolor='brown', alpha=0.8)
                ax.add_patch(rect)
                ax.text(x + 0.5, y + 0.5, '🚪', 
                       ha='center', va='center', fontsize=16)
        
        for key_pos, door_pos in self.keys.items():
            if door_pos not in self.collected_keys:
                y, x = key_pos
                ax.text(x + 0.5, y + 0.5, '🔑', 
                       ha='center', va='center', fontsize=16)
        
        # Draw agent
        y, x = self.agent_pos
        circle = Circle((x + 0.5, y + 0.5), 0.25,
                       color='blue', alpha=0.8)
        ax.add_patch(circle)
        
        # Draw Q-values if provided
        if Q is not None:
            state = self._get_state()
            if state in Q:
                y, x = self.agent_pos
                for action, (dy, dx) in self.actions.items():
                    q_value = Q[state][action]
                    ax.arrow(x + 0.5, y + 0.5, dx * 0.3, dy * 0.3,
                           head_width=0.1, head_length=0.05,
                           fc='green' if q_value > 0 else 'red',
                           alpha=min(abs(q_value) / 10, 1.0))
        
        # Battery indicator
        battery_color = 'green' if self.battery > 30 else 'red'
        ax.text(self.width + 0.5, 0.5, f'Battery: {self.battery}', 
               fontsize=12, fontweight='bold', color=battery_color)
        
        # Score
        ax.text(self.width + 0.5, 1.5, f'Score: {self.total_reward:.1f}', 
               fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title('Treasure Hunt Environment', fontsize=16, fontweight='bold')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='gold', label='Treasure'),
            mpatches.Patch(color='red', label='Trap'),
            mpatches.Patch(color='brown', label='Door'),
            mpatches.Patch(color='blue', label='Agent')
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig, ax
    
    def get_test_states(self):
        """Return representative states for debugging."""
        test_states = []
        
        # Starting position
        test_states.append((
            self.start_pos,
            frozenset(),
            frozenset(),
            self.max_battery // 10
        ))
        
        # Near a treasure
        if self.treasures:
            treasure_pos = list(self.treasures.keys())[0]
            test_states.append((
                treasure_pos,
                frozenset(),
                frozenset(),
                self.max_battery // 10
            ))
        
        # With key collected
        if self.keys:
            key_pos = list(self.keys.keys())[0]
            door_pos = self.keys[key_pos]
            test_states.append((
                key_pos,
                frozenset(),
                frozenset({door_pos}),
                self.max_battery // 10
            ))
        
        return test_states


if __name__ == "__main__":
    # Test the environment
    env = TreasureHuntEnv(width=8, height=8, num_treasures=3, num_traps=4)
    
    # Run a random episode
    state = env.reset()
    print(f"Initial state: {state}")
    
    for step in range(20):
        action = env.action_space.sample()
        next_state, reward, done = env.step(action)
        print(f"Step {step}: Action={action}, Reward={reward:.2f}, Done={done}")
        
        if done:
            print(f"Episode ended. Total reward: {env.total_reward:.1f}")
            break
    
    # Visualize final state
    env.render()
    plt.show()