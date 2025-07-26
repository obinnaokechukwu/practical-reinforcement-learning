"""
Capstone Project: End-to-End Production RL System for Drone Delivery

This implementation demonstrates a complete production-ready RL system
for autonomous drone delivery in urban environments.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
import asyncio
from collections import deque
import threading
import queue


@dataclass
class DroneState:
    """Complete state representation for a delivery drone."""
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    battery_level: float  # 0-100%
    payload_weight: float  # kg
    weather_conditions: Dict  # wind, rain, visibility
    nearby_obstacles: List[np.ndarray]  # positions of obstacles
    destination: np.ndarray  # target position
    timestamp: datetime


@dataclass
class SafetyConstraints:
    """Safety constraints for drone operations."""
    max_altitude: float = 120.0  # meters
    min_altitude: float = 30.0  # meters
    max_speed: float = 20.0  # m/s
    min_battery_reserve: float = 20.0  # %
    no_fly_zones: List[Tuple[np.ndarray, float]]  # center, radius
    max_wind_speed: float = 15.0  # m/s
    min_visibility: float = 1000.0  # meters


class DronePolicy(nn.Module):
    """Neural network policy for drone control."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, action_dim))
        layers.append(nn.Tanh())  # Bounded actions
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class SafetyFilter:
    """Real-time safety filter for drone actions."""
    
    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints
        self.logger = logging.getLogger(__name__)
        
    def filter_action(self, state: DroneState, proposed_action: np.ndarray) -> np.ndarray:
        """Filter proposed action to ensure safety constraints."""
        filtered_action = proposed_action.copy()
        
        # Check altitude constraints
        predicted_altitude = state.position[2] + proposed_action[2]
        if predicted_altitude > self.constraints.max_altitude:
            filtered_action[2] = self.constraints.max_altitude - state.position[2]
        elif predicted_altitude < self.constraints.min_altitude:
            filtered_action[2] = self.constraints.min_altitude - state.position[2]
        
        # Check speed constraints
        predicted_velocity = state.velocity + filtered_action
        speed = np.linalg.norm(predicted_velocity)
        if speed > self.constraints.max_speed:
            filtered_action *= self.constraints.max_speed / speed
        
        # Check no-fly zones
        predicted_position = state.position + predicted_velocity * 0.1  # dt = 0.1s
        for zone_center, zone_radius in self.constraints.no_fly_zones:
            if np.linalg.norm(predicted_position - zone_center) < zone_radius:
                # Steer away from no-fly zone
                away_direction = predicted_position - zone_center
                away_direction /= np.linalg.norm(away_direction)
                filtered_action[:2] = away_direction[:2] * np.linalg.norm(filtered_action[:2])
        
        # Check battery constraints
        if state.battery_level < self.constraints.min_battery_reserve:
            # Force landing
            filtered_action[2] = -2.0  # Descend
            self.logger.warning(f"Low battery: {state.battery_level}%. Forcing landing.")
        
        # Weather constraints
        if state.weather_conditions['wind_speed'] > self.constraints.max_wind_speed:
            # Reduce action magnitude in high winds
            filtered_action *= 0.5
            self.logger.warning(f"High winds: {state.weather_conditions['wind_speed']} m/s")
        
        return filtered_action


class PerformanceMonitor:
    """Monitor system performance and safety metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = {
            'delivery_success_rate': deque(maxlen=window_size),
            'average_delivery_time': deque(maxlen=window_size),
            'safety_violations': deque(maxlen=window_size),
            'battery_efficiency': deque(maxlen=window_size),
            'customer_satisfaction': deque(maxlen=window_size)
        }
        self.alert_thresholds = {
            'delivery_success_rate': 0.95,
            'safety_violations': 0.01,
            'battery_efficiency': 0.8
        }
        
    def update_metrics(self, episode_data: Dict):
        """Update performance metrics with new episode data."""
        for key, value in episode_data.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Check for alerts
        self._check_alerts()
    
    def _check_alerts(self):
        """Check if any metrics violate thresholds."""
        alerts = []
        
        if len(self.metrics['delivery_success_rate']) > 100:
            success_rate = np.mean(self.metrics['delivery_success_rate'])
            if success_rate < self.alert_thresholds['delivery_success_rate']:
                alerts.append(f"Low delivery success rate: {success_rate:.2%}")
        
        if len(self.metrics['safety_violations']) > 100:
            violation_rate = np.mean(self.metrics['safety_violations'])
            if violation_rate > self.alert_thresholds['safety_violations']:
                alerts.append(f"High safety violation rate: {violation_rate:.2%}")
        
        for alert in alerts:
            logging.error(f"ALERT: {alert}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics of all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if len(values) > 0:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'recent': list(values)[-10:]  # Last 10 values
                }
        return summary


class ExperienceCollector:
    """Collect and manage experience from drone fleet."""
    
    def __init__(self, buffer_size: int = 1000000):
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        
    def add_experience(self, experience: Dict):
        """Add new experience to buffer."""
        with self.lock:
            self.buffer.append(experience)
    
    def sample_batch(self, batch_size: int) -> List[Dict]:
        """Sample a batch of experiences."""
        with self.lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]


class PolicyServer:
    """Serve policies to drone fleet with versioning and rollback."""
    
    def __init__(self):
        self.current_policy = None
        self.policy_version = 0
        self.policy_history = {}
        self.performance_history = {}
        self.rollback_threshold = 0.9  # Performance drop threshold
        
    def update_policy(self, new_policy: DronePolicy, metrics: Dict):
        """Update policy with automatic rollback on performance degradation."""
        self.policy_version += 1
        
        # Store current policy
        self.policy_history[self.policy_version] = new_policy
        self.performance_history[self.policy_version] = metrics
        
        # Check if we should rollback
        if self.policy_version > 1:
            current_performance = metrics.get('delivery_success_rate', 0)
            previous_performance = self.performance_history[self.policy_version - 1].get(
                'delivery_success_rate', 0
            )
            
            if current_performance < previous_performance * self.rollback_threshold:
                logging.warning(
                    f"Performance degradation detected. Rolling back from v{self.policy_version} to v{self.policy_version - 1}"
                )
                self.policy_version -= 1
                self.current_policy = self.policy_history[self.policy_version]
                return
        
        self.current_policy = new_policy
        logging.info(f"Policy updated to version {self.policy_version}")
    
    def get_policy(self) -> Tuple[DronePolicy, int]:
        """Get current policy and version."""
        return self.current_policy, self.policy_version


class DroneFleetManager:
    """Manage fleet of delivery drones."""
    
    def __init__(self, num_drones: int, policy_server: PolicyServer,
                 experience_collector: ExperienceCollector,
                 performance_monitor: PerformanceMonitor):
        self.num_drones = num_drones
        self.policy_server = policy_server
        self.experience_collector = experience_collector
        self.performance_monitor = performance_monitor
        self.active_deliveries = {}
        self.delivery_queue = asyncio.Queue()
        
    async def assign_delivery(self, pickup_location: np.ndarray, 
                            delivery_location: np.ndarray,
                            package_weight: float):
        """Assign delivery to available drone."""
        delivery_task = {
            'id': datetime.now().timestamp(),
            'pickup': pickup_location,
            'delivery': delivery_location,
            'weight': package_weight,
            'status': 'pending'
        }
        
        await self.delivery_queue.put(delivery_task)
        logging.info(f"Delivery {delivery_task['id']} added to queue")
        
    async def drone_worker(self, drone_id: int):
        """Worker coroutine for individual drone."""
        safety_filter = SafetyFilter(SafetyConstraints())
        
        while True:
            # Get delivery task
            delivery = await self.delivery_queue.get()
            
            try:
                # Execute delivery
                result = await self._execute_delivery(drone_id, delivery, safety_filter)
                
                # Collect experience
                self.experience_collector.add_experience(result['experience'])
                
                # Update metrics
                self.performance_monitor.update_metrics(result['metrics'])
                
                logging.info(f"Drone {drone_id} completed delivery {delivery['id']}")
                
            except Exception as e:
                logging.error(f"Drone {drone_id} failed delivery: {e}")
                
            finally:
                self.delivery_queue.task_done()
    
    async def _execute_delivery(self, drone_id: int, delivery: Dict, 
                               safety_filter: SafetyFilter) -> Dict:
        """Execute a single delivery mission."""
        # Simplified delivery execution
        # In production, this would interface with actual drone hardware
        
        policy, version = self.policy_server.get_policy()
        
        # Simulate delivery
        total_time = 0
        battery_used = 0
        safety_violations = 0
        
        # Navigate to pickup
        # ... (simplified for brevity)
        
        # Navigate to delivery
        # ... (simplified for brevity)
        
        # Prepare results
        success = np.random.random() > 0.05  # 95% success rate for demo
        
        return {
            'experience': {
                'drone_id': drone_id,
                'delivery_id': delivery['id'],
                'trajectory': [],  # Would contain actual state-action pairs
                'reward': 1.0 if success else -1.0
            },
            'metrics': {
                'delivery_success_rate': 1.0 if success else 0.0,
                'average_delivery_time': total_time,
                'safety_violations': safety_violations,
                'battery_efficiency': 1.0 - battery_used / 100.0,
                'customer_satisfaction': np.random.uniform(0.8, 1.0) if success else 0.0
            }
        }
    
    async def start_fleet(self):
        """Start all drones in the fleet."""
        tasks = []
        for drone_id in range(self.num_drones):
            task = asyncio.create_task(self.drone_worker(drone_id))
            tasks.append(task)
        
        await asyncio.gather(*tasks)


class ProductionRLSystem:
    """Complete production RL system for drone delivery."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.policy_server = PolicyServer()
        self.experience_collector = ExperienceCollector(
            buffer_size=config.get('buffer_size', 1000000)
        )
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize fleet
        self.fleet_manager = DroneFleetManager(
            num_drones=config.get('num_drones', 10),
            policy_server=self.policy_server,
            experience_collector=self.experience_collector,
            performance_monitor=self.performance_monitor
        )
        
        # Training infrastructure
        self.training_enabled = config.get('continuous_learning', True)
        self.training_interval = config.get('training_interval', 3600)  # seconds
        
    async def run(self):
        """Run the production system."""
        # Start fleet
        fleet_task = asyncio.create_task(self.fleet_manager.start_fleet())
        
        # Start training loop if enabled
        if self.training_enabled:
            training_task = asyncio.create_task(self._training_loop())
        
        # Start monitoring
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Wait for all tasks
        tasks = [fleet_task, monitoring_task]
        if self.training_enabled:
            tasks.append(training_task)
        
        await asyncio.gather(*tasks)
    
    async def _training_loop(self):
        """Continuous training loop."""
        while True:
            await asyncio.sleep(self.training_interval)
            
            # Sample experience batch
            batch = self.experience_collector.sample_batch(batch_size=1000)
            
            if len(batch) > 100:  # Minimum experiences needed
                # Train new policy (simplified)
                new_policy = self._train_policy(batch)
                
                # Get current metrics
                metrics = self.performance_monitor.get_summary()
                
                # Update policy with automatic rollback
                self.policy_server.update_policy(new_policy, metrics)
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Get performance summary
            summary = self.performance_monitor.get_summary()
            
            # Log summary
            logging.info(f"Performance Summary: {json.dumps(summary, indent=2)}")
            
            # Check for critical issues
            if 'delivery_success_rate' in summary:
                success_rate = summary['delivery_success_rate']['mean']
                if success_rate < 0.9:
                    logging.critical(f"Critical: Low success rate {success_rate:.2%}")
    
    def _train_policy(self, experiences: List[Dict]) -> DronePolicy:
        """Train new policy from experiences (simplified)."""
        # In production, this would be a full RL training pipeline
        # For demo, return a new random policy
        state_dim = self.config.get('state_dim', 32)
        action_dim = self.config.get('action_dim', 3)
        hidden_sizes = self.config.get('hidden_sizes', [256, 256])
        
        new_policy = DronePolicy(state_dim, action_dim, hidden_sizes)
        
        # Simulate training
        # ... (actual training code would go here)
        
        return new_policy


async def demo_production_system():
    """Demonstrate the production RL system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # System configuration
    config = {
        'num_drones': 5,
        'buffer_size': 100000,
        'continuous_learning': True,
        'training_interval': 300,  # 5 minutes
        'state_dim': 32,
        'action_dim': 3,
        'hidden_sizes': [256, 256]
    }
    
    # Create production system
    system = ProductionRLSystem(config)
    
    # Initialize with a random policy
    initial_policy = DronePolicy(
        config['state_dim'],
        config['action_dim'],
        config['hidden_sizes']
    )
    system.policy_server.update_policy(initial_policy, {})
    
    # Create some delivery tasks
    async def generate_deliveries():
        for i in range(20):
            pickup = np.random.uniform(-10, 10, 3)
            delivery = np.random.uniform(-10, 10, 3)
            weight = np.random.uniform(0.5, 5.0)
            
            await system.fleet_manager.assign_delivery(pickup, delivery, weight)
            await asyncio.sleep(10)  # New delivery every 10 seconds
    
    # Run system with delivery generation
    await asyncio.gather(
        system.run(),
        generate_deliveries()
    )


if __name__ == "__main__":
    # Run the demo
    print("Starting Production RL System Demo...")
    print("This demonstrates a complete drone delivery system with:")
    print("- Real-time policy serving")
    print("- Safety filtering")
    print("- Performance monitoring")
    print("- Continuous learning")
    print("- Automatic rollback on performance degradation")
    print()
    
    try:
        asyncio.run(demo_production_system())
    except KeyboardInterrupt:
        print("\nShutting down system...")