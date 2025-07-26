"""
Real-time monitoring dashboard for production RL systems.

This module provides visualization and monitoring capabilities
for deployed RL systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Tuple
import threading
import queue
import time


class MetricsCollector:
    """Collect and aggregate metrics from RL system."""
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.metrics_queue = queue.Queue()
        self.aggregated_metrics = {}
        self.time_series_data = {}
        self.lock = threading.Lock()
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(target=self._aggregate_metrics)
        self.aggregation_thread.daemon = True
        self.aggregation_thread.start()
    
    def add_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add a metric value."""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics_queue.put({
            'name': metric_name,
            'value': value,
            'timestamp': timestamp
        })
    
    def _aggregate_metrics(self):
        """Background thread to aggregate metrics."""
        while True:
            try:
                metric = self.metrics_queue.get(timeout=1.0)
                
                with self.lock:
                    # Initialize if needed
                    if metric['name'] not in self.time_series_data:
                        self.time_series_data[metric['name']] = {
                            'timestamps': [],
                            'values': []
                        }
                    
                    # Add to time series
                    self.time_series_data[metric['name']]['timestamps'].append(
                        metric['timestamp']
                    )
                    self.time_series_data[metric['name']]['values'].append(
                        metric['value']
                    )
                    
                    # Clean old data
                    self._clean_old_data(metric['name'])
                    
                    # Update aggregated metrics
                    self._update_aggregates(metric['name'])
                    
            except queue.Empty:
                continue
    
    def _clean_old_data(self, metric_name: str):
        """Remove data older than window."""
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
        
        data = self.time_series_data[metric_name]
        valid_indices = [
            i for i, ts in enumerate(data['timestamps'])
            if ts > cutoff_time
        ]
        
        data['timestamps'] = [data['timestamps'][i] for i in valid_indices]
        data['values'] = [data['values'][i] for i in valid_indices]
    
    def _update_aggregates(self, metric_name: str):
        """Update aggregated statistics."""
        values = self.time_series_data[metric_name]['values']
        
        if values:
            self.aggregated_metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values),
                'recent': values[-10:]  # Last 10 values
            }
    
    def get_time_series(self, metric_name: str) -> Tuple[List[datetime], List[float]]:
        """Get time series data for a metric."""
        with self.lock:
            if metric_name in self.time_series_data:
                data = self.time_series_data[metric_name]
                return data['timestamps'].copy(), data['values'].copy()
            return [], []
    
    def get_aggregates(self, metric_name: str) -> Dict:
        """Get aggregated statistics for a metric."""
        with self.lock:
            return self.aggregated_metrics.get(metric_name, {}).copy()


class RLDashboard:
    """Real-time dashboard for RL system monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.fig = None
        self.axes = None
        self.lines = {}
        self.texts = {}
        
    def setup_dashboard(self):
        """Setup the dashboard layout."""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('RL System Monitoring Dashboard', fontsize=16)
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Performance metrics
        self.axes = {
            'episode_return': self.fig.add_subplot(gs[0, 0]),
            'success_rate': self.fig.add_subplot(gs[0, 1]),
            'safety_violations': self.fig.add_subplot(gs[0, 2]),
            'action_distribution': self.fig.add_subplot(gs[1, 0]),
            'state_coverage': self.fig.add_subplot(gs[1, 1]),
            'system_health': self.fig.add_subplot(gs[1, 2]),
            'learning_progress': self.fig.add_subplot(gs[2, :2]),
            'alerts': self.fig.add_subplot(gs[2, 2])
        }
        
        # Configure each subplot
        self._configure_subplots()
        
    def _configure_subplots(self):
        """Configure individual subplots."""
        # Episode return
        self.axes['episode_return'].set_title('Episode Returns')
        self.axes['episode_return'].set_xlabel('Time')
        self.axes['episode_return'].set_ylabel('Return')
        self.lines['episode_return'], = self.axes['episode_return'].plot([], [], 'b-')
        
        # Success rate
        self.axes['success_rate'].set_title('Success Rate')
        self.axes['success_rate'].set_xlabel('Time')
        self.axes['success_rate'].set_ylabel('Rate (%)')
        self.axes['success_rate'].set_ylim(0, 100)
        self.lines['success_rate'], = self.axes['success_rate'].plot([], [], 'g-')
        
        # Safety violations
        self.axes['safety_violations'].set_title('Safety Violations')
        self.axes['safety_violations'].set_xlabel('Time')
        self.axes['safety_violations'].set_ylabel('Count')
        self.lines['safety_violations'], = self.axes['safety_violations'].plot([], [], 'r-')
        
        # Action distribution
        self.axes['action_distribution'].set_title('Action Distribution')
        self.axes['action_distribution'].set_xlabel('Action')
        self.axes['action_distribution'].set_ylabel('Frequency')
        
        # State coverage heatmap
        self.axes['state_coverage'].set_title('State Space Coverage')
        
        # System health
        self.axes['system_health'].set_title('System Health')
        self.axes['system_health'].axis('off')
        
        # Learning progress
        self.axes['learning_progress'].set_title('Learning Progress')
        self.axes['learning_progress'].set_xlabel('Training Steps')
        self.axes['learning_progress'].set_ylabel('Performance')
        
        # Alerts
        self.axes['alerts'].set_title('System Alerts')
        self.axes['alerts'].axis('off')
        
    def update_dashboard(self, frame):
        """Update dashboard with latest data."""
        # Update episode returns
        self._update_time_series('episode_return', 'episode_return')
        
        # Update success rate
        self._update_time_series('success_rate', 'success_rate', scale=100)
        
        # Update safety violations
        self._update_time_series('safety_violations', 'safety_violations')
        
        # Update action distribution
        self._update_action_distribution()
        
        # Update state coverage
        self._update_state_coverage()
        
        # Update system health
        self._update_system_health()
        
        # Update learning progress
        self._update_learning_progress()
        
        # Update alerts
        self._update_alerts()
        
        return list(self.lines.values())
    
    def _update_time_series(self, metric_name: str, line_key: str, scale: float = 1.0):
        """Update a time series plot."""
        timestamps, values = self.metrics_collector.get_time_series(metric_name)
        
        if timestamps and values:
            # Convert timestamps to relative seconds
            start_time = timestamps[0]
            x_data = [(ts - start_time).total_seconds() for ts in timestamps]
            y_data = [v * scale for v in values]
            
            self.lines[line_key].set_data(x_data, y_data)
            self.axes[line_key].relim()
            self.axes[line_key].autoscale_view()
    
    def _update_action_distribution(self):
        """Update action distribution histogram."""
        # Get action data
        _, actions = self.metrics_collector.get_time_series('action_taken')
        
        if actions:
            self.axes['action_distribution'].clear()
            self.axes['action_distribution'].hist(actions, bins=20, alpha=0.7, color='blue')
            self.axes['action_distribution'].set_title('Action Distribution')
            self.axes['action_distribution'].set_xlabel('Action')
            self.axes['action_distribution'].set_ylabel('Frequency')
    
    def _update_state_coverage(self):
        """Update state space coverage heatmap."""
        # Simulate state coverage data (in practice, this would come from the RL system)
        coverage_data = np.random.rand(10, 10)
        
        self.axes['state_coverage'].clear()
        sns.heatmap(
            coverage_data,
            ax=self.axes['state_coverage'],
            cmap='YlOrRd',
            cbar_kws={'label': 'Visit Count'}
        )
        self.axes['state_coverage'].set_title('State Space Coverage')
    
    def _update_system_health(self):
        """Update system health indicators."""
        self.axes['system_health'].clear()
        self.axes['system_health'].axis('off')
        
        # Get system metrics
        metrics = {
            'CPU Usage': self.metrics_collector.get_aggregates('cpu_usage').get('mean', 0),
            'Memory Usage': self.metrics_collector.get_aggregates('memory_usage').get('mean', 0),
            'Inference Latency': self.metrics_collector.get_aggregates('inference_latency').get('mean', 0),
            'Queue Length': self.metrics_collector.get_aggregates('queue_length').get('mean', 0)
        }
        
        # Display metrics
        y_pos = 0.9
        for metric, value in metrics.items():
            color = 'green' if value < 80 else 'orange' if value < 90 else 'red'
            self.axes['system_health'].text(
                0.1, y_pos, f"{metric}: {value:.1f}%",
                transform=self.axes['system_health'].transAxes,
                fontsize=12, color=color
            )
            y_pos -= 0.2
    
    def _update_learning_progress(self):
        """Update learning progress chart."""
        # Get training metrics
        _, train_losses = self.metrics_collector.get_time_series('training_loss')
        _, val_scores = self.metrics_collector.get_time_series('validation_score')
        
        self.axes['learning_progress'].clear()
        
        if train_losses:
            steps = list(range(len(train_losses)))
            self.axes['learning_progress'].plot(steps, train_losses, 'b-', label='Training Loss')
        
        if val_scores:
            steps = list(range(len(val_scores)))
            self.axes['learning_progress'].plot(steps, val_scores, 'g-', label='Validation Score')
        
        self.axes['learning_progress'].set_title('Learning Progress')
        self.axes['learning_progress'].set_xlabel('Training Steps')
        self.axes['learning_progress'].set_ylabel('Metric Value')
        self.axes['learning_progress'].legend()
    
    def _update_alerts(self):
        """Update system alerts."""
        self.axes['alerts'].clear()
        self.axes['alerts'].axis('off')
        
        # Get recent alerts
        alerts = [
            ('WARNING', 'High latency detected', 'orange'),
            ('INFO', 'Policy updated to v2.3', 'blue'),
            ('ERROR', 'Safety violation in zone A', 'red'),
            ('INFO', 'Checkpoint saved', 'green')
        ]
        
        y_pos = 0.9
        for level, message, color in alerts[:5]:  # Show only recent 5
            self.axes['alerts'].text(
                0.05, y_pos,
                f"[{level}] {message}",
                transform=self.axes['alerts'].transAxes,
                fontsize=10,
                color=color,
                family='monospace'
            )
            y_pos -= 0.15
    
    def start_monitoring(self, update_interval: int = 1000):
        """Start the monitoring dashboard."""
        self.setup_dashboard()
        
        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self.update_dashboard,
            interval=update_interval,
            blit=False
        )
        
        plt.show()


def simulate_metrics(metrics_collector: MetricsCollector):
    """Simulate metrics for demonstration."""
    metric_configs = [
        ('episode_return', lambda: np.random.normal(100, 20)),
        ('success_rate', lambda: min(1.0, max(0, np.random.normal(0.95, 0.05)))),
        ('safety_violations', lambda: max(0, int(np.random.poisson(0.5)))),
        ('cpu_usage', lambda: min(100, max(0, np.random.normal(60, 15)))),
        ('memory_usage', lambda: min(100, max(0, np.random.normal(70, 10)))),
        ('inference_latency', lambda: max(0, np.random.normal(50, 10))),
        ('queue_length', lambda: max(0, int(np.random.normal(10, 5)))),
        ('training_loss', lambda: max(0, np.random.exponential(0.1))),
        ('validation_score', lambda: min(1.0, max(0, np.random.normal(0.8, 0.1)))),
        ('action_taken', lambda: int(np.random.choice([0, 1, 2, 3])))
    ]
    
    while True:
        for metric_name, generator in metric_configs:
            metrics_collector.add_metric(metric_name, generator())
        
        time.sleep(0.1)  # Simulate 10Hz updates


if __name__ == "__main__":
    # Create metrics collector
    collector = MetricsCollector(window_minutes=10)
    
    # Start metric simulation
    sim_thread = threading.Thread(target=simulate_metrics, args=(collector,))
    sim_thread.daemon = True
    sim_thread.start()
    
    # Create and start dashboard
    dashboard = RLDashboard(collector)
    
    print("Starting RL Monitoring Dashboard...")
    print("This dashboard shows real-time metrics for a production RL system")
    print("Including performance, safety, system health, and learning progress")
    
    dashboard.start_monitoring(update_interval=1000)  # Update every second