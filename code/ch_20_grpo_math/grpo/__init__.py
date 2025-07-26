"""
Group Relative Policy Optimization (GRPO) implementation.
Inspired by DeepSeek's approach for training reasoning models.
"""

from .trainer import GRPOTrainer
from .rewards import MathRewardFunction, CodeRewardFunction, FormatReward
from .data import MathDataset, CodeDataset, create_data_loader
from .utils import (
    extract_thinking,
    extract_answer,
    check_format_compliance,
    compute_group_statistics
)

__all__ = [
    'GRPOTrainer',
    'MathRewardFunction',
    'CodeRewardFunction',
    'FormatReward',
    'MathDataset',
    'CodeDataset',
    'create_data_loader',
    'extract_thinking',
    'extract_answer',
    'check_format_compliance',
    'compute_group_statistics'
]