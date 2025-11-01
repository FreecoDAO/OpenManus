"""
FreEco.ai Platform - Performance Module
Enhanced OpenManus with performance optimization
"""

from .optimizer import PerformanceOptimizer, default_optimizer
from .monitoring import MonitoringSystem, default_monitor

__all__ = [
    "PerformanceOptimizer",
    "default_optimizer",
    "MonitoringSystem",
    "default_monitor",
]

