"""
FreEco.ai Platform - Stability Module
Enhanced OpenManus with error handling, stability, and adaptation

This module provides comprehensive stability features:
- Intelligent retry mechanisms with exponential backoff
- Graceful degradation with feature fallbacks
- Error recovery with state management and rollback

Part of Enhancement #3: Error Handling, Stability & Adaptation
"""

from .retry_manager import (
    RetryManager,
    RetryConfig,
    RetryStats,
    RetryStrategy,
    retry,
    default_retry_manager,
)

from .degradation import (
    GracefulDegradation,
    QualityLevel,
    HealthStatus,
    ServiceProvider,
    Feature,
    DegradationEvent,
    HealthMetrics,
    default_degradation,
)

from .error_recovery import (
    ErrorRecoverySystem,
    RecoveryStrategy,
    State,
    Operation,
    ErrorPattern,
    Recovery,
    default_recovery,
)

__all__ = [
    # Retry Manager
    "RetryManager",
    "RetryConfig",
    "RetryStats",
    "RetryStrategy",
    "retry",
    "default_retry_manager",
    
    # Graceful Degradation
    "GracefulDegradation",
    "QualityLevel",
    "HealthStatus",
    "ServiceProvider",
    "Feature",
    "DegradationEvent",
    "HealthMetrics",
    "default_degradation",
    
    # Error Recovery
    "ErrorRecoverySystem",
    "RecoveryStrategy",
    "State",
    "Operation",
    "ErrorPattern",
    "Recovery",
    "default_recovery",
]

