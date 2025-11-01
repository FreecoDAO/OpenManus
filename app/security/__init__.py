"""
FreEco.ai Platform - Security Module
Enhanced OpenManus with military-grade security
"""

from .security_manager import SecurityManager, default_security
from .anti_hacking import AntiHackingSystem, default_anti_hacking

__all__ = [
    "SecurityManager",
    "default_security",
    "AntiHackingSystem",
    "default_anti_hacking",
]

