"""
FreEco.ai Platform - Ethics Module
Enhanced OpenManus with comprehensive ethical framework
"""

from .freeco_laws import FreEcoLawsEnforcer, default_freeco_laws, FreEcoBenchmark
from .ecological_principles import EcologicalSystem, default_ecological

__all__ = [
    "FreEcoLawsEnforcer",
    "default_freeco_laws",
    "FreEcoBenchmark",
    "EcologicalSystem",
    "default_ecological",
]

