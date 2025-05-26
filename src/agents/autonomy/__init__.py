"""
LazyCoder Autonomy Module
Manages autonomous operation levels and user confirmation workflows.
"""

from .autonomy_controller import AutonomyController, AutonomyLevel, ActionClassification

__all__ = [
    "AutonomyController",
    "AutonomyLevel", 
    "ActionClassification"
]