"""PINN-Vanila public API."""

from .case import BendGeometry, FlowCase, FluidProperties, ProjectConfig
from .model import PINN, PredictionResult

__all__ = [
    "BendGeometry",
    "FlowCase",
    "FluidProperties",
    "PINN",
    "PredictionResult",
    "ProjectConfig",
]

__version__ = "0.1.0"
