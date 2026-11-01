"""Environment modules for Project Genesis research.

Provides minimal embodied environments for testing agent controllers.
"""

from .gridworld import GridWorld, GridWorldConfig, PatchState

__all__ = ["GridWorld", "GridWorldConfig", "PatchState"]
