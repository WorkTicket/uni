"""Self-model module (Research Program D5).

Provides SelfModel and BodySchema for modeling an agent's own body, actions,
capabilities, and sensorimotor effects.
"""

from .model import SelfModel, BodySchema, make_default_self_model

__all__ = ["SelfModel", "BodySchema", "make_default_self_model"]
