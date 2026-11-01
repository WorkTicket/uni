"""World-model learning module (Research Program D3).

Compares reactive vs recurrent world-model controllers in embodied environments.
"""

from .experiment import (
    WorldModelExperimentResult, run_episode, run_comparison_experiment,
    make_world_model_manifest,
)

__all__ = [
    "WorldModelExperimentResult", "run_episode", "run_comparison_experiment",
    "make_world_model_manifest",
]
