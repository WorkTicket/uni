"""Global workspace module (Research Program D6).

Provides GlobalWorkspace and WorkspaceAgentController (ECS-IW architecture).
"""

from .model import GlobalWorkspace, WorkspaceContent, WorkspaceAgentController

__all__ = ["GlobalWorkspace", "WorkspaceContent", "WorkspaceAgentController"]
