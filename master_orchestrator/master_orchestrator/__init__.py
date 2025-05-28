"""Master Orchestrator - Agentic Multi-Project System."""

__version__ = "0.1.0"

from .core import MasterOrchestrator
from .agents import Agent, AgentFramework
from .knowledge_graph import KnowledgeGraph
from .config import OrchestratorConfig

__all__ = [
    "MasterOrchestrator",
    "Agent",
    "AgentFramework", 
    "KnowledgeGraph",
    "OrchestratorConfig",
]