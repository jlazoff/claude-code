"""Core Master Orchestrator System."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import structlog
from pydantic import BaseModel, Field

from .config import OrchestratorConfig
from .knowledge_graph import KnowledgeGraph
from .agents import AgentFramework
from .infrastructure import InfrastructureManager
from .repository_manager import RepositoryManager

logger = structlog.get_logger()


class SystemStatus(BaseModel):
    """System status model."""
    
    status: str = Field(description="Overall system status")
    uptime: str = Field(description="System uptime")
    active_agents: int = Field(description="Number of active agents")
    repositories_connected: int = Field(description="Connected repositories")
    hardware_nodes: int = Field(description="Connected hardware nodes")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class MasterOrchestrator:
    """
    Master Orchestrator - Central coordination system for agentic multi-project management.
    
    Coordinates 28+ AI repositories, manages distributed infrastructure,
    and provides unified interface for all operations.
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.logger = structlog.get_logger("master_orchestrator")
        
        # Core Components
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        self.agent_framework: Optional[AgentFramework] = None
        self.infrastructure: Optional[InfrastructureManager] = None
        self.repository_manager: Optional[RepositoryManager] = None
        
        # State
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
    async def initialize(self) -> None:
        """Initialize all core components."""
        self.logger.info("Initializing Master Orchestrator")
        
        try:
            # Initialize Knowledge Graph
            self.knowledge_graph = KnowledgeGraph(self.config.arangodb_config)
            await self.knowledge_graph.initialize()
            
            # Initialize Agent Framework
            self.agent_framework = AgentFramework(
                self.config.agent_config,
                self.knowledge_graph
            )
            await self.agent_framework.initialize()
            
            # Initialize Infrastructure Manager
            self.infrastructure = InfrastructureManager(self.config.infrastructure_config)
            await self.infrastructure.initialize()
            
            # Initialize Repository Manager
            self.repository_manager = RepositoryManager(
                self.config.repository_config,
                self.knowledge_graph
            )
            await self.repository_manager.initialize()
            
            self.logger.info("Master Orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Master Orchestrator: {e}")
            raise
    
    async def start(self) -> None:
        """Start the orchestrator system."""
        if self.is_running:
            self.logger.warning("Master Orchestrator already running")
            return
            
        self.logger.info("Starting Master Orchestrator")
        self.start_time = datetime.utcnow()
        self.is_running = True
        
        # Start core monitoring loops
        self.active_tasks["monitor_system"] = asyncio.create_task(self._monitor_system())
        self.active_tasks["process_workflows"] = asyncio.create_task(self._process_workflows())
        self.active_tasks["optimize_resources"] = asyncio.create_task(self._optimize_resources())
        
        # Start repository monitoring
        if self.repository_manager:
            self.active_tasks["monitor_repositories"] = asyncio.create_task(
                self.repository_manager.monitor_repositories()
            )
        
        # Start agent tasks
        if self.agent_framework:
            self.active_tasks["run_agents"] = asyncio.create_task(
                self.agent_framework.run_continuous()
            )
        
        self.logger.info("Master Orchestrator started successfully")
    
    async def stop(self) -> None:
        """Stop the orchestrator system."""
        self.logger.info("Stopping Master Orchestrator")
        self.is_running = False
        
        # Cancel all active tasks
        for task_name, task in self.active_tasks.items():
            self.logger.info(f"Cancelling task: {task_name}")
            task.cancel()
            
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Cleanup components
        if self.agent_framework:
            await self.agent_framework.shutdown()
        if self.knowledge_graph:
            await self.knowledge_graph.shutdown()
        if self.infrastructure:
            await self.infrastructure.shutdown()
        if self.repository_manager:
            await self.repository_manager.shutdown()
        
        self.active_tasks.clear()
        self.logger.info("Master Orchestrator stopped")
    
    async def get_status(self) -> SystemStatus:
        """Get current system status."""
        uptime = "0m"
        if self.start_time:
            delta = datetime.utcnow() - self.start_time
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes, _ = divmod(remainder, 60)
            uptime = f"{hours}h {minutes}m"
        
        active_agents = 0
        if self.agent_framework:
            active_agents = len(self.agent_framework.active_agents)
        
        repositories_connected = 0
        if self.repository_manager:
            repositories_connected = len(self.repository_manager.connected_repositories)
        
        hardware_nodes = 0
        if self.infrastructure:
            hardware_nodes = len(self.infrastructure.connected_nodes)
        
        return SystemStatus(
            status="running" if self.is_running else "stopped",
            uptime=uptime,
            active_agents=active_agents,
            repositories_connected=repositories_connected,
            hardware_nodes=hardware_nodes
        )
    
    async def execute_command(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a system command."""
        self.logger.info(f"Executing command: {command}", parameters=parameters)
        
        try:
            if command == "analyze_repository":
                return await self._analyze_repository(parameters)
            elif command == "create_agent":
                return await self._create_agent(parameters)
            elif command == "run_workflow":
                return await self._run_workflow(parameters)
            elif command == "optimize_system":
                return await self._optimize_system(parameters)
            else:
                raise ValueError(f"Unknown command: {command}")
                
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _monitor_system(self) -> None:
        """Continuous system monitoring loop."""
        while self.is_running:
            try:
                # Check system health
                status = await self.get_status()
                self.logger.debug("System status check", status=status.model_dump())
                
                # Update knowledge graph with system metrics
                if self.knowledge_graph:
                    await self.knowledge_graph.update_system_metrics(status)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _process_workflows(self) -> None:
        """Process workflow queue."""
        while self.is_running:
            try:
                # Check for new workflows from knowledge graph
                if self.knowledge_graph:
                    workflows = await self.knowledge_graph.get_pending_workflows()
                    for workflow in workflows:
                        await self._execute_workflow(workflow)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Workflow processing error: {e}")
                await asyncio.sleep(30)
    
    async def _optimize_resources(self) -> None:
        """Continuous resource optimization."""
        while self.is_running:
            try:
                # Optimize infrastructure resources
                if self.infrastructure:
                    await self.infrastructure.optimize_resources()
                
                # Optimize agent allocation
                if self.agent_framework:
                    await self.agent_framework.optimize_allocation()
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Resource optimization error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def _analyze_repository(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a repository."""
        if not self.repository_manager:
            raise RuntimeError("Repository manager not initialized")
        
        repo_path = parameters.get("path")
        if not repo_path:
            raise ValueError("Repository path required")
        
        analysis = await self.repository_manager.analyze_repository(Path(repo_path))
        return {"success": True, "analysis": analysis}
    
    async def _create_agent(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new agent."""
        if not self.agent_framework:
            raise RuntimeError("Agent framework not initialized")
        
        agent_type = parameters.get("type")
        agent_config = parameters.get("config", {})
        
        agent = await self.agent_framework.create_agent(agent_type, agent_config)
        return {"success": True, "agent_id": agent.id}
    
    async def _run_workflow(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a workflow."""
        workflow_id = parameters.get("workflow_id")
        if not workflow_id:
            raise ValueError("Workflow ID required")
        
        # Implementation depends on workflow system
        return {"success": True, "workflow_id": workflow_id, "status": "started"}
    
    async def _optimize_system(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system performance."""
        # Trigger immediate optimization
        if self.infrastructure:
            await self.infrastructure.optimize_resources()
        if self.agent_framework:
            await self.agent_framework.optimize_allocation()
        
        return {"success": True, "message": "System optimization triggered"}
    
    async def _execute_workflow(self, workflow: Dict[str, Any]) -> None:
        """Execute a specific workflow."""
        self.logger.info(f"Executing workflow: {workflow.get('id')}")
        # Implementation depends on workflow definition
        pass