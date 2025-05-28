"""DSPY-based Agentic Framework for Master Orchestrator."""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, Union
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import structlog
import dspy
from pydantic import BaseModel, Field

from .config import AgentConfig
from .knowledge_graph import KnowledgeGraph

logger = structlog.get_logger()


class AgentTask(BaseModel):
    """Agent task model."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = Field(description="Type of task")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, description="Task priority (1-10)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    status: str = Field(default="pending")  # pending, assigned, running, completed, failed
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = Field(default=None)


class AgentCapability(BaseModel):
    """Agent capability definition."""
    
    name: str = Field(description="Capability name")
    description: str = Field(description="Capability description")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True)


class Agent(ABC):
    """
    Base Agent class using DSPY for programmatic prompt engineering.
    
    All agents inherit from this class and implement specific capabilities
    using DSPY modules for consistent, optimizable performance.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        knowledge_graph: KnowledgeGraph,
        llm_config: Dict[str, Any]
    ):
        self.id = agent_id
        self.name = name
        self.knowledge_graph = knowledge_graph
        self.llm_config = llm_config
        self.logger = structlog.get_logger(f"agent.{self.name}")
        
        # Agent state
        self.status = "idle"  # idle, busy, error, offline
        self.current_task: Optional[AgentTask] = None
        self.capabilities: List[AgentCapability] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # DSPY setup
        self.dspy_modules: Dict[str, dspy.Module] = {}
        self._setup_dspy()
    
    def _setup_dspy(self) -> None:
        """Setup DSPY modules for this agent."""
        # Configure DSPY with the appropriate LLM
        if self.llm_config.get("provider") == "openai":
            lm = dspy.OpenAI(
                model=self.llm_config.get("model", "gpt-4"),
                api_key=self.llm_config.get("api_key"),
                max_tokens=self.llm_config.get("max_tokens", 4096)
            )
        elif self.llm_config.get("provider") == "anthropic":
            # Custom Anthropic implementation would go here
            # For now, fallback to OpenAI-compatible interface
            lm = dspy.OpenAI(
                model=self.llm_config.get("model", "claude-3-sonnet-20241022"),
                api_key=self.llm_config.get("api_key"),
                max_tokens=self.llm_config.get("max_tokens", 4096)
            )
        else:
            # Default to OpenAI
            lm = dspy.OpenAI(
                model="gpt-3.5-turbo",
                max_tokens=2048
            )
        
        dspy.settings.configure(lm=lm)
        
        # Initialize core DSPY modules
        self._initialize_dspy_modules()
    
    @abstractmethod
    def _initialize_dspy_modules(self) -> None:
        """Initialize DSPY modules specific to this agent type."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task."""
        pass
    
    async def assign_task(self, task: AgentTask) -> bool:
        """Assign a task to this agent."""
        if self.status != "idle":
            self.logger.warning(f"Agent {self.name} is not idle, cannot assign task")
            return False
        
        self.current_task = task
        task.assigned_at = datetime.utcnow()
        task.status = "assigned"
        self.status = "busy"
        
        self.logger.info(f"Task assigned to agent {self.name}", task_id=task.id)
        return True
    
    async def run_task(self) -> Optional[Dict[str, Any]]:
        """Execute the currently assigned task."""
        if not self.current_task:
            return None
        
        task = self.current_task
        task.status = "running"
        
        try:
            self.logger.info(f"Starting task execution", task_id=task.id)
            result = await self.execute_task(task)
            
            task.result = result
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            
            self.logger.info(f"Task completed successfully", task_id=task.id)
            return result
            
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            
            self.logger.error(f"Task execution failed: {e}", task_id=task.id)
            raise
        
        finally:
            self.current_task = None
            self.status = "idle"
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        return self.capabilities
    
    async def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update agent performance metrics."""
        self.performance_metrics.update(metrics)
        
        # Update knowledge graph
        await self.knowledge_graph.update_system_metrics({
            "agent_id": self.id,
            "metrics": self.performance_metrics,
            "timestamp": datetime.utcnow()
        })


class RepositoryAnalysisSignature(dspy.Signature):
    """DSPY signature for repository analysis."""
    
    repository_path = dspy.InputField(desc="Path to the repository to analyze")
    file_listing = dspy.InputField(desc="List of files in the repository")
    readme_content = dspy.InputField(desc="Content of the README file")
    
    analysis = dspy.OutputField(desc="Structured analysis of the repository including technologies, capabilities, and architecture insights")


class RepositoryAnalysisAgent(Agent):
    """Agent specialized in analyzing code repositories."""
    
    def _initialize_dspy_modules(self) -> None:
        """Initialize repository analysis DSPY modules."""
        
        class RepositoryAnalyzer(dspy.Module):
            def __init__(self):
                super().__init__()
                self.analyze = dspy.ChainOfThought(RepositoryAnalysisSignature)
            
            def forward(self, repository_path, file_listing, readme_content):
                return self.analyze(
                    repository_path=repository_path,
                    file_listing=file_listing,
                    readme_content=readme_content
                )
        
        self.dspy_modules["repository_analyzer"] = RepositoryAnalyzer()
        
        # Add capabilities
        self.capabilities = [
            AgentCapability(
                name="analyze_repository",
                description="Analyze code repositories for technologies, capabilities, and architecture"
            ),
            AgentCapability(
                name="extract_dependencies",
                description="Extract and analyze project dependencies"
            ),
            AgentCapability(
                name="assess_complexity",
                description="Assess code complexity and maintainability"
            )
        ]
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute repository analysis task."""
        if task.task_type == "analyze_repository":
            return await self._analyze_repository(task.parameters)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _analyze_repository(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a repository using DSPY."""
        repo_path = Path(parameters["repository_path"])
        
        # Gather repository information
        file_listing = []
        readme_content = ""
        
        try:
            # Get file listing
            for file_path in repo_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    relative_path = file_path.relative_to(repo_path)
                    file_listing.append(str(relative_path))
            
            # Read README if exists
            for readme_name in ["README.md", "README.txt", "README"]:
                readme_path = repo_path / readme_name
                if readme_path.exists():
                    try:
                        readme_content = readme_path.read_text(encoding='utf-8')
                        break
                    except:
                        continue
            
            # Use DSPY module for analysis
            analyzer = self.dspy_modules["repository_analyzer"]
            result = analyzer(
                repository_path=str(repo_path),
                file_listing="\n".join(file_listing[:100]),  # Limit for token efficiency
                readme_content=readme_content[:2000]  # Limit for token efficiency
            )
            
            return {
                "repository_path": str(repo_path),
                "analysis": result.analysis,
                "file_count": len(file_listing),
                "has_readme": bool(readme_content),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Repository analysis failed: {e}")
            raise


class TaskOrchestrationSignature(dspy.Signature):
    """DSPY signature for task orchestration."""
    
    available_agents = dspy.InputField(desc="List of available agents and their capabilities")
    pending_tasks = dspy.InputField(desc="List of pending tasks to be assigned")
    system_state = dspy.InputField(desc="Current system state and resource availability")
    
    task_assignments = dspy.OutputField(desc="Optimal task assignments for agents based on capabilities and system state")


class TaskOrchestratorAgent(Agent):
    """Agent responsible for orchestrating tasks across the system."""
    
    def _initialize_dspy_modules(self) -> None:
        """Initialize task orchestration DSPY modules."""
        
        class TaskOrchestrator(dspy.Module):
            def __init__(self):
                super().__init__()
                self.orchestrate = dspy.ChainOfThought(TaskOrchestrationSignature)
            
            def forward(self, available_agents, pending_tasks, system_state):
                return self.orchestrate(
                    available_agents=available_agents,
                    pending_tasks=pending_tasks,
                    system_state=system_state
                )
        
        self.dspy_modules["task_orchestrator"] = TaskOrchestrator()
        
        self.capabilities = [
            AgentCapability(
                name="orchestrate_tasks",
                description="Orchestrate and assign tasks to optimal agents"
            ),
            AgentCapability(
                name="optimize_resource_allocation",
                description="Optimize resource allocation across the system"
            )
        ]
    
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute task orchestration."""
        if task.task_type == "orchestrate_tasks":
            return await self._orchestrate_tasks(task.parameters)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _orchestrate_tasks(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate task assignments using DSPY."""
        # This would integrate with the AgentFramework to get real data
        orchestrator = self.dspy_modules["task_orchestrator"]
        
        result = orchestrator(
            available_agents=str(parameters.get("available_agents", [])),
            pending_tasks=str(parameters.get("pending_tasks", [])),
            system_state=str(parameters.get("system_state", {}))
        )
        
        return {
            "task_assignments": result.task_assignments,
            "timestamp": datetime.utcnow().isoformat()
        }


class AgentFramework:
    """
    DSPY-based Agent Framework for managing multiple agents.
    
    Provides centralized agent management, task distribution,
    and performance optimization using programmatic prompt engineering.
    """
    
    def __init__(self, config: AgentConfig, knowledge_graph: KnowledgeGraph):
        self.config = config
        self.knowledge_graph = knowledge_graph
        self.logger = structlog.get_logger("agent_framework")
        
        # Agent management
        self.agents: Dict[str, Agent] = {}
        self.task_queue: List[AgentTask] = []
        self.active_agents: Dict[str, Agent] = {}
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        # DSPY configuration
        self._setup_dspy_cache()
    
    def _setup_dspy_cache(self) -> None:
        """Setup DSPY caching for performance."""
        cache_dir = self.config.dspy_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure DSPY settings
        dspy.settings.configure(
            experimental=True,
            bypass_frequency_threshold=1,
            bypass_coverage_threshold=1
        )
    
    async def initialize(self) -> None:
        """Initialize the agent framework."""
        self.logger.info("Initializing Agent Framework")
        
        # Create default agents
        await self._create_default_agents()
        
        self.logger.info(f"Agent Framework initialized with {len(self.agents)} agents")
    
    async def _create_default_agents(self) -> None:
        """Create default system agents."""
        # Repository Analysis Agent
        repo_agent = RepositoryAnalysisAgent(
            agent_id=str(uuid.uuid4()),
            name="repository_analyzer",
            knowledge_graph=self.knowledge_graph,
            llm_config=self._get_llm_config()
        )
        self.agents[repo_agent.id] = repo_agent
        
        # Task Orchestrator Agent
        orchestrator_agent = TaskOrchestratorAgent(
            agent_id=str(uuid.uuid4()),
            name="task_orchestrator", 
            knowledge_graph=self.knowledge_graph,
            llm_config=self._get_llm_config()
        )
        self.agents[orchestrator_agent.id] = orchestrator_agent
    
    def _get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration for agents."""
        # Use first available LLM provider
        if self.config.llm_providers:
            provider = self.config.llm_providers[0]
            return {
                "provider": provider.name,
                "model": provider.model,
                "api_key": provider.api_key,
                "max_tokens": provider.max_tokens
            }
        
        # Fallback configuration
        return {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "max_tokens": 2048
        }
    
    async def create_agent(self, agent_type: str, config: Dict[str, Any]) -> Agent:
        """Create a new agent of specified type."""
        agent_id = str(uuid.uuid4())
        
        if agent_type == "repository_analyzer":
            agent = RepositoryAnalysisAgent(
                agent_id=agent_id,
                name=config.get("name", f"repo_agent_{agent_id[:8]}"),
                knowledge_graph=self.knowledge_graph,
                llm_config=self._get_llm_config()
            )
        elif agent_type == "task_orchestrator":
            agent = TaskOrchestratorAgent(
                agent_id=agent_id,
                name=config.get("name", f"orchestrator_{agent_id[:8]}"),
                knowledge_graph=self.knowledge_graph,
                llm_config=self._get_llm_config()
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.agents[agent_id] = agent
        
        # Add to knowledge graph
        await self.knowledge_graph.create_agent_node({
            "name": agent.name,
            "type": agent_type,
            "capabilities": [cap.name for cap in agent.capabilities]
        })
        
        self.logger.info(f"Created agent: {agent.name}", agent_id=agent_id)
        return agent
    
    async def add_task(self, task: AgentTask) -> None:
        """Add a task to the queue."""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        
        self.logger.info(f"Added task to queue", task_id=task.id, task_type=task.task_type)
    
    async def run_continuous(self) -> None:
        """Run continuous agent processing loop."""
        self.logger.info("Starting continuous agent processing")
        
        while True:
            try:
                await self._process_task_queue()
                await asyncio.sleep(1)  # Short delay between processing cycles
                
            except Exception as e:
                self.logger.error(f"Error in continuous processing: {e}")
                await asyncio.sleep(5)  # Longer delay on error
    
    async def _process_task_queue(self) -> None:
        """Process pending tasks and assign to available agents."""
        if not self.task_queue:
            return
        
        # Find available agents
        available_agents = [
            agent for agent in self.agents.values()
            if agent.status == "idle"
        ]
        
        if not available_agents:
            return
        
        # Assign tasks to agents
        tasks_to_remove = []
        
        for task in self.task_queue:
            if not available_agents:
                break
            
            # Find best agent for this task
            best_agent = self._find_best_agent(task, available_agents)
            if best_agent:
                success = await best_agent.assign_task(task)
                if success:
                    available_agents.remove(best_agent)
                    tasks_to_remove.append(task)
                    
                    # Start task execution
                    asyncio.create_task(self._execute_agent_task(best_agent))
        
        # Remove assigned tasks from queue
        for task in tasks_to_remove:
            self.task_queue.remove(task)
    
    def _find_best_agent(self, task: AgentTask, available_agents: List[Agent]) -> Optional[Agent]:
        """Find the best agent for a specific task."""
        # Simple capability matching for now
        # Could be enhanced with DSPY-based optimization
        
        for agent in available_agents:
            agent_capabilities = [cap.name for cap in agent.capabilities]
            if task.task_type in agent_capabilities:
                return agent
        
        return None
    
    async def _execute_agent_task(self, agent: Agent) -> None:
        """Execute a task on an agent."""
        try:
            result = await agent.run_task()
            if result:
                self.logger.info(f"Agent task completed", agent_name=agent.name)
        except Exception as e:
            self.logger.error(f"Agent task failed: {e}", agent_name=agent.name)
    
    async def optimize_allocation(self) -> None:
        """Optimize agent allocation and performance."""
        # This could use DSPY for optimization strategies
        self.logger.debug("Optimizing agent allocation")
        
        # Collect performance metrics
        metrics = {}
        for agent_id, agent in self.agents.items():
            metrics[agent_id] = agent.performance_metrics
        
        # Store in performance history
        self.performance_history.append({
            "timestamp": datetime.utcnow(),
            "metrics": metrics
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    async def shutdown(self) -> None:
        """Shutdown the agent framework."""
        self.logger.info("Shutting down Agent Framework")
        
        # Clear task queue
        self.task_queue.clear()
        
        # Clean up agents
        self.agents.clear()
        self.active_agents.clear()
        
        self.logger.info("Agent Framework shutdown complete")