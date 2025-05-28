"""FastAPI Web Interface for Master Orchestrator."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import structlog
from pydantic import BaseModel

from .core import MasterOrchestrator, SystemStatus
from .config import OrchestratorConfig
from .agents import AgentTask

logger = structlog.get_logger()

# Global orchestrator instance
orchestrator: Optional[MasterOrchestrator] = None


class TaskRequest(BaseModel):
    """Task creation request model."""
    
    task_type: str
    parameters: Dict[str, Any] = {}
    priority: int = 5


class AgentRequest(BaseModel):
    """Agent creation request model."""
    
    agent_type: str
    name: Optional[str] = None
    config: Dict[str, Any] = {}


class CommandRequest(BaseModel):
    """Command execution request model."""
    
    command: str
    parameters: Dict[str, Any] = {}


def create_app(config: OrchestratorConfig) -> FastAPI:
    """Create FastAPI application."""
    
    app = FastAPI(
        title="Master Orchestrator",
        description="Agentic Multi-Project Orchestration System",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount static files
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    # Initialize orchestrator
    async def get_orchestrator() -> MasterOrchestrator:
        global orchestrator
        if orchestrator is None:
            orchestrator = MasterOrchestrator(config)
            await orchestrator.initialize()
            await orchestrator.start()
        return orchestrator
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize orchestrator on startup."""
        await get_orchestrator()
        logger.info("Master Orchestrator API started")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        global orchestrator
        if orchestrator:
            await orchestrator.stop()
        logger.info("Master Orchestrator API shutdown")
    
    # Root endpoint - serve dashboard
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the main dashboard."""
        dashboard_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Master Orchestrator Dashboard</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
            <style>
                [x-cloak] { display: none !important; }
            </style>
        </head>
        <body class="bg-gray-100 min-h-screen">
            <div x-data="dashboard()" x-init="init()" class="container mx-auto px-4 py-8">
                <!-- Header -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <div class="flex items-center justify-between">
                        <div>
                            <h1 class="text-3xl font-bold text-gray-800">🚀 Master Orchestrator</h1>
                            <p class="text-gray-600 mt-2">Agentic Multi-Project Orchestration System</p>
                        </div>
                        <div class="flex items-center space-x-4">
                            <div class="text-right">
                                <div class="text-sm text-gray-500">Status</div>
                                <div class="font-semibold" :class="status.status === 'running' ? 'text-green-600' : 'text-red-600'" x-text="status.status"></div>
                            </div>
                            <div class="text-right">
                                <div class="text-sm text-gray-500">Uptime</div>
                                <div class="font-semibold text-gray-800" x-text="status.uptime"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- System Overview -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="flex items-center">
                            <div class="text-3xl">🤖</div>
                            <div class="ml-4">
                                <div class="text-2xl font-bold text-gray-800" x-text="status.active_agents"></div>
                                <div class="text-sm text-gray-500">Active Agents</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="flex items-center">
                            <div class="text-3xl">📁</div>
                            <div class="ml-4">
                                <div class="text-2xl font-bold text-gray-800" x-text="status.repositories_connected"></div>
                                <div class="text-sm text-gray-500">Repositories</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="flex items-center">
                            <div class="text-3xl">🖥️</div>
                            <div class="ml-4">
                                <div class="text-2xl font-bold text-gray-800" x-text="status.hardware_nodes"></div>
                                <div class="text-sm text-gray-500">Hardware Nodes</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <div class="flex items-center">
                            <div class="text-3xl">📊</div>
                            <div class="ml-4">
                                <div class="text-2xl font-bold text-gray-800" x-text="tasks.length"></div>
                                <div class="text-sm text-gray-500">Active Tasks</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Actions Panel -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">Quick Actions</h2>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <button @click="analyzeRepo()" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors">
                            📊 Analyze Repository
                        </button>
                        <button @click="createAgent()" class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors">
                            🤖 Create Agent
                        </button>
                        <button @click="optimizeSystem()" class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition-colors">
                            ⚡ Optimize System
                        </button>
                    </div>
                </div>
                
                <!-- System Logs -->
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold text-gray-800 mb-4">System Activity</h2>
                    <div class="space-y-2 max-h-96 overflow-y-auto">
                        <template x-for="log in logs" :key="log.id">
                            <div class="flex items-center space-x-2 p-2 bg-gray-50 rounded">
                                <div class="text-sm text-gray-500" x-text="log.timestamp"></div>
                                <div class="text-sm" x-text="log.message"></div>
                            </div>
                        </template>
                        <div x-show="logs.length === 0" class="text-gray-500 text-center py-8">
                            No recent activity
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                function dashboard() {
                    return {
                        status: {
                            status: 'unknown',
                            uptime: '0m',
                            active_agents: 0,
                            repositories_connected: 0,
                            hardware_nodes: 0
                        },
                        tasks: [],
                        logs: [],
                        
                        async init() {
                            await this.refreshStatus();
                            
                            // Refresh status every 10 seconds
                            setInterval(() => {
                                this.refreshStatus();
                            }, 10000);
                        },
                        
                        async refreshStatus() {
                            try {
                                const response = await fetch('/api/status');
                                this.status = await response.json();
                            } catch (error) {
                                console.error('Failed to fetch status:', error);
                            }
                        },
                        
                        async analyzeRepo() {
                            const path = prompt('Enter repository path:');
                            if (path) {
                                try {
                                    const response = await fetch('/api/commands/execute', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({
                                            command: 'analyze_repository',
                                            parameters: { path: path }
                                        })
                                    });
                                    const result = await response.json();
                                    this.addLog(`Repository analysis ${result.success ? 'completed' : 'failed'}: ${path}`);
                                } catch (error) {
                                    this.addLog(`Repository analysis failed: ${error.message}`);
                                }
                            }
                        },
                        
                        async createAgent() {
                            const type = prompt('Enter agent type (repository_analyzer, task_orchestrator):');
                            if (type) {
                                try {
                                    const response = await fetch('/api/agents', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/json' },
                                        body: JSON.stringify({
                                            agent_type: type,
                                            name: `${type}_${Date.now()}`
                                        })
                                    });
                                    const result = await response.json();
                                    this.addLog(`Agent created: ${type}`);
                                    await this.refreshStatus();
                                } catch (error) {
                                    this.addLog(`Agent creation failed: ${error.message}`);
                                }
                            }
                        },
                        
                        async optimizeSystem() {
                            try {
                                const response = await fetch('/api/commands/execute', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({
                                        command: 'optimize_system',
                                        parameters: {}
                                    })
                                });
                                const result = await response.json();
                                this.addLog('System optimization triggered');
                            } catch (error) {
                                this.addLog(`System optimization failed: ${error.message}`);
                            }
                        },
                        
                        addLog(message) {
                            this.logs.unshift({
                                id: Date.now(),
                                timestamp: new Date().toLocaleTimeString(),
                                message: message
                            });
                            
                            // Keep only last 50 logs
                            if (this.logs.length > 50) {
                                this.logs = this.logs.slice(0, 50);
                            }
                        }
                    }
                }
            </script>
        </body>
        </html>
        """
        return dashboard_html
    
    # API Routes
    @app.get("/api/status", response_model=SystemStatus)
    async def get_status():
        """Get system status."""
        orch = await get_orchestrator()
        return await orch.get_status()
    
    @app.get("/api/agents")
    async def list_agents():
        """List all agents."""
        orch = await get_orchestrator()
        if orch.agent_framework:
            agents_info = []
            for agent_id, agent in orch.agent_framework.agents.items():
                agents_info.append({
                    "id": agent_id,
                    "name": agent.name,
                    "status": agent.status,
                    "capabilities": [cap.name for cap in agent.capabilities],
                    "current_task": agent.current_task.id if agent.current_task else None
                })
            return {"agents": agents_info}
        return {"agents": []}
    
    @app.post("/api/agents")
    async def create_agent(request: AgentRequest, background_tasks: BackgroundTasks):
        """Create a new agent."""
        orch = await get_orchestrator()
        
        try:
            result = await orch.execute_command("create_agent", {
                "type": request.agent_type,
                "config": request.config
            })
            
            if result.get("success"):
                return {"success": True, "agent_id": result["agent_id"]}
            else:
                raise HTTPException(status_code=400, detail=result.get("error", "Agent creation failed"))
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/repositories")
    async def list_repositories():
        """List all repositories."""
        orch = await get_orchestrator()
        if orch.repository_manager:
            summary = await orch.repository_manager.get_repository_summary()
            repos = []
            for name, info in orch.repository_manager.connected_repositories.items():
                repos.append({
                    "name": name,
                    "path": info.path,
                    "languages": info.languages,
                    "technologies": info.technologies,
                    "capabilities": info.capabilities,
                    "file_count": info.file_count,
                    "last_modified": info.last_modified.isoformat() if info.last_modified else None
                })
            return {"repositories": repos, "summary": summary}
        return {"repositories": [], "summary": {}}
    
    @app.post("/api/tasks")
    async def create_task(request: TaskRequest):
        """Create a new task."""
        orch = await get_orchestrator()
        
        if orch.agent_framework:
            task = AgentTask(
                task_type=request.task_type,
                parameters=request.parameters,
                priority=request.priority
            )
            
            await orch.agent_framework.add_task(task)
            return {"success": True, "task_id": task.id}
        
        raise HTTPException(status_code=500, detail="Agent framework not available")
    
    @app.post("/api/commands/execute")
    async def execute_command(request: CommandRequest):
        """Execute a system command."""
        orch = await get_orchestrator()
        
        try:
            result = await orch.execute_command(request.command, request.parameters)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/infrastructure")
    async def get_infrastructure():
        """Get infrastructure status."""
        orch = await get_orchestrator()
        if orch.infrastructure:
            return {
                "nodes": {
                    node_id: {
                        "name": node.name,
                        "type": node.type,
                        "status": node.status,
                        "ip_address": node.ip_address,
                        "cpu_usage": node.cpu_usage,
                        "memory_usage": node.memory_usage,
                        "disk_usage": node.disk_usage
                    }
                    for node_id, node in orch.infrastructure.connected_nodes.items()
                },
                "containers": {
                    container_id: {
                        "name": container.name,
                        "image": container.image,
                        "status": container.status,
                        "ports": container.ports
                    }
                    for container_id, container in orch.infrastructure.running_containers.items()
                }
            }
        return {"nodes": {}, "containers": {}}
    
    @app.post("/api/infrastructure/deploy")
    async def deploy_infrastructure_component(
        component: str,
        background_tasks: BackgroundTasks
    ):
        """Deploy infrastructure component."""
        orch = await get_orchestrator()
        
        if not orch.infrastructure:
            raise HTTPException(status_code=500, detail="Infrastructure manager not available")
        
        async def deploy_task():
            if component == "arangodb":
                success = await orch.infrastructure.deploy_arangodb()
            elif component == "monitoring":
                success = await orch.infrastructure.deploy_monitoring_stack()
            else:
                success = False
            
            logger.info(f"Infrastructure deployment {component}: {'success' if success else 'failed'}")
        
        background_tasks.add_task(deploy_task)
        return {"success": True, "message": f"Deploying {component}"}
    
    @app.get("/api/knowledge-graph/overview")
    async def get_knowledge_graph_overview():
        """Get knowledge graph overview."""
        orch = await get_orchestrator()
        if orch.knowledge_graph:
            overview = await orch.knowledge_graph.get_system_overview()
            return overview
        return {}
    
    return app


async def run_server(config: OrchestratorConfig):
    """Run the web server."""
    app = create_app(config)
    
    uvicorn_config = uvicorn.Config(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info",
        workers=1  # Single worker for now to avoid orchestrator conflicts
    )
    
    server = uvicorn.Server(uvicorn_config)
    await server.serve()