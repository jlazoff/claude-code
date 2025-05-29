#!/usr/bin/env python3
"""
AG-UI Frontend Orchestrator
Complete async frontend integration with self-executing project orchestration
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import websockets
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Import all our orchestrators
from local_agentic_framework import LocalAgenticFramework
from github_discovery_orchestrator import GitHubDiscoveryOrchestrator
from digital_twin_orchestrator import DigitalTwinOrchestrator
from distributed_inference_orchestrator import DistributedInferenceOrchestrator
from mcp_server_manager import MCPServerManager
from youtube_agent_launcher import YouTubeAgentLauncher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectTask(BaseModel):
    """Model for project tasks"""
    task_id: str = Field(..., description="Unique task ID")
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    priority: str = Field(default="medium", description="Task priority")
    status: str = Field(default="pending", description="Task status")
    project_type: str = Field(..., description="Type of project")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Task requirements")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    estimated_time: str = Field(default="unknown", description="Estimated completion time")
    assigned_agents: List[str] = Field(default_factory=list, description="Assigned agents")
    progress: float = Field(default=0.0, description="Completion progress (0-1)")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    completion_criteria: List[str] = Field(default_factory=list, description="Criteria for completion")

class UserDecision(BaseModel):
    """Model for user decisions in the frontend"""
    decision_id: str = Field(..., description="Unique decision ID")
    title: str = Field(..., description="Decision title")
    description: str = Field(..., description="Decision description")
    options: List[Dict[str, Any]] = Field(..., description="Available options")
    recommended_option: Optional[str] = Field(None, description="AI recommended option")
    context: Dict[str, Any] = Field(default_factory=dict, description="Decision context")
    pros_cons: Dict[str, List[str]] = Field(default_factory=dict, description="Pros and cons for each option")
    urgency: str = Field(default="medium", description="Decision urgency")
    category: str = Field(..., description="Decision category")
    requires_user_input: bool = Field(default=True, description="Whether user input is required")

class AGUIFrontendOrchestrator:
    """Main orchestrator with AG-UI frontend and continuous execution"""
    
    def __init__(self):
        self.foundation_dir = Path("foundation_data")
        self.frontend_dir = self.foundation_dir / "agui_frontend"
        self.projects_dir = self.foundation_dir / "projects"
        self.decisions_dir = self.foundation_dir / "decisions"
        
        # Create directories
        for dir_path in [self.foundation_dir, self.frontend_dir, self.projects_dir, self.decisions_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize all orchestrators
        self.framework = LocalAgenticFramework()
        self.github_orchestrator = None
        self.digital_twin = None
        self.inference_orchestrator = None
        self.mcp_manager = None
        self.youtube_launcher = None
        
        # System state
        self.active_projects: Dict[str, ProjectTask] = {}
        self.pending_decisions: Dict[str, UserDecision] = {}
        self.connected_clients: List[WebSocket] = []
        self.is_running = False
        
        # FastAPI app
        self.app = FastAPI(title="Claude Code Master Orchestrator", version="1.0.0")
        self.setup_fastapi_routes()
        
        # Background tasks
        self.background_tasks = []
        
        logger.info("AG-UI Frontend Orchestrator initialized")

    async def initialize_system(self):
        """Initialize all system components"""
        logger.info("🚀 Initializing complete system...")
        
        # Wait for framework initialization
        await asyncio.sleep(5)
        
        # Initialize orchestrators
        self.github_orchestrator = GitHubDiscoveryOrchestrator(self.framework)
        self.digital_twin = DigitalTwinOrchestrator(self.framework)
        self.inference_orchestrator = DistributedInferenceOrchestrator()
        self.mcp_manager = MCPServerManager()
        self.youtube_launcher = YouTubeAgentLauncher()
        
        # Wait for digital twin initialization
        await asyncio.sleep(3)
        
        # Create initial frontend
        await self.create_agui_frontend()
        
        # Start background execution
        self.is_running = True
        asyncio.create_task(self.continuous_execution_loop())
        asyncio.create_task(self.todo_monitoring_loop())
        asyncio.create_task(self.project_execution_loop())
        
        logger.info("✅ System initialization complete")

    def setup_fastapi_routes(self):
        """Set up FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_frontend():
            """Serve the main frontend"""
            frontend_file = self.frontend_dir / "index.html"
            if frontend_file.exists():
                return frontend_file.read_text()
            else:
                return "<h1>AG-UI Frontend Loading...</h1><p>Please wait while the system initializes.</p>"
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication"""
            await websocket.accept()
            self.connected_clients.append(websocket)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    await self.handle_websocket_message(websocket, message)
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
        
        @self.app.get("/api/status")
        async def get_system_status():
            """Get comprehensive system status"""
            return await self.get_comprehensive_status()
        
        @self.app.get("/api/projects")
        async def get_projects():
            """Get all active projects"""
            return {"projects": [project.model_dump() for project in self.active_projects.values()]}
        
        @self.app.get("/api/decisions")
        async def get_pending_decisions():
            """Get pending user decisions"""
            return {"decisions": [decision.model_dump() for decision in self.pending_decisions.values()]}
        
        @self.app.post("/api/decisions/{decision_id}/choose")
        async def make_decision(decision_id: str, choice: Dict[str, Any]):
            """Make a decision"""
            return await self.handle_user_decision(decision_id, choice)
        
        @self.app.get("/api/todos")
        async def get_todos():
            """Get current todos"""
            return await self.get_current_todos()
        
        @self.app.post("/api/execute-project")
        async def execute_project(project_data: Dict[str, Any]):
            """Execute a new project"""
            return await self.create_and_execute_project(project_data)
        
        # Serve static files
        self.app.mount("/static", StaticFiles(directory=str(self.frontend_dir)), name="static")

    async def handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = message.get("type")
        
        if message_type == "get_status":
            status = await self.get_comprehensive_status()
            await websocket.send_text(json.dumps({"type": "status_update", "data": status}))
        
        elif message_type == "make_decision":
            decision_id = message.get("decision_id")
            choice = message.get("choice")
            result = await self.handle_user_decision(decision_id, choice)
            await websocket.send_text(json.dumps({"type": "decision_result", "data": result}))
        
        elif message_type == "execute_project":
            project_data = message.get("project_data")
            result = await self.create_and_execute_project(project_data)
            await websocket.send_text(json.dumps({"type": "project_result", "data": result}))
        
        elif message_type == "get_predictions":
            context = message.get("context", {})
            predictions = await self.get_ai_predictions(context)
            await websocket.send_text(json.dumps({"type": "predictions", "data": predictions}))

    async def broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if self.connected_clients:
            message_json = json.dumps(message, default=str)
            disconnected = []
            
            for client in self.connected_clients:
                try:
                    await client.send_text(message_json)
                except:
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.connected_clients.remove(client)

    async def create_agui_frontend(self):
        """Create the AG-UI frontend"""
        logger.info("🎨 Creating AG-UI frontend...")
        
        frontend_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Code Master Orchestrator - AG-UI</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff; 
            overflow-x: hidden;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { 
            text-align: center; 
            margin-bottom: 30px; 
            background: rgba(255,255,255,0.1); 
            padding: 20px; 
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }
        .card { 
            background: rgba(255,255,255,0.15); 
            padding: 25px; 
            border-radius: 15px; 
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        .card h3 { 
            margin-bottom: 15px; 
            color: #FFD700; 
            font-size: 1.3em;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-indicator { 
            display: inline-block; 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            margin-right: 8px; 
        }
        .status-green { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
        .status-yellow { background: #ffaa00; box-shadow: 0 0 10px #ffaa00; }
        .status-red { background: #ff4444; box-shadow: 0 0 10px #ff4444; }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin: 12px 0; 
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric-label { color: #ccc; }
        .metric-value { font-weight: bold; color: #fff; }
        .button { 
            background: linear-gradient(45deg, #00ff88, #00ccff); 
            color: #000; 
            border: none; 
            padding: 12px 24px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-weight: bold;
            margin: 5px;
            transition: all 0.3s ease;
        }
        .button:hover { 
            transform: scale(1.05); 
            box-shadow: 0 5px 15px rgba(0,255,136,0.4);
        }
        .button.secondary { 
            background: linear-gradient(45deg, #667eea, #764ba2); 
            color: #fff; 
        }
        .decision-card {
            background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
        }
        .project-list { max-height: 300px; overflow-y: auto; }
        .project-item {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #00ff88;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00ccff);
            transition: width 0.3s ease;
        }
        .log-container { 
            background: rgba(0,0,0,0.3); 
            padding: 15px; 
            border-radius: 10px; 
            font-family: 'Courier New', monospace; 
            font-size: 12px; 
            max-height: 300px; 
            overflow-y: auto; 
            border: 1px solid rgba(255,255,255,0.2);
        }
        .option-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .option-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .option-card:hover {
            border-color: #00ff88;
            background: rgba(0,255,136,0.2);
        }
        .option-card.recommended {
            border-color: #FFD700;
            background: rgba(255,215,0,0.2);
        }
        .pros-cons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 10px;
        }
        .pros, .cons {
            padding: 10px;
            border-radius: 8px;
        }
        .pros { background: rgba(0,255,136,0.2); }
        .cons { background: rgba(255,68,68,0.2); }
        .loading { 
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #00ff88;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="root"></div>
    
    <script type="text/babel">
        function MasterOrchestrator() {
            const [systemStatus, setSystemStatus] = React.useState(null);
            const [projects, setProjects] = React.useState([]);
            const [decisions, setDecisions] = React.useState([]);
            const [logs, setLogs] = React.useState([]);
            const [connected, setConnected] = React.useState(false);
            const [ws, setWs] = React.useState(null);
            
            React.useEffect(() => {
                // Initialize WebSocket connection
                const websocket = new WebSocket(`ws://${window.location.host}/ws`);
                
                websocket.onopen = () => {
                    setConnected(true);
                    setWs(websocket);
                    websocket.send(JSON.stringify({type: "get_status"}));
                };
                
                websocket.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    handleWebSocketMessage(message);
                };
                
                websocket.onclose = () => {
                    setConnected(false);
                    setTimeout(() => {
                        window.location.reload();
                    }, 5000);
                };
                
                // Fetch initial data
                fetchData();
                
                // Set up periodic updates
                const interval = setInterval(fetchData, 30000);
                
                return () => {
                    clearInterval(interval);
                    if (websocket) websocket.close();
                };
            }, []);
            
            const handleWebSocketMessage = (message) => {
                switch (message.type) {
                    case "status_update":
                        setSystemStatus(message.data);
                        break;
                    case "projects_update":
                        setProjects(message.data);
                        break;
                    case "decisions_update":
                        setDecisions(message.data);
                        break;
                    case "log_update":
                        setLogs(prev => [...prev.slice(-100), message.data]);
                        break;
                }
            };
            
            const fetchData = async () => {
                try {
                    const [statusRes, projectsRes, decisionsRes] = await Promise.all([
                        fetch('/api/status'),
                        fetch('/api/projects'),
                        fetch('/api/decisions')
                    ]);
                    
                    const [status, projects, decisions] = await Promise.all([
                        statusRes.json(),
                        projectsRes.json(),
                        decisionsRes.json()
                    ]);
                    
                    setSystemStatus(status);
                    setProjects(projects.projects || []);
                    setDecisions(decisions.decisions || []);
                } catch (error) {
                    console.error('Failed to fetch data:', error);
                }
            };
            
            const makeDecision = async (decisionId, choice) => {
                if (ws) {
                    ws.send(JSON.stringify({
                        type: "make_decision",
                        decision_id: decisionId,
                        choice: choice
                    }));
                }
            };
            
            const executeProject = async (projectData) => {
                if (ws) {
                    ws.send(JSON.stringify({
                        type: "execute_project",
                        project_data: projectData
                    }));
                }
            };
            
            if (!systemStatus) {
                return (
                    <div className="container">
                        <div className="header">
                            <h1>🤖 Claude Code Master Orchestrator</h1>
                            <div className="loading"></div>
                            <p>Initializing system...</p>
                        </div>
                    </div>
                );
            }
            
            return (
                <div className="container">
                    <div className="header">
                        <h1>🤖 Claude Code Master Orchestrator</h1>
                        <p>
                            <span className={`status-indicator ${connected ? 'status-green' : 'status-red'}`}></span>
                            {connected ? 'Connected' : 'Disconnected'} | Last updated: {new Date().toLocaleTimeString()}
                        </p>
                    </div>
                    
                    {decisions.length > 0 && (
                        <div className="decision-card card">
                            <h3>⚡ Pending Decisions ({decisions.length})</h3>
                            {decisions.map(decision => (
                                <DecisionCard key={decision.decision_id} decision={decision} onDecision={makeDecision} />
                            ))}
                        </div>
                    )}
                    
                    <div className="grid">
                        <SystemStatusCard status={systemStatus} />
                        <InferenceClusterCard cluster={systemStatus.inference_cluster} />
                        <AgentsCard agents={systemStatus.agents} />
                        <ProjectsCard projects={projects} onExecute={executeProject} />
                    </div>
                    
                    <div className="grid">
                        <DatabasesCard databases={systemStatus.databases} />
                        <MCPServersCard mcpServers={systemStatus.mcp_servers} />
                        <DigitalTwinCard digitalTwin={systemStatus.digital_twin} />
                        <SystemLogsCard logs={logs} />
                    </div>
                </div>
            );
        }
        
        function SystemStatusCard({ status }) {
            return (
                <div className="card">
                    <h3>🖥️ System Status</h3>
                    <div className="metric">
                        <span className="metric-label">Framework:</span>
                        <span className="metric-value">
                            <span className="status-indicator status-green"></span>
                            Active
                        </span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">CPU Usage:</span>
                        <span className="metric-value">{status.metrics?.cpu_usage?.toFixed(1) || 0}%</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Memory Usage:</span>
                        <span className="metric-value">{status.metrics?.memory_usage?.toFixed(1) || 0}%</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Uptime:</span>
                        <span className="metric-value">{status.metrics?.uptime || '0h 0m'}</span>
                    </div>
                </div>
            );
        }
        
        function InferenceClusterCard({ cluster }) {
            return (
                <div className="card">
                    <h3>🧠 Distributed Inference</h3>
                    <div className="metric">
                        <span className="metric-label">Active Nodes:</span>
                        <span className="metric-value">{cluster?.active_nodes || 0}/{cluster?.total_nodes || 0}</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Throughput:</span>
                        <span className="metric-value">{cluster?.throughput || 0} req/s</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Models:</span>
                        <span className="metric-value">{cluster?.deployed_models?.length || 0}</span>
                    </div>
                    {cluster?.nodes?.map((node, i) => (
                        <div key={i} className="metric">
                            <span className="metric-label">{node.name}:</span>
                            <span className="metric-value">
                                <span className={`status-indicator status-${node.status === 'active' ? 'green' : 'red'}`}></span>
                                {node.status}
                            </span>
                        </div>
                    ))}
                </div>
            );
        }
        
        function AgentsCard({ agents }) {
            return (
                <div className="card">
                    <h3>🤖 Active Agents</h3>
                    <div className="metric">
                        <span className="metric-label">Total Agents:</span>
                        <span className="metric-value">{agents?.total || 0}</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Active:</span>
                        <span className="metric-value">{agents?.active || 0}</span>
                    </div>
                    {agents?.agents?.map((agent, i) => (
                        <div key={i} className="metric">
                            <span className="metric-label">{agent.id}:</span>
                            <span className="metric-value">
                                <span className={`status-indicator status-${agent.status === 'active' ? 'green' : 'yellow'}`}></span>
                                {agent.type}
                            </span>
                        </div>
                    ))}
                </div>
            );
        }
        
        function ProjectsCard({ projects, onExecute }) {
            return (
                <div className="card">
                    <h3>📋 Active Projects ({projects.length})</h3>
                    <button className="button" onClick={() => onExecute({
                        title: "New Auto-Generated Project",
                        type: "research_implementation",
                        priority: "high"
                    })}>
                        ➕ Create Project
                    </button>
                    <div className="project-list">
                        {projects.map((project, i) => (
                            <div key={i} className="project-item">
                                <div style={{fontWeight: 'bold'}}>{project.title}</div>
                                <div style={{fontSize: '0.9em', opacity: 0.8}}>{project.description}</div>
                                <div className="progress-bar">
                                    <div className="progress-fill" style={{width: `${project.progress * 100}%`}}></div>
                                </div>
                                <div style={{marginTop: '5px', fontSize: '0.8em'}}>
                                    {project.status} | {project.priority} priority
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            );
        }
        
        function DecisionCard({ decision, onDecision }) {
            return (
                <div style={{margin: '15px 0', padding: '15px', background: 'rgba(255,255,255,0.1)', borderRadius: '10px'}}>
                    <h4>{decision.title}</h4>
                    <p>{decision.description}</p>
                    
                    <div className="option-grid">
                        {decision.options.map((option, i) => (
                            <div 
                                key={i} 
                                className={`option-card ${option.id === decision.recommended_option ? 'recommended' : ''}`}
                                onClick={() => onDecision(decision.decision_id, option)}
                            >
                                <div style={{fontWeight: 'bold'}}>{option.label || option.id}</div>
                                <div style={{fontSize: '0.9em', opacity: 0.8}}>{option.description}</div>
                                {option.id === decision.recommended_option && (
                                    <div style={{color: '#FFD700', fontSize: '0.8em', marginTop: '5px'}}>
                                        ⭐ AI Recommended
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                    
                    {decision.pros_cons && (
                        <div className="pros-cons">
                            <div className="pros">
                                <strong>Pros:</strong>
                                <ul>
                                    {(decision.pros_cons.pros || []).map((pro, i) => (
                                        <li key={i}>{pro}</li>
                                    ))}
                                </ul>
                            </div>
                            <div className="cons">
                                <strong>Cons:</strong>
                                <ul>
                                    {(decision.pros_cons.cons || []).map((con, i) => (
                                        <li key={i}>{con}</li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    )}
                </div>
            );
        }
        
        function DatabasesCard({ databases }) {
            return (
                <div className="card">
                    <h3>🗄️ Databases</h3>
                    <div className="metric">
                        <span className="metric-label">ArangoDB:</span>
                        <span className="metric-value">
                            <span className={`status-indicator status-${databases?.arangodb === 'active' ? 'green' : 'red'}`}></span>
                            {databases?.arangodb || 'inactive'}
                        </span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Iceberg:</span>
                        <span className="metric-value">
                            <span className={`status-indicator status-${databases?.iceberg === 'active' ? 'green' : 'red'}`}></span>
                            {databases?.iceberg || 'inactive'}
                        </span>
                    </div>
                </div>
            );
        }
        
        function MCPServersCard({ mcpServers }) {
            return (
                <div className="card">
                    <h3>🔌 MCP Servers</h3>
                    <div className="metric">
                        <span className="metric-label">Deployed:</span>
                        <span className="metric-value">{mcpServers?.deployed || 0}</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Failed:</span>
                        <span className="metric-value">{mcpServers?.failed || 0}</span>
                    </div>
                    {mcpServers?.servers?.map((server, i) => (
                        <div key={i} className="metric">
                            <span className="metric-label">{server}:</span>
                            <span className="metric-value">
                                <span className="status-indicator status-green"></span>
                                Active
                            </span>
                        </div>
                    ))}
                </div>
            );
        }
        
        function DigitalTwinCard({ digitalTwin }) {
            return (
                <div className="card">
                    <h3>🧬 Digital Twin</h3>
                    <div className="metric">
                        <span className="metric-label">Status:</span>
                        <span className="metric-value">
                            <span className="status-indicator status-green"></span>
                            {digitalTwin?.status || 'active'}
                        </span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Interactions:</span>
                        <span className="metric-value">{digitalTwin?.interactions || 0}</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Patterns:</span>
                        <span className="metric-value">{digitalTwin?.patterns || 0}</span>
                    </div>
                    <div className="metric">
                        <span className="metric-label">Accuracy:</span>
                        <span className="metric-value">{((digitalTwin?.accuracy || 0.5) * 100).toFixed(1)}%</span>
                    </div>
                </div>
            );
        }
        
        function SystemLogsCard({ logs }) {
            return (
                <div className="card" style={{gridColumn: 'span 2'}}>
                    <h3>📝 System Logs</h3>
                    <div className="log-container">
                        {logs.slice(-20).map((log, i) => (
                            <div key={i}>{log}</div>
                        ))}
                        {logs.length === 0 && <div>No recent logs...</div>}
                    </div>
                </div>
            );
        }
        
        ReactDOM.render(<MasterOrchestrator />, document.getElementById('root'));
    </script>
</body>
</html>'''
        
        # Save frontend
        frontend_file = self.frontend_dir / "index.html"
        frontend_file.write_text(frontend_html)
        
        logger.info("✅ AG-UI frontend created")

    async def continuous_execution_loop(self):
        """Continuous execution loop that monitors and executes tasks"""
        logger.info("🔄 Starting continuous execution loop...")
        
        while self.is_running:
            try:
                # Check for pending projects
                await self.check_and_execute_pending_projects()
                
                # Monitor system health
                await self.monitor_system_health()
                
                # Auto-generate new projects based on discoveries
                await self.auto_generate_projects()
                
                # Update clients
                await self.broadcast_status_update()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in continuous execution loop: {e}")
                await asyncio.sleep(60)

    async def todo_monitoring_loop(self):
        """Monitor and auto-execute todos"""
        logger.info("📋 Starting todo monitoring loop...")
        
        while self.is_running:
            try:
                # Get current todos
                todos = await self.get_current_todos()
                
                # Auto-execute high priority todos
                for todo in todos.get("todos", []):
                    if todo.get("priority") == "high" and todo.get("status") == "pending":
                        await self.auto_execute_todo(todo)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in todo monitoring loop: {e}")
                await asyncio.sleep(120)

    async def project_execution_loop(self):
        """Execute active projects"""
        logger.info("🏗️ Starting project execution loop...")
        
        while self.is_running:
            try:
                # Execute active projects
                for project in self.active_projects.values():
                    if project.status == "in_progress":
                        await self.advance_project(project)
                
                await asyncio.sleep(45)  # Check every 45 seconds
                
            except Exception as e:
                logger.error(f"Error in project execution loop: {e}")
                await asyncio.sleep(90)

    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Get status from all components
        framework_status = await self.framework.get_framework_status() if self.framework else {}
        
        digital_twin_status = {}
        if self.digital_twin:
            try:
                digital_twin_status = await self.digital_twin.get_twin_status()
            except:
                digital_twin_status = {"status": "inactive"}
        
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework_status": framework_status.get("framework_status", "unknown"),
            "inference_cluster": {
                "total_nodes": framework_status.get("inference_servers", {}).get("total", 0),
                "active_nodes": framework_status.get("inference_servers", {}).get("active", 0),
                "throughput": 0,
                "deployed_models": ["microsoft/DialoGPT-small"],
                "nodes": [
                    {"name": s["name"], "status": s["status"]} 
                    for s in framework_status.get("inference_servers", {}).get("servers", [])
                ]
            },
            "agents": framework_status.get("agents", {}),
            "databases": framework_status.get("databases", {}),
            "mcp_servers": framework_status.get("mcp_servers", {"deployed": 0, "failed": 0, "servers": []}),
            "digital_twin": {
                "status": digital_twin_status.get("status", "inactive"),
                "interactions": digital_twin_status.get("statistics", {}).get("total_interactions", 0),
                "patterns": digital_twin_status.get("statistics", {}).get("total_patterns", 0),
                "accuracy": digital_twin_status.get("learning_metrics", {}).get("pattern_accuracy", 0.5)
            },
            "metrics": framework_status.get("metrics", {})
        }

    async def get_current_todos(self) -> Dict[str, Any]:
        """Get current todos from the system"""
        # This would integrate with the todo system from previous implementations
        return {
            "todos": [
                {
                    "id": "repo_analysis",
                    "content": "Analyze discovered GitHub repositories for containerization",
                    "status": "pending",
                    "priority": "high"
                },
                {
                    "id": "mcp_integration",
                    "content": "Complete MCP server integration with Aider and Vertex AI",
                    "status": "in_progress", 
                    "priority": "high"
                },
                {
                    "id": "youtube_monitoring",
                    "content": "Monitor YouTube channels for new research content",
                    "status": "pending",
                    "priority": "medium"
                }
            ]
        }

    async def auto_execute_todo(self, todo: Dict[str, Any]):
        """Auto-execute a todo item"""
        logger.info(f"🤖 Auto-executing todo: {todo['content']}")
        
        todo_id = todo["id"]
        
        if todo_id == "repo_analysis" and self.github_orchestrator:
            # Execute GitHub repository analysis
            await self.github_orchestrator.orchestrate_discovery_and_containerization()
            
        elif todo_id == "mcp_integration" and self.mcp_manager:
            # Execute MCP server deployment
            await self.mcp_manager.deploy_all_mcp_servers()
            
        elif todo_id == "youtube_monitoring" and self.youtube_launcher:
            # Launch YouTube monitoring
            await self.youtube_launcher.launch_youtube_agent("https://www.youtube.com/@TwoMinutePapers", continuous=True)

    async def create_and_execute_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and execute a new project"""
        project_id = f"proj_{int(time.time())}"
        
        project = ProjectTask(
            task_id=project_id,
            title=project_data.get("title", "Auto-Generated Project"),
            description=project_data.get("description", "Automatically generated project"),
            priority=project_data.get("priority", "medium"),
            project_type=project_data.get("type", "general"),
            requirements=project_data.get("requirements", {}),
            estimated_time=project_data.get("estimated_time", "1-2 weeks"),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            completion_criteria=project_data.get("completion_criteria", ["Functional implementation", "Testing complete", "Documentation created"])
        )
        
        # Store project
        self.active_projects[project_id] = project
        
        # Save project to disk
        project_file = self.projects_dir / f"project_{project_id}.json"
        with open(project_file, 'w') as f:
            json.dump(project.model_dump(), f, indent=2, default=str)
        
        # Start project execution
        project.status = "in_progress"
        asyncio.create_task(self.execute_project_async(project))
        
        logger.info(f"✅ Created and started project: {project.title}")
        
        return {"project_id": project_id, "status": "created", "project": project.model_dump()}

    async def execute_project_async(self, project: ProjectTask):
        """Execute a project asynchronously"""
        logger.info(f"🚀 Executing project: {project.title}")
        
        try:
            # Assign agents based on project type
            if project.project_type == "research_implementation":
                project.assigned_agents = ["research_analyzer", "code_generator"]
            elif project.project_type == "github_analysis":
                project.assigned_agents = ["system_orchestrator", "knowledge_curator"]
            else:
                project.assigned_agents = ["system_orchestrator"]
            
            # Execute project phases
            phases = [
                ("Planning", 0.1),
                ("Research", 0.3),
                ("Implementation", 0.7),
                ("Testing", 0.9),
                ("Completion", 1.0)
            ]
            
            for phase_name, target_progress in phases:
                logger.info(f"📈 Project {project.task_id} - Phase: {phase_name}")
                
                # Simulate work with actual task processing
                if phase_name == "Research" and self.github_orchestrator:
                    # Do actual research
                    await asyncio.create_task(self.github_orchestrator.discover_github_repositories())
                elif phase_name == "Implementation":
                    # Process with framework
                    task_result = await self.framework.process_task({
                        "type": project.project_type,
                        "content": project.description
                    })
                    project.requirements["implementation_result"] = task_result
                
                project.progress = target_progress
                project.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # Broadcast progress update
                await self.broadcast_to_clients({
                    "type": "project_progress",
                    "data": {"project_id": project.task_id, "progress": project.progress, "phase": phase_name}
                })
                
                await asyncio.sleep(10)  # Simulate phase duration
            
            project.status = "completed"
            logger.info(f"✅ Project completed: {project.title}")
            
        except Exception as e:
            logger.error(f"❌ Project execution failed: {e}")
            project.status = "failed"
            project.requirements["error"] = str(e)

    async def handle_user_decision(self, decision_id: str, choice: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user decision"""
        if decision_id in self.pending_decisions:
            decision = self.pending_decisions[decision_id]
            
            # Record interaction with digital twin
            if self.digital_twin:
                await self.digital_twin.record_interaction(
                    interaction_type=decision.category,
                    context=decision.context,
                    user_choice=choice.get("id", str(choice)),
                    available_options=decision.options,
                    response_time=5.0,  # Would be calculated from UI
                    confidence=0.8
                )
            
            # Remove from pending decisions
            del self.pending_decisions[decision_id]
            
            # Execute the chosen action
            result = await self.execute_decision_choice(decision, choice)
            
            logger.info(f"✅ User decision executed: {decision.title} -> {choice}")
            
            return {"status": "executed", "result": result}
        
        return {"status": "not_found"}

    async def execute_decision_choice(self, decision: UserDecision, choice: Dict[str, Any]) -> Any:
        """Execute the user's choice"""
        choice_id = choice.get("id")
        
        if decision.category == "project_creation":
            if choice_id == "auto_generate":
                return await self.create_and_execute_project({
                    "title": "Auto-Generated Research Project",
                    "type": "research_implementation",
                    "priority": "high"
                })
        elif decision.category == "tool_deployment":
            if choice_id == "deploy_all":
                # Deploy all tools
                results = []
                if self.mcp_manager:
                    results.append(await self.mcp_manager.deploy_all_mcp_servers())
                return results
        
        return {"choice": choice_id, "executed": True}

    async def get_ai_predictions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI predictions for the given context"""
        if self.digital_twin:
            return await self.digital_twin.predict_user_choice(
                interaction_type=context.get("type", "general"),
                context=context,
                available_options=context.get("options", [])
            )
        
        return {"predicted_choice": None, "confidence": 0.0, "reasoning": ["Digital twin not available"]}

    async def broadcast_status_update(self):
        """Broadcast status update to all clients"""
        status = await self.get_comprehensive_status()
        await self.broadcast_to_clients({
            "type": "status_update",
            "data": status
        })

    async def check_and_execute_pending_projects(self):
        """Check for and execute pending projects"""
        # Auto-generate projects based on system state
        if len(self.active_projects) < 3:  # Maintain at least 3 active projects
            await self.create_and_execute_project({
                "title": f"Auto-Generated Project {len(self.active_projects) + 1}",
                "description": "Automatically generated project to maintain system activity",
                "type": "research_implementation",
                "priority": "medium"
            })

    async def monitor_system_health(self):
        """Monitor overall system health"""
        status = await self.get_comprehensive_status()
        
        # Check for issues and create decisions if needed
        if status["inference_cluster"]["active_nodes"] == 0:
            # Create decision for inference setup
            decision = UserDecision(
                decision_id=f"decision_{int(time.time())}",
                title="Inference Cluster Setup Required",
                description="No active inference nodes detected. How would you like to proceed?",
                options=[
                    {"id": "setup_local", "label": "Set up local inference", "description": "Deploy inference servers on local machine"},
                    {"id": "setup_distributed", "label": "Set up distributed inference", "description": "Deploy across network devices"},
                    {"id": "skip", "label": "Skip for now", "description": "Continue without inference cluster"}
                ],
                recommended_option="setup_local",
                category="infrastructure",
                context={"issue": "no_inference_nodes"}
            )
            
            self.pending_decisions[decision.decision_id] = decision

    async def auto_generate_projects(self):
        """Auto-generate new projects based on discoveries"""
        # This would analyze the knowledge graph and generate relevant projects
        pass

    async def advance_project(self, project: ProjectTask):
        """Advance a project's progress"""
        if project.progress < 1.0:
            project.progress = min(1.0, project.progress + 0.05)  # 5% progress increment
            project.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            
            if project.progress >= 1.0:
                project.status = "completed"

    async def run_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server"""
        logger.info(f"🌐 Starting AG-UI server on {host}:{port}")
        
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=False
        )
        
        server = uvicorn.Server(config)
        await server.serve()

async def main():
    """Main execution function"""
    orchestrator = AGUIFrontendOrchestrator()
    
    # Initialize system
    await orchestrator.initialize_system()
    
    print("🚀 Claude Code Master Orchestrator with AG-UI Frontend")
    print("="*60)
    print("🌐 Frontend: http://localhost:8000")
    print("📊 Real-time WebSocket updates")
    print("🤖 Autonomous project execution")
    print("🧬 Digital twin learning")
    print("📋 Continuous todo monitoring")
    print("="*60)
    
    # Run server
    await orchestrator.run_server()

if __name__ == "__main__":
    asyncio.run(main())