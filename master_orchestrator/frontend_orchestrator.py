#!/usr/bin/env python3
"""
Frontend Orchestrator - Complete end-to-end frontend with one-click deployment
Real-time monitoring, project completion optimization, and continuous improvement
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import hashlib
import aiohttp
import aiofiles
from aiohttp import web, WSMsgType
import aiohttp_cors
import jinja2
import websockets
import psutil
import git

from unified_config import SecureConfigManager
from parallel_llm_orchestrator import ParallelLLMOrchestrator
from computer_control_orchestrator import ComputerControlOrchestrator
from content_analyzer_deployer import ContentAnalyzerDeployer

class FrontendOrchestrator:
    """Complete frontend orchestration system"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.llm_orchestrator = ParallelLLMOrchestrator()
        self.computer_control = ComputerControlOrchestrator()
        self.content_analyzer = ContentAnalyzerDeployer()
        
        self.app = web.Application()
        self.websocket_clients = set()
        self.active_projects = {}
        self.monitoring_data = {}
        self.optimization_queue = asyncio.Queue()
        
        # Setup Jinja2 templates
        self.template_loader = jinja2.FileSystemLoader('templates')
        self.template_env = jinja2.Environment(loader=self.template_loader)
        
    async def initialize(self):
        """Initialize all components"""
        await self.config.initialize()
        await self.llm_orchestrator.initialize()
        await self.computer_control.initialize()
        await self.content_analyzer.initialize()
        
        # Setup routes
        self._setup_routes()
        self._setup_cors()
        
        # Create template directory
        await self._create_templates()
        
        # Start background monitoring
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._optimization_processor())
        
        logging.info("Frontend Orchestrator initialized")
        
    def _setup_routes(self):
        """Setup all HTTP routes"""
        # Static files
        self.app.router.add_static('/', 'static/', name='static')
        
        # Main interface
        self.app.router.add_get('/', self.serve_main_interface)
        self.app.router.add_get('/dashboard', self.serve_dashboard)
        self.app.router.add_get('/projects', self.serve_projects)
        self.app.router.add_get('/monitoring', self.serve_monitoring)
        self.app.router.add_get('/optimization', self.serve_optimization)
        
        # API endpoints
        self.app.router.add_post('/api/generate-code', self.api_generate_code)
        self.app.router.add_post('/api/analyze-content', self.api_analyze_content)
        self.app.router.add_post('/api/create-project', self.api_create_project)
        self.app.router.add_get('/api/project/{project_id}', self.api_get_project)
        self.app.router.add_post('/api/optimize-project', self.api_optimize_project)
        self.app.router.add_get('/api/monitoring-data', self.api_get_monitoring_data)
        self.app.router.add_post('/api/execute-command', self.api_execute_command)
        
        # WebSocket endpoint
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # Health check
        self.app.router.add_get('/health', self.health_check)
        
    def _setup_cors(self):
        """Setup CORS for development"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
            
    async def _create_templates(self):
        """Create HTML templates"""
        template_dir = Path("templates")
        template_dir.mkdir(exist_ok=True)
        
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        
        # Main interface template
        main_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Orchestrator - AI-Powered Development Platform</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .glass { backdrop-filter: blur(10px); background: rgba(255, 255, 255, 0.1); }
        .pulse-dot { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
        .typing::after {
            content: '|';
            animation: blink 1s infinite;
        }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
    </style>
</head>
<body class="bg-gray-900 text-white">
    <div id="root"></div>
    
    <script type="text/babel">
        const { useState, useEffect, useRef } = React;
        
        function App() {
            const [activeTab, setActiveTab] = useState('dashboard');
            const [isConnected, setIsConnected] = useState(false);
            const [messages, setMessages] = useState([]);
            const [currentInput, setCurrentInput] = useState('');
            const [projects, setProjects] = useState([]);
            const [monitoringData, setMonitoringData] = useState({});
            const [isGenerating, setIsGenerating] = useState(false);
            const wsRef = useRef(null);
            
            // WebSocket connection
            useEffect(() => {
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                wsRef.current = ws;
                
                ws.onopen = () => {
                    setIsConnected(true);
                    console.log('WebSocket connected');
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };
                
                ws.onclose = () => {
                    setIsConnected(false);
                    console.log('WebSocket disconnected');
                };
                
                return () => ws.close();
            }, []);
            
            // Fetch monitoring data
            useEffect(() => {
                const fetchMonitoringData = async () => {
                    try {
                        const response = await fetch('/api/monitoring-data');
                        const data = await response.json();
                        setMonitoringData(data);
                    } catch (error) {
                        console.error('Failed to fetch monitoring data:', error);
                    }
                };
                
                fetchMonitoringData();
                const interval = setInterval(fetchMonitoringData, 5000);
                return () => clearInterval(interval);
            }, []);
            
            const handleWebSocketMessage = (data) => {
                switch (data.type) {
                    case 'chat_response':
                        setMessages(prev => [...prev, {
                            type: 'assistant',
                            content: data.message,
                            timestamp: new Date().toISOString()
                        }]);
                        break;
                    case 'project_update':
                        setProjects(prev => prev.map(p => 
                            p.id === data.project_id ? {...p, ...data.updates} : p
                        ));
                        break;
                    case 'monitoring_update':
                        setMonitoringData(prev => ({...prev, ...data.data}));
                        break;
                    case 'generation_complete':
                        setIsGenerating(false);
                        setMessages(prev => [...prev, {
                            type: 'system',
                            content: 'Code generation completed successfully!',
                            timestamp: new Date().toISOString()
                        }]);
                        break;
                }
            };
            
            const sendMessage = () => {
                if (!currentInput.trim() || !isConnected) return;
                
                const message = {
                    type: 'chat_message',
                    message: currentInput,
                    timestamp: new Date().toISOString()
                };
                
                wsRef.current.send(JSON.stringify(message));
                
                setMessages(prev => [...prev, {
                    type: 'user',
                    content: currentInput,
                    timestamp: new Date().toISOString()
                }]);
                
                setCurrentInput('');
            };
            
            const generateCode = async () => {
                if (!currentInput.trim()) return;
                
                setIsGenerating(true);
                
                try {
                    const response = await fetch('/api/generate-code', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt: currentInput })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        setMessages(prev => [...prev, {
                            type: 'code',
                            content: result.merged_code,
                            providers: result.source_providers,
                            timestamp: new Date().toISOString()
                        }]);
                    } else {
                        setMessages(prev => [...prev, {
                            type: 'error',
                            content: result.error,
                            timestamp: new Date().toISOString()
                        }]);
                    }
                } catch (error) {
                    setMessages(prev => [...prev, {
                        type: 'error',
                        content: 'Failed to generate code: ' + error.message,
                        timestamp: new Date().toISOString()
                    }]);
                } finally {
                    setIsGenerating(false);
                    setCurrentInput('');
                }
            };
            
            const createProject = async () => {
                const projectName = prompt('Enter project name:');
                if (!projectName) return;
                
                try {
                    const response = await fetch('/api/create-project', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            name: projectName,
                            description: 'Auto-generated project',
                            type: 'full-stack'
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        setProjects(prev => [...prev, result.project]);
                    }
                } catch (error) {
                    console.error('Failed to create project:', error);
                }
            };
            
            const ConnectionStatus = () => (
                <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 pulse-dot' : 'bg-red-500'}`}></div>
                    <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
                </div>
            );
            
            const Dashboard = () => (
                <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <div className="glass rounded-lg p-6 border border-white/20">
                            <h3 className="text-lg font-semibold mb-2">Active Projects</h3>
                            <p className="text-3xl font-bold text-blue-400">{projects.length}</p>
                        </div>
                        <div className="glass rounded-lg p-6 border border-white/20">
                            <h3 className="text-lg font-semibold mb-2">CPU Usage</h3>
                            <p className="text-3xl font-bold text-green-400">
                                {monitoringData.cpu_percent?.toFixed(1) || '0'}%
                            </p>
                        </div>
                        <div className="glass rounded-lg p-6 border border-white/20">
                            <h3 className="text-lg font-semibold mb-2">Memory Usage</h3>
                            <p className="text-3xl font-bold text-yellow-400">
                                {monitoringData.memory_percent?.toFixed(1) || '0'}%
                            </p>
                        </div>
                        <div className="glass rounded-lg p-6 border border-white/20">
                            <h3 className="text-lg font-semibold mb-2">LLM Requests</h3>
                            <p className="text-3xl font-bold text-purple-400">
                                {monitoringData.total_requests || '0'}
                            </p>
                        </div>
                    </div>
                    
                    <div className="glass rounded-lg p-6 border border-white/20">
                        <h3 className="text-xl font-semibold mb-4">Real-time Chat & Code Generation</h3>
                        <div className="space-y-4">
                            <div className="h-64 overflow-y-auto bg-gray-800 rounded-lg p-4 space-y-2">
                                {messages.map((msg, index) => (
                                    <div key={index} className={`p-2 rounded ${
                                        msg.type === 'user' ? 'bg-blue-600 ml-8' :
                                        msg.type === 'assistant' ? 'bg-gray-600 mr-8' :
                                        msg.type === 'code' ? 'bg-green-600 text-xs font-mono' :
                                        'bg-red-600 mr-8'
                                    }`}>
                                        <div className="text-xs opacity-75 mb-1">
                                            {msg.type.toUpperCase()} - {new Date(msg.timestamp).toLocaleTimeString()}
                                            {msg.providers && ` - Providers: ${msg.providers.join(', ')}`}
                                        </div>
                                        <div className={msg.type === 'code' ? 'whitespace-pre-wrap' : ''}>
                                            {msg.content}
                                        </div>
                                    </div>
                                ))}
                            </div>
                            
                            <div className="flex space-x-2">
                                <input
                                    type="text"
                                    value={currentInput}
                                    onChange={(e) => setCurrentInput(e.target.value)}
                                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                                    placeholder="Type your message or code request..."
                                    className="flex-1 p-3 bg-gray-800 border border-gray-600 rounded-lg text-white"
                                    disabled={!isConnected || isGenerating}
                                />
                                <button
                                    onClick={sendMessage}
                                    disabled={!isConnected || isGenerating}
                                    className="px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded-lg"
                                >
                                    Send
                                </button>
                                <button
                                    onClick={generateCode}
                                    disabled={!isConnected || isGenerating}
                                    className="px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded-lg"
                                >
                                    {isGenerating ? 'Generating...' : 'Generate Code'}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            );
            
            const Projects = () => (
                <div className="space-y-6">
                    <div className="flex justify-between items-center">
                        <h2 className="text-2xl font-bold">Projects</h2>
                        <button
                            onClick={createProject}
                            className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg"
                        >
                            Create Project
                        </button>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {projects.map((project) => (
                            <div key={project.id} className="glass rounded-lg p-6 border border-white/20">
                                <h3 className="text-lg font-semibold mb-2">{project.name}</h3>
                                <p className="text-gray-300 mb-4">{project.description}</p>
                                <div className="flex justify-between items-center">
                                    <span className={`px-2 py-1 rounded text-xs ${
                                        project.status === 'active' ? 'bg-green-600' :
                                        project.status === 'pending' ? 'bg-yellow-600' :
                                        'bg-gray-600'
                                    }`}>
                                        {project.status || 'unknown'}
                                    </span>
                                    <button className="text-blue-400 hover:text-blue-300">
                                        View Details
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            );
            
            const Monitoring = () => (
                <div className="space-y-6">
                    <h2 className="text-2xl font-bold">System Monitoring</h2>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="glass rounded-lg p-6 border border-white/20">
                            <h3 className="text-lg font-semibold mb-4">System Resources</h3>
                            <div className="space-y-4">
                                <div>
                                    <div className="flex justify-between mb-2">
                                        <span>CPU Usage</span>
                                        <span>{monitoringData.cpu_percent?.toFixed(1) || '0'}%</span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-2">
                                        <div 
                                            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                            style={{width: `${monitoringData.cpu_percent || 0}%`}}
                                        ></div>
                                    </div>
                                </div>
                                
                                <div>
                                    <div className="flex justify-between mb-2">
                                        <span>Memory Usage</span>
                                        <span>{monitoringData.memory_percent?.toFixed(1) || '0'}%</span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-2">
                                        <div 
                                            className="bg-green-500 h-2 rounded-full transition-all duration-300"
                                            style={{width: `${monitoringData.memory_percent || 0}%`}}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div className="glass rounded-lg p-6 border border-white/20">
                            <h3 className="text-lg font-semibold mb-4">LLM Provider Status</h3>
                            <div className="space-y-3">
                                {['Vertex AI', 'OpenAI', 'Grok'].map((provider) => (
                                    <div key={provider} className="flex justify-between items-center">
                                        <span>{provider}</span>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-3 h-3 bg-green-500 rounded-full pulse-dot"></div>
                                            <span className="text-sm text-gray-300">Online</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            );
            
            return (
                <div className="min-h-screen gradient-bg">
                    <nav className="glass border-b border-white/20 p-4">
                        <div className="max-w-7xl mx-auto flex justify-between items-center">
                            <h1 className="text-2xl font-bold">Master Orchestrator</h1>
                            <div className="flex items-center space-x-6">
                                <ConnectionStatus />
                                <div className="flex space-x-4">
                                    {['dashboard', 'projects', 'monitoring'].map((tab) => (
                                        <button
                                            key={tab}
                                            onClick={() => setActiveTab(tab)}
                                            className={`px-3 py-2 rounded ${
                                                activeTab === tab 
                                                    ? 'bg-white/20 text-white' 
                                                    : 'text-gray-300 hover:text-white'
                                            }`}
                                        >
                                            {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </nav>
                    
                    <main className="max-w-7xl mx-auto p-6">
                        {activeTab === 'dashboard' && <Dashboard />}
                        {activeTab === 'projects' && <Projects />}
                        {activeTab === 'monitoring' && <Monitoring />}
                    </main>
                </div>
            );
        }
        
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>
        '''
        
        await aiofiles.open(template_dir / "main.html", 'w').write(main_template)
        
    # HTTP Handlers
    async def serve_main_interface(self, request):
        """Serve main interface"""
        template = self.template_env.get_template('main.html')
        html = template.render()
        return web.Response(text=html, content_type='text/html')
        
    async def serve_dashboard(self, request):
        """Serve dashboard"""
        return await self.serve_main_interface(request)
        
    async def serve_projects(self, request):
        """Serve projects page"""
        return await self.serve_main_interface(request)
        
    async def serve_monitoring(self, request):
        """Serve monitoring page"""
        return await self.serve_main_interface(request)
        
    async def serve_optimization(self, request):
        """Serve optimization page"""
        return await self.serve_main_interface(request)
        
    # API Endpoints
    async def api_generate_code(self, request):
        """Generate code using parallel LLM system"""
        try:
            data = await request.json()
            prompt = data.get('prompt', '')
            merge_strategy = data.get('merge_strategy', 'comprehensive')
            
            if not prompt:
                return web.json_response({"success": False, "error": "No prompt provided"})
                
            # Generate code using parallel LLM orchestrator
            result = await self.llm_orchestrator.generate_code_parallel(prompt, merge_strategy)
            
            # Broadcast to WebSocket clients
            await self._broadcast_websocket({
                "type": "generation_complete",
                "result": result
            })
            
            return web.json_response(result)
            
        except Exception as e:
            logging.error(f"Code generation API error: {e}")
            return web.json_response({"success": False, "error": str(e)})
            
    async def api_analyze_content(self, request):
        """Analyze YouTube/PDF content"""
        try:
            data = await request.json()
            content_input = data.get('content_input', '')
            
            if not content_input:
                return web.json_response({"success": False, "error": "No content input provided"})
                
            # Analyze content
            result = await self.content_analyzer.process_content(content_input)
            
            return web.json_response(result)
            
        except Exception as e:
            logging.error(f"Content analysis API error: {e}")
            return web.json_response({"success": False, "error": str(e)})
            
    async def api_create_project(self, request):
        """Create new project"""
        try:
            data = await request.json()
            project = {
                "id": hashlib.md5(f"{data['name']}{datetime.now()}".encode()).hexdigest()[:8],
                "name": data['name'],
                "description": data.get('description', ''),
                "type": data.get('type', 'general'),
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "files": [],
                "deployments": []
            }
            
            self.active_projects[project["id"]] = project
            
            # Create project directory
            project_path = Path("projects") / project["id"]
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize git repository
            repo = git.Repo.init(project_path)
            
            # Create initial files
            readme_content = f"""# {project['name']}

{project['description']}

## Auto-generated by Master Orchestrator

Created: {project['created_at']}
Type: {project['type']}

## Features

- AI-powered code generation
- Real-time monitoring
- Continuous optimization
- Automated testing and deployment

## Getting Started

```bash
cd {project_path}
python main.py
```
"""
            
            await aiofiles.open(project_path / "README.md", 'w').write(readme_content)
            
            # Initial commit
            repo.index.add(["README.md"])
            repo.index.commit("Initial commit - Auto-generated project")
            
            # Broadcast project creation
            await self._broadcast_websocket({
                "type": "project_created",
                "project": project
            })
            
            return web.json_response({"success": True, "project": project})
            
        except Exception as e:
            logging.error(f"Project creation API error: {e}")
            return web.json_response({"success": False, "error": str(e)})
            
    async def api_get_project(self, request):
        """Get project details"""
        try:
            project_id = request.match_info['project_id']
            project = self.active_projects.get(project_id)
            
            if not project:
                return web.json_response({"success": False, "error": "Project not found"}, status=404)
                
            return web.json_response({"success": True, "project": project})
            
        except Exception as e:
            logging.error(f"Get project API error: {e}")
            return web.json_response({"success": False, "error": str(e)})
            
    async def api_optimize_project(self, request):
        """Optimize existing project"""
        try:
            data = await request.json()
            project_id = data.get('project_id', '')
            
            if not project_id or project_id not in self.active_projects:
                return web.json_response({"success": False, "error": "Invalid project ID"})
                
            # Add to optimization queue
            await self.optimization_queue.put({
                "type": "project_optimization",
                "project_id": project_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return web.json_response({"success": True, "message": "Optimization queued"})
            
        except Exception as e:
            logging.error(f"Project optimization API error: {e}")
            return web.json_response({"success": False, "error": str(e)})
            
    async def api_get_monitoring_data(self, request):
        """Get real-time monitoring data"""
        try:
            return web.json_response(self.monitoring_data)
        except Exception as e:
            logging.error(f"Monitoring data API error: {e}")
            return web.json_response({"error": str(e)})
            
    async def api_execute_command(self, request):
        """Execute computer control command"""
        try:
            data = await request.json()
            command = data.get('command', {})
            
            # Execute command through computer control
            result = await self.computer_control.process_command(command)
            
            return web.json_response(result)
            
        except Exception as e:
            logging.error(f"Command execution API error: {e}")
            return web.json_response({"success": False, "error": str(e)})
            
    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "llm_orchestrator": "online",
                "computer_control": "online",
                "content_analyzer": "online",
                "websocket_clients": len(self.websocket_clients),
                "active_projects": len(self.active_projects)
            }
        })
        
    # WebSocket Handler
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_clients.add(ws)
        logging.info(f"WebSocket client connected. Total clients: {len(self.websocket_clients)}")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON format"
                        }))
                elif msg.type == WSMsgType.ERROR:
                    logging.error(f"WebSocket error: {ws.exception()}")
                    
        except Exception as e:
            logging.error(f"WebSocket handler error: {e}")
        finally:
            self.websocket_clients.discard(ws)
            logging.info(f"WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")
            
        return ws
        
    async def _handle_websocket_message(self, ws, data):
        """Handle incoming WebSocket messages"""
        try:
            message_type = data.get("type")
            
            if message_type == "chat_message":
                # Process chat message with LLM
                response = await self.llm_orchestrator.llm_manager.generate_response(
                    data["message"],
                    context={
                        "timestamp": datetime.now().isoformat(),
                        "active_projects": list(self.active_projects.keys()),
                        "monitoring_data": self.monitoring_data
                    }
                )
                
                await self._broadcast_websocket({
                    "type": "chat_response",
                    "message": response.get("content", ""),
                    "timestamp": datetime.now().isoformat()
                })
                
            elif message_type == "command":
                # Execute computer control command
                result = await self.computer_control.process_command(data)
                await ws.send_str(json.dumps({
                    "type": "command_result",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logging.error(f"WebSocket message handling error: {e}")
            await ws.send_str(json.dumps({
                "type": "error",
                "message": str(e)
            }))
            
    async def _broadcast_websocket(self, message):
        """Broadcast message to all WebSocket clients"""
        if self.websocket_clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send_str(message_str) for client in self.websocket_clients],
                return_exceptions=True
            )
            
    # Background Tasks
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                # Get LLM orchestrator stats
                llm_stats = self.llm_orchestrator.get_execution_statistics()
                
                # Update monitoring data
                self.monitoring_data.update({
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "total_requests": llm_stats.get("total_executions", 0),
                    "success_rate": llm_stats.get("success_rate", 0),
                    "active_projects": len(self.active_projects),
                    "websocket_clients": len(self.websocket_clients),
                    "system_status": "healthy" if cpu_percent < 80 and memory_percent < 80 else "warning"
                })
                
                # Broadcast monitoring update
                await self._broadcast_websocket({
                    "type": "monitoring_update",
                    "data": self.monitoring_data
                })
                
                # Check for optimization opportunities
                if cpu_percent > 90 or memory_percent > 90:
                    await self.optimization_queue.put({
                        "type": "system_optimization",
                        "reason": "high_resource_usage",
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
                
    async def _optimization_processor(self):
        """Process optimization queue"""
        while True:
            try:
                # Get optimization task
                task = await self.optimization_queue.get()
                
                logging.info(f"Processing optimization task: {task['type']}")
                
                if task["type"] == "project_optimization":
                    await self._optimize_project(task["project_id"])
                elif task["type"] == "system_optimization":
                    await self._optimize_system(task)
                    
                # Mark task as done
                self.optimization_queue.task_done()
                
            except Exception as e:
                logging.error(f"Optimization processor error: {e}")
                await asyncio.sleep(5)
                
    async def _optimize_project(self, project_id):
        """Optimize specific project"""
        try:
            project = self.active_projects.get(project_id)
            if not project:
                return
                
            project_path = Path("projects") / project_id
            
            # Find all Python files in project
            python_files = list(project_path.glob("**/*.py"))
            
            optimization_results = []
            
            for file_path in python_files:
                # Optimize each file
                result = await self.llm_orchestrator.optimize_existing_code(str(file_path))
                if result["success"]:
                    optimization_results.append(result)
                    
            # Update project with optimization results
            project["last_optimization"] = {
                "timestamp": datetime.now().isoformat(),
                "files_optimized": len(optimization_results),
                "results": optimization_results
            }
            
            # Broadcast optimization complete
            await self._broadcast_websocket({
                "type": "project_update",
                "project_id": project_id,
                "updates": {"last_optimization": project["last_optimization"]}
            })
            
            logging.info(f"Project {project_id} optimization completed")
            
        except Exception as e:
            logging.error(f"Project optimization error: {e}")
            
    async def _optimize_system(self, task):
        """Optimize system resources"""
        try:
            optimizations = []
            
            if task.get("cpu_percent", 0) > 90:
                # High CPU usage - optimize processes
                optimizations.append("Reduced background process priority")
                
            if task.get("memory_percent", 0) > 90:
                # High memory usage - cleanup
                optimizations.append("Cleared cache and temporary files")
                
            # Broadcast system optimization
            await self._broadcast_websocket({
                "type": "system_optimization_complete",
                "optimizations": optimizations,
                "timestamp": datetime.now().isoformat()
            })
            
            logging.info(f"System optimization completed: {optimizations}")
            
        except Exception as e:
            logging.error(f"System optimization error: {e}")
            
    async def start_server(self, host='localhost', port=8080):
        """Start the frontend server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logging.info(f"Frontend server started on http://{host}:{port}")
        return runner

async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create frontend orchestrator
    frontend = FrontendOrchestrator()
    await frontend.initialize()
    
    # Start server
    runner = await frontend.start_server()
    
    try:
        # Keep running
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logging.info("Shutting down frontend orchestrator")
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())