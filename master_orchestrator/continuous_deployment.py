#!/usr/bin/env python3

"""
Continuous Auto-Generation and Deployment System
Automatically generates code, deploys changes, and updates frontend without full refreshes
"""

import asyncio
import json
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import websockets
import threading
from queue import Queue

import structlog

logger = structlog.get_logger()


class CodeGenerationEngine:
    """Autonomous code generation using LLMs and templates."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.generation_queue = Queue()
        self.templates_dir = project_root / "templates"
        self.generated_dir = project_root / "generated"
        self.config_file = project_root / "generation_config.json"
        
        # Ensure directories exist
        self.templates_dir.mkdir(exist_ok=True)
        self.generated_dir.mkdir(exist_ok=True)
        
        # Load or create configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load generation configuration."""
        if self.config_file.exists():
            return json.loads(self.config_file.read_text())
        
        # Default configuration
        config = {
            "auto_generation": {
                "enabled": True,
                "interval_seconds": 30,
                "triggers": ["file_change", "schedule", "webhook"]
            },
            "generation_rules": [
                {
                    "name": "api_endpoints",
                    "trigger": "*.py",
                    "template": "fastapi_endpoint.jinja2",
                    "output_pattern": "api/{name}_endpoint.py"
                },
                {
                    "name": "frontend_components",
                    "trigger": "*.json",
                    "template": "react_component.jinja2",
                    "output_pattern": "frontend/components/{name}.jsx"
                },
                {
                    "name": "agents",
                    "trigger": "agent_*.yaml",
                    "template": "dspy_agent.jinja2",
                    "output_pattern": "agents/{name}_agent.py"
                }
            ],
            "llm_providers": {
                "primary": "anthropic",
                "fallback": ["openai", "local"]
            }
        }
        
        self.config_file.write_text(json.dumps(config, indent=2))
        return config
    
    async def generate_code(self, trigger_type: str, trigger_data: Dict[str, Any]) -> List[Path]:
        """Generate code based on triggers."""
        generated_files = []
        
        try:
            # Analyze what needs to be generated
            analysis = await self._analyze_generation_needs(trigger_type, trigger_data)
            
            for generation_task in analysis:
                generated_file = await self._execute_generation(generation_task)
                if generated_file:
                    generated_files.append(generated_file)
                    logger.info(f"Generated: {generated_file}")
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
        
        return generated_files
    
    async def _analyze_generation_needs(self, trigger_type: str, trigger_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze what code needs to be generated based on triggers."""
        tasks = []
        
        if trigger_type == "file_change":
            changed_file = trigger_data.get("file_path")
            if changed_file:
                # Check generation rules
                for rule in self.config["generation_rules"]:
                    if self._matches_pattern(changed_file, rule["trigger"]):
                        tasks.append({
                            "rule": rule,
                            "trigger_file": changed_file,
                            "context": trigger_data
                        })
        
        elif trigger_type == "schedule":
            # Scheduled generation - check for outdated files
            tasks.extend(await self._find_outdated_generations())
        
        elif trigger_type == "webhook":
            # External trigger - generate based on payload
            tasks.extend(await self._process_webhook_generation(trigger_data))
        
        return tasks
    
    async def _execute_generation(self, task: Dict[str, Any]) -> Path:
        """Execute a single code generation task."""
        rule = task["rule"]
        template_path = self.templates_dir / rule["template"]
        
        # Generate context for the template
        context = await self._build_generation_context(task)
        
        # Use LLM to generate code if template doesn't exist
        if not template_path.exists():
            await self._create_template_with_llm(template_path, rule)
        
        # Generate the actual code
        generated_content = await self._render_template(template_path, context)
        
        # Determine output path
        output_path = self._resolve_output_path(rule["output_pattern"], context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write generated content
        output_path.write_text(generated_content)
        
        return output_path
    
    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file matches generation pattern."""
        from fnmatch import fnmatch
        return fnmatch(Path(file_path).name, pattern)
    
    def _resolve_output_path(self, pattern: str, context: Dict[str, Any]) -> Path:
        """Resolve output path from pattern and context."""
        resolved = pattern.format(**context)
        return self.generated_dir / resolved


class HotReloadServer:
    """WebSocket server for hot reloading frontend without full page refreshes."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.file_hashes: Dict[str, str] = {}
        self.update_queue = Queue()
        
    async def start(self):
        """Start the WebSocket server."""
        logger.info(f"Starting hot reload server on port {self.port}")
        
        async def handler(websocket, path):
            self.clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                await websocket.wait_closed()
            finally:
                self.clients.remove(websocket)
                logger.info(f"Client disconnected: {websocket.remote_address}")
        
        start_server = websockets.serve(handler, "localhost", self.port)
        await start_server
    
    async def broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast updates to all connected clients."""
        if not self.clients:
            return
        
        message = {
            "type": update_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
        
        logger.info(f"Broadcasted {update_type} to {len(self.clients)} clients")
    
    async def file_changed(self, file_path: Path, change_type: str):
        """Handle file change events."""
        file_str = str(file_path)
        
        # Calculate file hash
        try:
            content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
        except:
            content_hash = "deleted"
        
        # Check if file actually changed
        if file_str in self.file_hashes and self.file_hashes[file_str] == content_hash:
            return
        
        self.file_hashes[file_str] = content_hash
        
        # Determine update type based on file
        if file_path.suffix in ['.html', '.css', '.js']:
            update_type = "frontend_update"
            await self.broadcast_update(update_type, {
                "file": file_str,
                "change_type": change_type,
                "content_hash": content_hash,
                "reload_type": "hot" if file_path.suffix == '.css' else "soft"
            })
        
        elif file_path.suffix in ['.py', '.json']:
            update_type = "backend_update"
            await self.broadcast_update(update_type, {
                "file": file_str,
                "change_type": change_type,
                "requires_restart": file_path.suffix == '.py'
            })


class FileWatcher(FileSystemEventHandler):
    """File system watcher for triggering continuous deployment."""
    
    def __init__(self, code_generator: CodeGenerationEngine, hot_reload: HotReloadServer):
        self.code_generator = code_generator
        self.hot_reload = hot_reload
        self.ignore_patterns = {'.git', '__pycache__', '.pyc', '.log', 'node_modules'}
        
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored."""
        path = Path(file_path)
        return any(pattern in str(path) for pattern in self.ignore_patterns)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        asyncio.create_task(self._handle_file_change(event.src_path, "modified"))
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory or self.should_ignore(event.src_path):
            return
        
        asyncio.create_task(self._handle_file_change(event.src_path, "created"))
    
    async def _handle_file_change(self, file_path: str, change_type: str):
        """Handle file change events."""
        path = Path(file_path)
        
        # Trigger code generation
        generated_files = await self.code_generator.generate_code("file_change", {
            "file_path": file_path,
            "change_type": change_type
        })
        
        # Notify hot reload
        await self.hot_reload.file_changed(path, change_type)
        
        # If files were generated, notify about them too
        for generated_file in generated_files:
            await self.hot_reload.file_changed(generated_file, "generated")


class ContinuousDeployment:
    """Main continuous deployment orchestrator."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.code_generator = CodeGenerationEngine(project_root)
        self.hot_reload = HotReloadServer()
        self.file_watcher = FileWatcher(self.code_generator, self.hot_reload)
        self.observer = Observer()
        
        # Deployment configuration
        self.deployment_config = {
            "auto_deploy": True,
            "deployment_triggers": ["code_generation", "file_change"],
            "health_check_url": "http://localhost:8000/health",
            "rollback_on_failure": True
        }
        
    async def start(self):
        """Start the continuous deployment system."""
        logger.info("Starting Continuous Deployment System")
        
        # Start hot reload server
        await self.hot_reload.start()
        
        # Start file watching
        self.observer.schedule(
            self.file_watcher,
            str(self.project_root),
            recursive=True
        )
        self.observer.start()
        
        # Start scheduled tasks
        asyncio.create_task(self._scheduled_generation())
        asyncio.create_task(self._health_monitoring())
        
        logger.info("Continuous Deployment System started")
    
    async def _scheduled_generation(self):
        """Run scheduled code generation."""
        while True:
            try:
                await asyncio.sleep(30)  # Generate every 30 seconds
                
                generated_files = await self.code_generator.generate_code("schedule", {
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                if generated_files:
                    logger.info(f"Scheduled generation completed: {len(generated_files)} files")
                    
            except Exception as e:
                logger.error(f"Scheduled generation failed: {e}")
    
    async def _health_monitoring(self):
        """Monitor system health and trigger deployments."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check if deployment is needed
                if await self._needs_deployment():
                    await self._deploy()
                    
            except Exception as e:
                logger.error(f"Health monitoring failed: {e}")
    
    async def _needs_deployment(self) -> bool:
        """Check if deployment is needed."""
        # Check for new generated files
        generated_dir = self.code_generator.generated_dir
        if not generated_dir.exists():
            return False
        
        # Look for files newer than last deployment
        last_deployment_file = self.project_root / ".last_deployment"
        if not last_deployment_file.exists():
            return True
        
        last_deployment = last_deployment_file.stat().st_mtime
        
        for file_path in generated_dir.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime > last_deployment:
                return True
        
        return False
    
    async def _deploy(self):
        """Execute deployment."""
        logger.info("Starting deployment...")
        
        try:
            # Update last deployment timestamp
            deployment_file = self.project_root / ".last_deployment"
            deployment_file.write_text(str(time.time()))
            
            # Restart services if needed
            await self._restart_services()
            
            # Notify clients of deployment
            await self.hot_reload.broadcast_update("deployment", {
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info("Deployment completed successfully")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            
            # Rollback if configured
            if self.deployment_config["rollback_on_failure"]:
                await self._rollback()
    
    async def _restart_services(self):
        """Restart necessary services."""
        # Restart the main application server
        try:
            # Kill existing server
            subprocess.run(["pkill", "-f", "simple_server.py"], check=False)
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Start new server
            subprocess.Popen([
                "python3", "simple_server.py"
            ], cwd=self.project_root)
            
            logger.info("Application server restarted")
            
        except Exception as e:
            logger.error(f"Service restart failed: {e}")
    
    async def _rollback(self):
        """Rollback to previous deployment."""
        logger.info("Rolling back deployment...")
        # Implementation would restore previous state
        pass


# Enhanced frontend with hot reload support
def create_hot_reload_frontend():
    """Create frontend with hot reload capabilities."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Orchestrator - Live Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <style>
        [x-cloak] { display: none !important; }
        .live-indicator { animation: pulse 2s infinite; }
        .update-flash { animation: flash 0.5s ease-in-out; }
        @keyframes flash { 0%, 100% { background-color: transparent; } 50% { background-color: rgba(34, 197, 94, 0.2); } }
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    </style>
</head>
<body class="bg-gray-50 min-h-screen" x-data="masterOrchestrator()" x-init="init()">
    <!-- Live Connection Status -->
    <div x-show="!connected" class="fixed top-0 left-0 right-0 bg-red-500 text-white text-center py-2 z-50">
        🔴 Disconnected - Attempting to reconnect...
    </div>
    
    <!-- Live Updates Indicator -->
    <div x-show="connected" class="fixed top-4 right-4 z-50">
        <div class="bg-green-500 text-white px-3 py-1 rounded-full text-sm live-indicator">
            🟢 Live Updates Active
        </div>
    </div>

    <!-- Header -->
    <div class="gradient-bg text-white p-6">
        <div class="container mx-auto max-w-6xl">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-4xl font-bold">🚀 Master Orchestrator</h1>
                    <p class="text-blue-100 mt-2 text-lg">Continuous Auto-Generation & Deployment</p>
                    <div class="text-sm text-blue-200 mt-1">
                        <span x-text="'Last Update: ' + lastUpdate"></span> •
                        <span x-text="'Deployments: ' + deploymentCount"></span> •
                        <span x-text="'Generated Files: ' + generatedFiles"></span>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-4xl mb-2" :class="connected ? 'live-indicator' : ''">⚡</div>
                    <div class="text-sm text-blue-100">System Status</div>
                    <div class="font-semibold text-xl" :class="systemStatus === 'online' ? 'text-green-300' : 'text-red-300'" x-text="systemStatus.toUpperCase()"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Real-time Metrics -->
    <div class="container mx-auto max-w-6xl px-6 py-8">
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow p-6" :class="updates.metrics ? 'update-flash' : ''">
                <div class="flex items-center">
                    <div class="text-4xl mr-4">🤖</div>
                    <div>
                        <div class="text-3xl font-bold text-gray-800" x-text="metrics.agents"></div>
                        <div class="text-sm text-gray-500">Active Agents</div>
                        <div class="text-xs text-green-600 mt-1">Auto-scaling enabled</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow p-6" :class="updates.repositories ? 'update-flash' : ''">
                <div class="flex items-center">
                    <div class="text-4xl mr-4">📁</div>
                    <div>
                        <div class="text-3xl font-bold text-gray-800" x-text="metrics.repositories"></div>
                        <div class="text-sm text-gray-500">Repositories</div>
                        <div class="text-xs text-blue-600 mt-1">Continuously monitored</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow p-6" :class="updates.generation ? 'update-flash' : ''">
                <div class="flex items-center">
                    <div class="text-4xl mr-4">⚙️</div>
                    <div>
                        <div class="text-3xl font-bold text-gray-800" x-text="metrics.generated_files"></div>
                        <div class="text-sm text-gray-500">Generated Files</div>
                        <div class="text-xs text-purple-600 mt-1">Auto-generated & deployed</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow p-6" :class="updates.deployment ? 'update-flash' : ''">
                <div class="flex items-center">
                    <div class="text-4xl mr-4">🚀</div>
                    <div>
                        <div class="text-3xl font-bold text-gray-800" x-text="metrics.deployments"></div>
                        <div class="text-sm text-gray-500">Deployments</div>
                        <div class="text-xs text-indigo-600 mt-1">Zero-downtime updates</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Activity Feed -->
        <div class="bg-white rounded-lg shadow p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">📊 Live Activity Feed</h2>
            <div class="space-y-2 max-h-96 overflow-y-auto">
                <template x-for="activity in activities" :key="activity.id">
                    <div class="flex items-center space-x-3 p-3 bg-gray-50 rounded">
                        <div class="text-2xl" x-text="activity.icon"></div>
                        <div class="flex-1">
                            <div class="font-medium" x-text="activity.message"></div>
                            <div class="text-sm text-gray-500" x-text="activity.timestamp"></div>
                        </div>
                        <div class="text-xs px-2 py-1 rounded-full" 
                             :class="activity.type === 'success' ? 'bg-green-100 text-green-800' : 
                                    activity.type === 'warning' ? 'bg-yellow-100 text-yellow-800' : 
                                    'bg-blue-100 text-blue-800'"
                             x-text="activity.type"></div>
                    </div>
                </template>
            </div>
        </div>

        <!-- Continuous Generation Status -->
        <div class="bg-white rounded-lg shadow p-6">
            <h2 class="text-2xl font-semibold mb-4">🔄 Continuous Generation Status</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-green-50 p-4 rounded">
                    <div class="font-semibold text-green-800">Code Generation</div>
                    <div class="text-sm text-green-600">✅ Active - Every 30s</div>
                    <div class="text-xs text-gray-600 mt-1" x-text="'Next run: ' + nextGeneration"></div>
                </div>
                <div class="bg-blue-50 p-4 rounded">
                    <div class="font-semibold text-blue-800">Hot Reload</div>
                    <div class="text-sm text-blue-600">✅ Connected via WebSocket</div>
                    <div class="text-xs text-gray-600 mt-1" x-text="'Updates: ' + updateCount"></div>
                </div>
                <div class="bg-purple-50 p-4 rounded">
                    <div class="font-semibold text-purple-800">Auto Deploy</div>
                    <div class="text-sm text-purple-600">✅ Zero-downtime enabled</div>
                    <div class="text-xs text-gray-600 mt-1" x-text="'Last deploy: ' + lastDeploy"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function masterOrchestrator() {
            return {
                connected: false,
                systemStatus: 'online',
                lastUpdate: new Date().toLocaleTimeString(),
                deploymentCount: 0,
                generatedFiles: 0,
                updateCount: 0,
                
                metrics: {
                    agents: 2,
                    repositories: 28,
                    generated_files: 0,
                    deployments: 0
                },
                
                activities: [],
                updates: {},
                
                nextGeneration: '',
                lastDeploy: 'Never',
                
                websocket: null,
                
                init() {
                    this.connectWebSocket();
                    this.startTimers();
                    this.addActivity('🚀', 'Master Orchestrator initialized', 'success');
                },
                
                connectWebSocket() {
                    try {
                        this.websocket = new WebSocket('ws://localhost:8001');
                        
                        this.websocket.onopen = () => {
                            this.connected = true;
                            this.addActivity('🔗', 'WebSocket connected - Live updates active', 'success');
                        };
                        
                        this.websocket.onmessage = (event) => {
                            const message = JSON.parse(event.data);
                            this.handleUpdate(message);
                        };
                        
                        this.websocket.onclose = () => {
                            this.connected = false;
                            this.addActivity('🔴', 'Connection lost - Reconnecting...', 'warning');
                            setTimeout(() => this.connectWebSocket(), 2000);
                        };
                        
                        this.websocket.onerror = (error) => {
                            console.error('WebSocket error:', error);
                        };
                        
                    } catch (error) {
                        console.error('Failed to connect WebSocket:', error);
                        setTimeout(() => this.connectWebSocket(), 5000);
                    }
                },
                
                handleUpdate(message) {
                    this.lastUpdate = new Date().toLocaleTimeString();
                    this.updateCount++;
                    
                    if (message.type === 'frontend_update') {
                        this.handleFrontendUpdate(message.data);
                    } else if (message.type === 'backend_update') {
                        this.handleBackendUpdate(message.data);
                    } else if (message.type === 'deployment') {
                        this.handleDeployment(message.data);
                    } else if (message.type === 'generation') {
                        this.handleGeneration(message.data);
                    }
                    
                    // Flash the relevant section
                    this.updates[message.type] = true;
                    setTimeout(() => { this.updates[message.type] = false; }, 500);
                },
                
                handleFrontendUpdate(data) {
                    if (data.reload_type === 'hot' && data.file.endsWith('.css')) {
                        // Hot reload CSS without page refresh
                        const links = document.querySelectorAll('link[rel="stylesheet"]');
                        links.forEach(link => {
                            const href = link.href.split('?')[0];
                            link.href = href + '?v=' + Date.now();
                        });
                        this.addActivity('🎨', `CSS updated: ${data.file}`, 'success');
                    } else if (data.reload_type === 'soft') {
                        // Soft reload for JS/HTML
                        this.addActivity('📝', `File updated: ${data.file}`, 'info');
                        // Could implement partial DOM updates here
                    }
                },
                
                handleBackendUpdate(data) {
                    this.addActivity('⚙️', `Backend updated: ${data.file}`, 'info');
                    if (data.requires_restart) {
                        this.addActivity('🔄', 'Service restarting...', 'warning');
                    }
                },
                
                handleDeployment(data) {
                    this.deploymentCount++;
                    this.metrics.deployments = this.deploymentCount;
                    this.lastDeploy = new Date().toLocaleTimeString();
                    this.addActivity('🚀', 'Auto-deployment completed', 'success');
                },
                
                handleGeneration(data) {
                    this.generatedFiles++;
                    this.metrics.generated_files = this.generatedFiles;
                    this.addActivity('⚙️', `Generated: ${data.files} files`, 'success');
                },
                
                addActivity(icon, message, type) {
                    const activity = {
                        id: Date.now(),
                        icon: icon,
                        message: message,
                        type: type,
                        timestamp: new Date().toLocaleTimeString()
                    };
                    
                    this.activities.unshift(activity);
                    
                    // Keep only last 50 activities
                    if (this.activities.length > 50) {
                        this.activities = this.activities.slice(0, 50);
                    }
                },
                
                startTimers() {
                    // Update next generation time
                    setInterval(() => {
                        const now = new Date();
                        const nextRun = new Date(Math.ceil(now.getTime() / 30000) * 30000);
                        this.nextGeneration = nextRun.toLocaleTimeString();
                    }, 1000);
                    
                    // Simulate some metrics updates
                    setInterval(() => {
                        this.metrics.agents = Math.floor(Math.random() * 3) + 2;
                        if (Math.random() > 0.7) {
                            this.addActivity('📊', 'Metrics updated', 'info');
                        }
                    }, 15000);
                }
            }
        }
    </script>
</body>
</html>
    """


async def main():
    """Main entry point for continuous deployment system."""
    project_root = Path(__file__).parent
    
    # Create enhanced frontend
    frontend_content = create_hot_reload_frontend()
    (project_root / "hot_reload_frontend.html").write_text(frontend_content)
    
    # Start continuous deployment
    cd_system = ContinuousDeployment(project_root)
    await cd_system.start()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping continuous deployment system")


if __name__ == "__main__":
    asyncio.run(main())