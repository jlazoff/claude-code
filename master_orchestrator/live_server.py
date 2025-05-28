#!/usr/bin/env python3

"""
Live Master Orchestrator Server
Integrates continuous deployment with hot-reloading frontend
"""

import asyncio
import json
import time
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import websockets
from typing import Set, Dict, Any
from datetime import datetime
import subprocess

class LiveWebSocketServer:
    """WebSocket server for real-time updates."""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        
    async def start(self):
        """Start WebSocket server."""
        print(f"🔗 Starting WebSocket server on port {self.port}")
        
        async def handler(websocket, path):
            self.clients.add(websocket)
            print(f"🔗 Client connected: {websocket.remote_address}")
            
            try:
                async for message in websocket:
                    # Handle client messages if needed
                    pass
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.clients.remove(websocket)
                print(f"🔌 Client disconnected")
        
        self.server = await websockets.serve(handler, "localhost", self.port)
        return self.server
    
    async def broadcast(self, message_type: str, data: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        
        message = {
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message_json)
            except:
                disconnected.add(client)
        
        self.clients -= disconnected
        print(f"📡 Broadcasted {message_type} to {len(self.clients)} clients")


class LiveHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler with live dashboard."""
    
    def __init__(self, websocket_server, *args, **kwargs):
        self.websocket_server = websocket_server
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.serve_live_dashboard()
        elif path == '/api/status':
            self.serve_status()
        elif path == '/api/trigger-generation':
            self.trigger_generation()
        elif path == '/api/trigger-deployment':
            self.trigger_deployment()
        else:
            self.send_404()
    
    def serve_live_dashboard(self):
        """Serve the live dashboard with WebSocket integration."""
        dashboard_html = """
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
        .typing { animation: typing 2s infinite; }
        @keyframes typing { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0.3; } }
    </style>
</head>
<body class="bg-gray-50 min-h-screen" x-data="liveOrchestrator()" x-init="init()">
    <!-- Connection Status -->
    <div x-show="!connected" class="fixed top-0 left-0 right-0 bg-red-500 text-white text-center py-2 z-50">
        🔴 Disconnected - Attempting to reconnect...
    </div>
    
    <!-- Live Updates Indicator -->
    <div x-show="connected" class="fixed top-4 right-4 z-50">
        <div class="bg-green-500 text-white px-3 py-1 rounded-full text-sm live-indicator">
            🟢 Live - No Refresh Needed
        </div>
    </div>

    <!-- Header -->
    <div class="gradient-bg text-white p-6">
        <div class="container mx-auto max-w-6xl">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-4xl font-bold">🚀 Master Orchestrator</h1>
                    <p class="text-blue-100 mt-2 text-lg">Continuous Auto-Generation & Hot Reload</p>
                    <div class="text-sm text-blue-200 mt-1">
                        <span x-text="'Updates: ' + updateCount"></span> •
                        <span x-text="'Generations: ' + generationCount"></span> •
                        <span x-text="'Deployments: ' + deploymentCount"></span>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-4xl mb-2" :class="isGenerating ? 'typing' : 'live-indicator'">
                        <span x-text="isGenerating ? '⚙️' : '⚡'"></span>
                    </div>
                    <div class="text-sm text-blue-100">System Status</div>
                    <div class="font-semibold text-xl text-green-300">LIVE</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Real-time Control Panel -->
    <div class="container mx-auto max-w-6xl px-6 py-8">
        <!-- Live Controls -->
        <div class="bg-white rounded-lg shadow p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">🎛️ Live Controls</h2>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <button @click="triggerGeneration()" 
                        :disabled="isGenerating"
                        :class="isGenerating ? 'bg-gray-400' : 'bg-blue-500 hover:bg-blue-600'"
                        class="text-white px-4 py-3 rounded-lg transition-colors">
                    <div class="text-2xl mb-1">⚙️</div>
                    <div class="font-semibold">Generate Code</div>
                    <div class="text-xs" x-text="isGenerating ? 'Generating...' : 'Trigger Now'"></div>
                </button>
                
                <button @click="triggerDeployment()"
                        :disabled="isDeploying"
                        :class="isDeploying ? 'bg-gray-400' : 'bg-green-500 hover:bg-green-600'"
                        class="text-white px-4 py-3 rounded-lg transition-colors">
                    <div class="text-2xl mb-1">🚀</div>
                    <div class="font-semibold">Deploy Now</div>
                    <div class="text-xs" x-text="isDeploying ? 'Deploying...' : 'Zero Downtime'"></div>
                </button>
                
                <button @click="refreshMetrics()" class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-3 rounded-lg transition-colors">
                    <div class="text-2xl mb-1">📊</div>
                    <div class="font-semibold">Refresh Data</div>
                    <div class="text-xs">Update Metrics</div>
                </button>
                
                <button @click="toggleAutoMode()" 
                        :class="autoMode ? 'bg-orange-500 hover:bg-orange-600' : 'bg-gray-500 hover:bg-gray-600'"
                        class="text-white px-4 py-3 rounded-lg transition-colors">
                    <div class="text-2xl mb-1" x-text="autoMode ? '🔄' : '⏸️'"></div>
                    <div class="font-semibold" x-text="autoMode ? 'Auto Mode' : 'Manual Mode'"></div>
                    <div class="text-xs" x-text="autoMode ? 'Click to Pause' : 'Click to Resume'"></div>
                </button>
            </div>
        </div>

        <!-- Live Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow p-6" :class="flashMetrics.agents ? 'update-flash' : ''">
                <div class="flex items-center">
                    <div class="text-4xl mr-4">🤖</div>
                    <div>
                        <div class="text-3xl font-bold text-gray-800" x-text="metrics.agents"></div>
                        <div class="text-sm text-gray-500">Active Agents</div>
                        <div class="text-xs text-green-600 mt-1">Auto-scaling</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow p-6" :class="flashMetrics.repositories ? 'update-flash' : ''">
                <div class="flex items-center">
                    <div class="text-4xl mr-4">📁</div>
                    <div>
                        <div class="text-3xl font-bold text-gray-800" x-text="metrics.repositories"></div>
                        <div class="text-sm text-gray-500">Repositories</div>
                        <div class="text-xs text-blue-600 mt-1">Live monitoring</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow p-6" :class="flashMetrics.generated ? 'update-flash' : ''">
                <div class="flex items-center">
                    <div class="text-4xl mr-4">📝</div>
                    <div>
                        <div class="text-3xl font-bold text-gray-800" x-text="metrics.generated_files"></div>
                        <div class="text-sm text-gray-500">Generated Files</div>
                        <div class="text-xs text-purple-600 mt-1">Auto-generated</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow p-6" :class="flashMetrics.deployments ? 'update-flash' : ''">
                <div class="flex items-center">
                    <div class="text-4xl mr-4">🚀</div>
                    <div>
                        <div class="text-3xl font-bold text-gray-800" x-text="metrics.deployments"></div>
                        <div class="text-sm text-gray-500">Deployments</div>
                        <div class="text-xs text-indigo-600 mt-1">Zero-downtime</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Activity Stream -->
        <div class="bg-white rounded-lg shadow p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">📡 Live Activity Stream</h2>
            <div class="space-y-2 max-h-96 overflow-y-auto">
                <template x-for="activity in activities.slice(0, 20)" :key="activity.id">
                    <div class="flex items-center space-x-3 p-3 rounded transition-all duration-300"
                         :class="activity.isNew ? 'bg-green-50 border border-green-200' : 'bg-gray-50'">
                        <div class="text-2xl" x-text="activity.icon"></div>
                        <div class="flex-1">
                            <div class="font-medium" x-text="activity.message"></div>
                            <div class="text-sm text-gray-500" x-text="activity.timestamp"></div>
                        </div>
                        <div class="text-xs px-2 py-1 rounded-full" 
                             :class="activity.type === 'success' ? 'bg-green-100 text-green-800' : 
                                    activity.type === 'warning' ? 'bg-yellow-100 text-yellow-800' : 
                                    activity.type === 'error' ? 'bg-red-100 text-red-800' :
                                    'bg-blue-100 text-blue-800'"
                             x-text="activity.type"></div>
                    </div>
                </template>
            </div>
        </div>

        <!-- System Health -->
        <div class="bg-white rounded-lg shadow p-6">
            <h2 class="text-2xl font-semibold mb-4">💚 System Health</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-green-50 p-4 rounded">
                    <div class="font-semibold text-green-800">WebSocket Connection</div>
                    <div class="text-sm" :class="connected ? 'text-green-600' : 'text-red-600'">
                        <span x-text="connected ? '✅ Connected' : '🔴 Disconnected'"></span>
                    </div>
                    <div class="text-xs text-gray-600 mt-1" x-text="'Messages: ' + messageCount"></div>
                </div>
                
                <div class="bg-blue-50 p-4 rounded">
                    <div class="font-semibold text-blue-800">Auto Generation</div>
                    <div class="text-sm" :class="autoMode ? 'text-blue-600' : 'text-gray-600'">
                        <span x-text="autoMode ? '✅ Active' : '⏸️ Paused'"></span>
                    </div>
                    <div class="text-xs text-gray-600 mt-1" x-text="'Last: ' + lastGeneration"></div>
                </div>
                
                <div class="bg-purple-50 p-4 rounded">
                    <div class="font-semibold text-purple-800">Hot Reload</div>
                    <div class="text-sm text-purple-600">✅ No Refresh Needed</div>
                    <div class="text-xs text-gray-600 mt-1" x-text="'Updates: ' + updateCount"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function liveOrchestrator() {
            return {
                connected: false,
                autoMode: true,
                isGenerating: false,
                isDeploying: false,
                
                updateCount: 0,
                generationCount: 0,
                deploymentCount: 0,
                messageCount: 0,
                
                metrics: {
                    agents: 2,
                    repositories: 28,
                    generated_files: 0,
                    deployments: 0
                },
                
                flashMetrics: {},
                activities: [],
                
                lastGeneration: 'Never',
                websocket: null,
                
                init() {
                    this.connectWebSocket();
                    this.addActivity('🚀', 'Live dashboard initialized', 'success');
                    this.startHeartbeat();
                },
                
                connectWebSocket() {
                    try {
                        this.websocket = new WebSocket('ws://localhost:8001');
                        
                        this.websocket.onopen = () => {
                            this.connected = true;
                            this.addActivity('🔗', 'WebSocket connected - Live updates active', 'success');
                        };
                        
                        this.websocket.onmessage = (event) => {
                            this.messageCount++;
                            const message = JSON.parse(event.data);
                            this.handleLiveUpdate(message);
                        };
                        
                        this.websocket.onclose = () => {
                            this.connected = false;
                            this.addActivity('🔴', 'Connection lost - Reconnecting...', 'warning');
                            setTimeout(() => this.connectWebSocket(), 2000);
                        };
                        
                    } catch (error) {
                        console.error('WebSocket connection failed:', error);
                        setTimeout(() => this.connectWebSocket(), 5000);
                    }
                },
                
                handleLiveUpdate(message) {
                    this.updateCount++;
                    
                    if (message.type === 'generation') {
                        this.handleGeneration(message.data);
                    } else if (message.type === 'deployment') {
                        this.handleDeployment(message.data);
                    } else if (message.type === 'metrics') {
                        this.handleMetrics(message.data);
                    } else if (message.type === 'frontend_update') {
                        this.handleFrontendUpdate(message.data);
                    }
                },
                
                handleGeneration(data) {
                    this.generationCount++;
                    this.metrics.generated_files += data.files_count || 1;
                    this.lastGeneration = new Date().toLocaleTimeString();
                    this.isGenerating = false;
                    
                    this.flashMetrics.generated = true;
                    setTimeout(() => { this.flashMetrics.generated = false; }, 1000);
                    
                    this.addActivity('⚙️', `Generated ${data.files_count || 1} files`, 'success');
                },
                
                handleDeployment(data) {
                    this.deploymentCount++;
                    this.metrics.deployments = this.deploymentCount;
                    this.isDeploying = false;
                    
                    this.flashMetrics.deployments = true;
                    setTimeout(() => { this.flashMetrics.deployments = false; }, 1000);
                    
                    this.addActivity('🚀', 'Zero-downtime deployment completed', 'success');
                },
                
                handleMetrics(data) {
                    const oldMetrics = {...this.metrics};
                    this.metrics = {...this.metrics, ...data};
                    
                    // Flash changed metrics
                    Object.keys(data).forEach(key => {
                        if (oldMetrics[key] !== data[key]) {
                            this.flashMetrics[key] = true;
                            setTimeout(() => { this.flashMetrics[key] = false; }, 1000);
                        }
                    });
                },
                
                handleFrontendUpdate(data) {
                    if (data.reload_type === 'hot') {
                        // Hot reload CSS
                        const links = document.querySelectorAll('link[rel="stylesheet"]');
                        links.forEach(link => {
                            const href = link.href.split('?')[0];
                            link.href = href + '?v=' + Date.now();
                        });
                        this.addActivity('🎨', 'Styles updated (no refresh)', 'info');
                    }
                },
                
                triggerGeneration() {
                    if (this.isGenerating) return;
                    
                    this.isGenerating = true;
                    this.addActivity('⚙️', 'Manual generation triggered', 'info');
                    
                    fetch('/api/trigger-generation')
                        .then(() => {
                            // Will be handled by WebSocket update
                        })
                        .catch(error => {
                            this.isGenerating = false;
                            this.addActivity('❌', 'Generation failed', 'error');
                        });
                },
                
                triggerDeployment() {
                    if (this.isDeploying) return;
                    
                    this.isDeploying = true;
                    this.addActivity('🚀', 'Manual deployment triggered', 'info');
                    
                    fetch('/api/trigger-deployment')
                        .then(() => {
                            // Will be handled by WebSocket update
                        })
                        .catch(error => {
                            this.isDeploying = false;
                            this.addActivity('❌', 'Deployment failed', 'error');
                        });
                },
                
                refreshMetrics() {
                    this.addActivity('📊', 'Refreshing metrics...', 'info');
                    
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {
                            this.handleMetrics(data.metrics || {});
                            this.addActivity('📊', 'Metrics updated', 'success');
                        });
                },
                
                toggleAutoMode() {
                    this.autoMode = !this.autoMode;
                    this.addActivity(
                        this.autoMode ? '🔄' : '⏸️', 
                        `Auto mode ${this.autoMode ? 'enabled' : 'disabled'}`, 
                        'info'
                    );
                },
                
                addActivity(icon, message, type) {
                    const activity = {
                        id: Date.now(),
                        icon: icon,
                        message: message,
                        type: type,
                        timestamp: new Date().toLocaleTimeString(),
                        isNew: true
                    };
                    
                    this.activities.unshift(activity);
                    
                    // Remove new flag after animation
                    setTimeout(() => {
                        activity.isNew = false;
                    }, 2000);
                    
                    // Keep only last 100 activities
                    if (this.activities.length > 100) {
                        this.activities = this.activities.slice(0, 100);
                    }
                },
                
                startHeartbeat() {
                    setInterval(() => {
                        if (this.connected && this.websocket) {
                            this.websocket.send(JSON.stringify({type: 'heartbeat'}));
                        }
                    }, 30000);
                }
            }
        }
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(dashboard_html.encode())
    
    def serve_status(self):
        """Serve current status."""
        status = {
            "status": "online",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "agents": 2,
                "repositories": 28,
                "generated_files": 15,
                "deployments": 3
            }
        }
        self.send_json_response(status)
    
    def trigger_generation(self):
        """Trigger code generation."""
        # Simulate generation
        asyncio.create_task(self.websocket_server.broadcast("generation", {
            "files_count": 2,
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        self.send_json_response({"success": True, "message": "Generation triggered"})
    
    def trigger_deployment(self):
        """Trigger deployment."""
        # Simulate deployment
        asyncio.create_task(self.websocket_server.broadcast("deployment", {
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        self.send_json_response({"success": True, "message": "Deployment triggered"})
    
    def send_json_response(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_404(self):
        """Send 404 response."""
        self.send_response(404)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


class LiveServer:
    """Integrated live server with WebSocket and HTTP."""
    
    def __init__(self, http_port: int = 8000, ws_port: int = 8001):
        self.http_port = http_port
        self.ws_port = ws_port
        self.websocket_server = LiveWebSocketServer(ws_port)
        
    def create_handler(self):
        """Create HTTP handler with WebSocket server reference."""
        def handler(*args, **kwargs):
            return LiveHTTPHandler(self.websocket_server, *args, **kwargs)
        return handler
    
    async def start(self):
        """Start both HTTP and WebSocket servers."""
        print("🚀 Starting Live Master Orchestrator Server")
        print("=" * 50)
        
        # Start WebSocket server
        await self.websocket_server.start()
        
        # Start HTTP server in thread
        handler_class = self.create_handler()
        httpd = HTTPServer(('', self.http_port), handler_class)
        
        def run_http():
            print(f"🌐 HTTP server running on http://localhost:{self.http_port}")
            httpd.serve_forever()
        
        http_thread = threading.Thread(target=run_http, daemon=True)
        http_thread.start()
        
        print(f"🔗 WebSocket server running on ws://localhost:{self.ws_port}")
        print("✨ Live updates active - No browser refresh needed!")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50)
        
        # Start periodic updates
        asyncio.create_task(self.send_periodic_updates())
        
        # Keep running
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            print("\n🛑 Stopping Live Server")
            httpd.shutdown()
    
    async def send_periodic_updates(self):
        """Send periodic updates to demonstrate live functionality."""
        count = 0
        while True:
            await asyncio.sleep(10)  # Every 10 seconds
            count += 1
            
            # Send sample metrics update
            await self.websocket_server.broadcast("metrics", {
                "agents": 2 + (count % 3),
                "generated_files": count * 2,
                "last_update": datetime.utcnow().isoformat()
            })


async def main():
    """Main entry point."""
    live_server = LiveServer()
    await live_server.start()


if __name__ == "__main__":
    asyncio.run(main())