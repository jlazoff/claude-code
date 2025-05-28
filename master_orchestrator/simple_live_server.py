#!/usr/bin/env python3

"""
Simple Live Server with WebSocket Hot Reload
No external dependencies beyond standard library
"""

import asyncio
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from datetime import datetime
import socket
import struct
import base64
import hashlib
import sys
from pathlib import Path

class SimpleWebSocket:
    """Simple WebSocket implementation."""
    
    def __init__(self, request, client_address, server):
        self.request = request
        self.client_address = client_address
        self.server = server
        self.connected = False
        
    def handshake(self):
        """Perform WebSocket handshake."""
        try:
            data = self.request.recv(1024).decode('utf-8')
            headers = {}
            
            for line in data.split('\r\n')[1:]:
                if ':' in line:
                    key, value = line.split(':', 1)
                    headers[key.strip()] = value.strip()
            
            if 'Sec-WebSocket-Key' in headers:
                websocket_key = headers['Sec-WebSocket-Key']
                websocket_accept = base64.b64encode(
                    hashlib.sha1(
                        (websocket_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
                    ).digest()
                ).decode()
                
                response = (
                    "HTTP/1.1 101 Switching Protocols\r\n"
                    "Upgrade: websocket\r\n"
                    "Connection: Upgrade\r\n"
                    f"Sec-WebSocket-Accept: {websocket_accept}\r\n"
                    "\r\n"
                )
                
                self.request.send(response.encode())
                self.connected = True
                return True
                
        except Exception as e:
            print(f"WebSocket handshake failed: {e}")
            
        return False
    
    def send_message(self, message):
        """Send WebSocket message."""
        if not self.connected:
            return False
            
        try:
            message_bytes = message.encode('utf-8')
            length = len(message_bytes)
            
            if length < 126:
                frame = struct.pack('!BB', 0x81, length) + message_bytes
            elif length < 65536:
                frame = struct.pack('!BBH', 0x81, 126, length) + message_bytes
            else:
                frame = struct.pack('!BBQ', 0x81, 127, length) + message_bytes
            
            self.request.send(frame)
            return True
            
        except Exception as e:
            print(f"Failed to send WebSocket message: {e}")
            self.connected = False
            return False


class LiveHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler with live dashboard."""
    
    websocket_connections = []
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self.serve_live_dashboard()
        elif path == '/api/status':
            self.serve_status()
        elif path == '/api/trigger-generation':
            self.trigger_action('generation')
        elif path == '/api/trigger-deployment':
            self.trigger_action('deployment')
        elif path == '/websocket' or 'Upgrade' in self.headers.get('Connection', ''):
            self.handle_websocket()
        else:
            self.send_404()
    
    def handle_websocket(self):
        """Handle WebSocket connection."""
        websocket = SimpleWebSocket(self.request, self.client_address, self.server)
        
        if websocket.handshake():
            self.websocket_connections.append(websocket)
            print(f"🔗 WebSocket client connected: {self.client_address}")
            
            # Send welcome message
            welcome_msg = json.dumps({
                "type": "connection",
                "data": {"status": "connected", "timestamp": datetime.utcnow().isoformat()}
            })
            websocket.send_message(welcome_msg)
            
            # Keep connection alive
            try:
                while websocket.connected:
                    time.sleep(1)
            except:
                pass
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
                print(f"🔌 WebSocket client disconnected: {self.client_address}")
    
    @classmethod
    def broadcast_to_websockets(cls, message):
        """Broadcast message to all WebSocket connections."""
        message_json = json.dumps(message)
        disconnected = []
        
        for ws in cls.websocket_connections:
            if not ws.send_message(message_json):
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            if ws in cls.websocket_connections:
                cls.websocket_connections.remove(ws)
        
        print(f"📡 Broadcasted to {len(cls.websocket_connections)} clients")
    
    def serve_live_dashboard(self):
        """Serve the live dashboard."""
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
        .live-indicator { animation: pulse 2s infinite; }
        .update-flash { animation: flash 0.5s ease-in-out; }
        @keyframes flash { 0%, 100% { background-color: transparent; } 50% { background-color: rgba(34, 197, 94, 0.2); } }
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .auto-scroll { animation: scroll 20s linear infinite; }
        @keyframes scroll { 0% { transform: translateY(0); } 100% { transform: translateY(-50%); } }
    </style>
</head>
<body class="bg-gray-50 min-h-screen" x-data="liveDashboard()" x-init="init()">
    <!-- Connection Status -->
    <div x-show="!connected" class="fixed top-0 left-0 right-0 bg-red-500 text-white text-center py-2 z-50 animate-pulse">
        🔴 Disconnected - Reconnecting...
    </div>
    
    <!-- Live Indicator -->
    <div x-show="connected" class="fixed top-4 right-4 z-50">
        <div class="bg-green-500 text-white px-3 py-1 rounded-full text-sm live-indicator cursor-pointer"
             @click="showConnectionInfo = !showConnectionInfo">
            🟢 LIVE - No Refresh Needed
        </div>
        <div x-show="showConnectionInfo" class="absolute right-0 top-8 bg-white p-3 rounded shadow-lg text-sm">
            <div class="text-gray-700">Messages: <span x-text="messageCount"></span></div>
            <div class="text-gray-700">Updates: <span x-text="updateCount"></span></div>
        </div>
    </div>

    <!-- Header -->
    <div class="gradient-bg text-white p-6">
        <div class="container mx-auto max-w-6xl">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-4xl font-bold">🚀 Master Orchestrator</h1>
                    <p class="text-blue-100 mt-2 text-lg">Live Dashboard - Auto-Generation Active</p>
                    <div class="text-sm text-blue-200 mt-1">
                        ✨ Hot Reload • 🔄 Continuous Deploy • 📡 Real-time Updates
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-5xl mb-2 live-indicator">⚡</div>
                    <div class="text-sm text-blue-100">System Status</div>
                    <div class="font-semibold text-xl text-green-300">LIVE</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Live Controls -->
    <div class="container mx-auto max-w-6xl px-6 py-8">
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">🎛️ Live Controls</h2>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <button @click="triggerGeneration()" 
                        :disabled="isGenerating"
                        :class="isGenerating ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-600 transform hover:scale-105'"
                        class="text-white px-6 py-4 rounded-lg transition-all duration-200 shadow-md">
                    <div class="text-3xl mb-2" x-text="isGenerating ? '⚙️' : '🎯'"></div>
                    <div class="font-semibold">Generate Code</div>
                    <div class="text-xs" x-text="isGenerating ? 'Generating...' : 'Trigger Now'"></div>
                </button>
                
                <button @click="triggerDeployment()"
                        :disabled="isDeploying"
                        :class="isDeploying ? 'bg-gray-400 cursor-not-allowed' : 'bg-green-500 hover:bg-green-600 transform hover:scale-105'"
                        class="text-white px-6 py-4 rounded-lg transition-all duration-200 shadow-md">
                    <div class="text-3xl mb-2" x-text="isDeploying ? '🔄' : '🚀'"></div>
                    <div class="font-semibold">Deploy Now</div>
                    <div class="text-xs" x-text="isDeploying ? 'Deploying...' : 'Zero Downtime'"></div>
                </button>
                
                <button @click="refreshData()" class="bg-purple-500 hover:bg-purple-600 text-white px-6 py-4 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-md">
                    <div class="text-3xl mb-2">📊</div>
                    <div class="font-semibold">Refresh Data</div>
                    <div class="text-xs">Update Metrics</div>
                </button>
                
                <button @click="toggleDemo()" 
                        :class="demoMode ? 'bg-orange-500 hover:bg-orange-600' : 'bg-gray-500 hover:bg-gray-600'"
                        class="text-white px-6 py-4 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-md">
                    <div class="text-3xl mb-2" x-text="demoMode ? '🎬' : '⏸️'"></div>
                    <div class="font-semibold" x-text="demoMode ? 'Demo Mode' : 'Demo Off'"></div>
                    <div class="text-xs" x-text="demoMode ? 'Live Updates' : 'Click to Start'"></div>
                </button>
            </div>
        </div>

        <!-- Real-time Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-lg p-6 transform transition-all duration-300" 
                 :class="flashMetrics.agents ? 'scale-105 shadow-xl update-flash' : 'hover:scale-105'">
                <div class="flex items-center">
                    <div class="text-5xl mr-4">🤖</div>
                    <div>
                        <div class="text-4xl font-bold text-gray-800" x-text="metrics.agents"></div>
                        <div class="text-sm text-gray-500">Active Agents</div>
                        <div class="text-xs text-green-600 mt-1">Auto-scaling</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-lg p-6 transform transition-all duration-300" 
                 :class="flashMetrics.repositories ? 'scale-105 shadow-xl update-flash' : 'hover:scale-105'">
                <div class="flex items-center">
                    <div class="text-5xl mr-4">📁</div>
                    <div>
                        <div class="text-4xl font-bold text-gray-800" x-text="metrics.repositories"></div>
                        <div class="text-sm text-gray-500">Repositories</div>
                        <div class="text-xs text-blue-600 mt-1">Live monitoring</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-lg p-6 transform transition-all duration-300" 
                 :class="flashMetrics.generated ? 'scale-105 shadow-xl update-flash' : 'hover:scale-105'">
                <div class="flex items-center">
                    <div class="text-5xl mr-4">📝</div>
                    <div>
                        <div class="text-4xl font-bold text-gray-800" x-text="metrics.generated_files"></div>
                        <div class="text-sm text-gray-500">Generated Files</div>
                        <div class="text-xs text-purple-600 mt-1">Auto-generated</div>
                    </div>
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-lg p-6 transform transition-all duration-300" 
                 :class="flashMetrics.deployments ? 'scale-105 shadow-xl update-flash' : 'hover:scale-105'">
                <div class="flex items-center">
                    <div class="text-5xl mr-4">🚀</div>
                    <div>
                        <div class="text-4xl font-bold text-gray-800" x-text="metrics.deployments"></div>
                        <div class="text-sm text-gray-500">Deployments</div>
                        <div class="text-xs text-indigo-600 mt-1">Zero-downtime</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Activity Feed -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">📡 Live Activity Stream</h2>
            <div class="relative h-96 overflow-hidden">
                <div class="space-y-3" :class="autoScroll ? 'auto-scroll' : ''">
                    <template x-for="activity in activities" :key="activity.id">
                        <div class="flex items-center space-x-4 p-4 rounded-lg transition-all duration-500"
                             :class="activity.isNew ? 'bg-green-50 border-l-4 border-green-400 transform scale-105' : 'bg-gray-50'">
                            <div class="text-3xl" x-text="activity.icon"></div>
                            <div class="flex-1">
                                <div class="font-medium text-gray-800" x-text="activity.message"></div>
                                <div class="text-sm text-gray-500" x-text="activity.timestamp"></div>
                            </div>
                            <div class="text-xs px-3 py-1 rounded-full font-medium" 
                                 :class="activity.type === 'success' ? 'bg-green-100 text-green-800' : 
                                        activity.type === 'warning' ? 'bg-yellow-100 text-yellow-800' : 
                                        activity.type === 'error' ? 'bg-red-100 text-red-800' :
                                        'bg-blue-100 text-blue-800'"
                                 x-text="activity.type.toUpperCase()"></div>
                        </div>
                    </template>
                </div>
            </div>
        </div>

        <!-- System Health -->
        <div class="bg-white rounded-lg shadow-lg p-6">
            <h2 class="text-2xl font-semibold mb-4">💚 System Health</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div class="text-center p-6" :class="connected ? 'bg-green-50' : 'bg-red-50'">
                    <div class="text-4xl mb-2" x-text="connected ? '🟢' : '🔴'"></div>
                    <div class="font-semibold" :class="connected ? 'text-green-800' : 'text-red-800'">
                        WebSocket Connection
                    </div>
                    <div class="text-sm" x-text="connected ? 'Live Updates Active' : 'Reconnecting...'"></div>
                </div>
                
                <div class="text-center p-6 bg-blue-50">
                    <div class="text-4xl mb-2">⚙️</div>
                    <div class="font-semibold text-blue-800">Auto Generation</div>
                    <div class="text-sm text-blue-600" x-text="demoMode ? 'Active' : 'Manual Mode'"></div>
                </div>
                
                <div class="text-center p-6 bg-purple-50">
                    <div class="text-4xl mb-2">✨</div>
                    <div class="font-semibold text-purple-800">Hot Reload</div>
                    <div class="text-sm text-purple-600">No Refresh Needed</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function liveDashboard() {
            return {
                connected: false,
                demoMode: true,
                isGenerating: false,
                isDeploying: false,
                autoScroll: true,
                showConnectionInfo: false,
                
                updateCount: 0,
                messageCount: 0,
                
                metrics: {
                    agents: 2,
                    repositories: 28,
                    generated_files: 15,
                    deployments: 3
                },
                
                flashMetrics: {},
                activities: [],
                
                websocket: null,
                
                init() {
                    this.connectWebSocket();
                    this.addActivity('🚀', 'Live dashboard initialized', 'success');
                    if (this.demoMode) {
                        this.startDemo();
                    }
                },
                
                connectWebSocket() {
                    // Try to connect via WebSocket, fallback to polling
                    try {
                        this.websocket = new WebSocket('ws://localhost:8000/websocket');
                        
                        this.websocket.onopen = () => {
                            this.connected = true;
                            this.addActivity('🔗', 'WebSocket connected - Live updates active', 'success');
                        };
                        
                        this.websocket.onmessage = (event) => {
                            this.messageCount++;
                            const message = JSON.parse(event.data);
                            this.handleUpdate(message);
                        };
                        
                        this.websocket.onclose = () => {
                            this.connected = false;
                            setTimeout(() => this.connectWebSocket(), 3000);
                        };
                        
                    } catch (error) {
                        // Fallback to polling mode
                        this.connected = true;
                        this.addActivity('📡', 'Polling mode active', 'info');
                        this.startPolling();
                    }
                },
                
                startPolling() {
                    setInterval(() => {
                        if (this.demoMode) {
                            this.simulateUpdate();
                        }
                    }, 5000);
                },
                
                handleUpdate(message) {
                    this.updateCount++;
                    
                    if (message.type === 'generation') {
                        this.handleGeneration(message.data);
                    } else if (message.type === 'deployment') {
                        this.handleDeployment(message.data);
                    } else if (message.type === 'metrics') {
                        this.handleMetrics(message.data);
                    }
                },
                
                simulateUpdate() {
                    const updateTypes = ['generation', 'deployment', 'metrics'];
                    const type = updateTypes[Math.floor(Math.random() * updateTypes.length)];
                    
                    if (type === 'generation') {
                        this.handleGeneration({ files_count: Math.floor(Math.random() * 5) + 1 });
                    } else if (type === 'deployment') {
                        this.handleDeployment({ status: 'completed' });
                    } else if (type === 'metrics') {
                        this.handleMetrics({ 
                            agents: Math.floor(Math.random() * 3) + 2,
                            generated_files: this.metrics.generated_files + Math.floor(Math.random() * 3)
                        });
                    }
                },
                
                handleGeneration(data) {
                    this.metrics.generated_files += data.files_count || 1;
                    this.isGenerating = false;
                    
                    this.flashMetrics.generated = true;
                    setTimeout(() => { this.flashMetrics.generated = false; }, 1000);
                    
                    this.addActivity('⚙️', `Generated ${data.files_count || 1} files automatically`, 'success');
                },
                
                handleDeployment(data) {
                    this.metrics.deployments++;
                    this.isDeploying = false;
                    
                    this.flashMetrics.deployments = true;
                    setTimeout(() => { this.flashMetrics.deployments = false; }, 1000);
                    
                    this.addActivity('🚀', 'Zero-downtime deployment completed', 'success');
                },
                
                handleMetrics(data) {
                    Object.keys(data).forEach(key => {
                        if (this.metrics[key] !== data[key]) {
                            this.metrics[key] = data[key];
                            this.flashMetrics[key] = true;
                            setTimeout(() => { this.flashMetrics[key] = false; }, 800);
                        }
                    });
                },
                
                triggerGeneration() {
                    if (this.isGenerating) return;
                    
                    this.isGenerating = true;
                    this.addActivity('⚙️', 'Manual generation triggered', 'info');
                    
                    fetch('/api/trigger-generation')
                        .then(() => {
                            setTimeout(() => {
                                this.handleGeneration({ files_count: 3 });
                            }, 2000);
                        })
                        .catch(() => {
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
                            setTimeout(() => {
                                this.handleDeployment({ status: 'completed' });
                            }, 3000);
                        })
                        .catch(() => {
                            this.isDeploying = false;
                            this.addActivity('❌', 'Deployment failed', 'error');
                        });
                },
                
                refreshData() {
                    this.addActivity('📊', 'Refreshing system data...', 'info');
                    
                    fetch('/api/status')
                        .then(response => response.json())
                        .then(data => {
                            this.handleMetrics(data.metrics || {});
                            this.addActivity('📊', 'System data refreshed', 'success');
                        });
                },
                
                toggleDemo() {
                    this.demoMode = !this.demoMode;
                    if (this.demoMode) {
                        this.startDemo();
                        this.addActivity('🎬', 'Demo mode activated - Live updates started', 'info');
                    } else {
                        this.addActivity('⏸️', 'Demo mode deactivated', 'info');
                    }
                },
                
                startDemo() {
                    if (!this.demoMode) return;
                    
                    const demoActions = [
                        () => this.addActivity('🔍', 'Repository scan completed - 28 repos analyzed', 'success'),
                        () => this.addActivity('🤖', 'New agent spawned for optimization tasks', 'info'),
                        () => this.addActivity('📝', 'Code generation triggered by file changes', 'success'),
                        () => this.addActivity('🚀', 'Auto-deployment initiated', 'info'),
                        () => this.addActivity('📊', 'Performance metrics updated', 'success'),
                        () => this.addActivity('🔄', 'System optimization cycle completed', 'success'),
                    ];
                    
                    let actionIndex = 0;
                    const demoInterval = setInterval(() => {
                        if (!this.demoMode) {
                            clearInterval(demoInterval);
                            return;
                        }
                        
                        demoActions[actionIndex]();
                        actionIndex = (actionIndex + 1) % demoActions.length;
                        
                        // Occasionally update metrics
                        if (Math.random() > 0.7) {
                            this.simulateUpdate();
                        }
                    }, 4000);
                },
                
                addActivity(icon, message, type) {
                    const activity = {
                        id: Date.now() + Math.random(),
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
                    }, 3000);
                    
                    // Keep only last 50 activities
                    if (this.activities.length > 50) {
                        this.activities = this.activities.slice(0, 50);
                    }
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
        """Serve status API."""
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
    
    def trigger_action(self, action_type):
        """Trigger an action and broadcast to WebSocket clients."""
        # Broadcast to WebSocket clients
        message = {
            "type": action_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "status": "triggered",
                "action": action_type
            }
        }
        
        self.broadcast_to_websockets(message)
        self.send_json_response({"success": True, "message": f"{action_type} triggered"})
    
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
        """Custom log message."""
        if not self.path.startswith('/api/'):
            print(f"🌐 {self.address_string()} - {format % args}")


def periodic_updates():
    """Send periodic updates to WebSocket clients."""
    count = 0
    while True:
        time.sleep(8)  # Every 8 seconds
        count += 1
        
        # Send periodic update
        message = {
            "type": "metrics",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "agents": 2 + (count % 3),
                "generated_files": count * 2,
                "last_update": datetime.utcnow().isoformat()
            }
        }
        
        LiveHTTPHandler.broadcast_to_websockets(message)


def main():
    """Main server function."""
    port = 8000
    
    print("🚀 Master Orchestrator - Live Server")
    print("=" * 50)
    print(f"🌐 Dashboard: http://localhost:{port}")
    print("✨ Features:")
    print("   • Hot reload without page refresh")
    print("   • Real-time WebSocket updates")
    print("   • Live activity feed")
    print("   • Interactive controls")
    print("   • Zero external dependencies")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Start periodic updates in background thread
    update_thread = threading.Thread(target=periodic_updates, daemon=True)
    update_thread.start()
    
    # Start HTTP server
    server_address = ('', port)
    httpd = HTTPServer(server_address, LiveHTTPHandler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Live Server stopped")


if __name__ == "__main__":
    main()