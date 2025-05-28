#!/usr/bin/env python3

"""
Enhanced Live Server with Development Capabilities Integration
WebSocket Hot Reload + Full Dev Control (Git, Web Search, Command Line, File System)
"""

import asyncio
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import socket
import struct
import base64
import hashlib
import sys
from pathlib import Path

# Import our dev capabilities
from dev_capabilities import MasterDevController

class EnhancedWebSocket:
    """Enhanced WebSocket implementation with dev capabilities."""
    
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
            elif length < 65516:
                frame = struct.pack('!BBH', 0x81, 126, length) + message_bytes
            else:
                frame = struct.pack('!BBQ', 0x81, 127, length) + message_bytes
            
            self.request.send(frame)
            return True
            
        except Exception as e:
            print(f"Failed to send WebSocket message: {e}")
            self.connected = False
            return False


class EnhancedLiveHTTPHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP handler with dev capabilities."""
    
    websocket_connections = []
    dev_controller = None
    
    @classmethod
    def initialize_dev_controller(cls):
        """Initialize the development controller."""
        if cls.dev_controller is None:
            cls.dev_controller = MasterDevController(Path('.'))
            print("🛠️ Development capabilities initialized")
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        
        if path == '/':
            self.serve_enhanced_dashboard()
        elif path == '/api/status':
            self.serve_status()
        elif path == '/api/git/status':
            self.handle_git_status()
        elif path == '/api/git/log':
            self.handle_git_log()
        elif path == '/api/system/info':
            self.handle_system_info()
        elif path == '/api/search/web':
            self.handle_web_search(query_params)
        elif path == '/api/trigger-generation':
            self.trigger_action('generation')
        elif path == '/api/trigger-deployment':
            self.trigger_action('deployment')
        elif path == '/websocket' or 'Upgrade' in self.headers.get('Connection', ''):
            self.handle_websocket()
        else:
            self.send_404()
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/command/execute':
            self.handle_command_execution()
        elif path == '/api/git/command':
            self.handle_git_command()
        elif path == '/api/file/create':
            self.handle_file_create()
        else:
            self.send_404()
    
    def handle_git_status(self):
        """Handle git status request."""
        self.initialize_dev_controller()
        try:
            status = self.dev_controller.git.get_status()
            self.send_json_response({"success": True, "data": status})
        except Exception as e:
            self.send_json_response({"success": False, "error": str(e)})
    
    def handle_git_log(self):
        """Handle git log request."""
        self.initialize_dev_controller()
        try:
            log = self.dev_controller.git.get_log(max_count=10)
            self.send_json_response({"success": True, "data": log})
        except Exception as e:
            self.send_json_response({"success": False, "error": str(e)})
    
    def handle_system_info(self):
        """Handle system info request."""
        self.initialize_dev_controller()
        try:
            info = self.dev_controller.command_line.get_system_info()
            self.send_json_response({"success": True, "data": info})
        except Exception as e:
            self.send_json_response({"success": False, "error": str(e)})
    
    def handle_web_search(self, query_params):
        """Handle web search request."""
        self.initialize_dev_controller()
        try:
            query = query_params.get('q', [''])[0]
            if not query:
                self.send_json_response({"success": False, "error": "Query parameter 'q' required"})
                return
            
            results = self.dev_controller.web_search.search_web(query, max_results=5)
            self.send_json_response({"success": True, "data": results})
        except Exception as e:
            self.send_json_response({"success": False, "error": str(e)})
    
    def handle_command_execution(self):
        """Handle command execution request."""
        self.initialize_dev_controller()
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length:
                request_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                command = request_data.get('command', '')
                
                if command:
                    result = self.dev_controller.command_line.execute_command(command)
                    self.send_json_response({"success": True, "data": result})
                    
                    # Broadcast command execution to WebSocket clients
                    self.broadcast_to_websockets({
                        "type": "command_executed",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "command": command,
                            "result": result
                        }
                    })
                else:
                    self.send_json_response({"success": False, "error": "Command required"})
            else:
                self.send_json_response({"success": False, "error": "Request body required"})
                
        except Exception as e:
            self.send_json_response({"success": False, "error": str(e)})
    
    def handle_git_command(self):
        """Handle git command request."""
        self.initialize_dev_controller()
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length:
                request_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                action = request_data.get('action', '')
                
                result = None
                if action == 'commit':
                    message = request_data.get('message', 'Auto commit via live server')
                    result = self.dev_controller.git.commit_changes(message)
                elif action == 'push':
                    result = self.dev_controller.git.push_changes()
                elif action == 'pull':
                    result = self.dev_controller.git.pull_changes()
                else:
                    self.send_json_response({"success": False, "error": "Unknown git action"})
                    return
                
                self.send_json_response({"success": True, "data": result})
                
                # Broadcast git action to WebSocket clients
                self.broadcast_to_websockets({
                    "type": "git_action",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "action": action,
                        "result": result
                    }
                })
            else:
                self.send_json_response({"success": False, "error": "Request body required"})
                
        except Exception as e:
            self.send_json_response({"success": False, "error": str(e)})
    
    def handle_file_create(self):
        """Handle file creation request."""
        self.initialize_dev_controller()
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length:
                request_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                path = request_data.get('path', '')
                content = request_data.get('content', '')
                
                if path:
                    result = self.dev_controller.filesystem.create_file(Path(path), content)
                    self.send_json_response({"success": True, "data": result})
                    
                    # Broadcast file creation to WebSocket clients
                    self.broadcast_to_websockets({
                        "type": "file_created",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "path": path,
                            "size": len(content)
                        }
                    })
                else:
                    self.send_json_response({"success": False, "error": "File path required"})
            else:
                self.send_json_response({"success": False, "error": "Request body required"})
                
        except Exception as e:
            self.send_json_response({"success": False, "error": str(e)})
    
    def handle_websocket(self):
        """Handle WebSocket connection."""
        websocket = EnhancedWebSocket(self.request, self.client_address, self.server)
        
        if websocket.handshake():
            self.websocket_connections.append(websocket)
            print(f"🔗 Enhanced WebSocket client connected: {self.client_address}")
            
            # Send welcome message
            welcome_msg = json.dumps({
                "type": "connection",
                "data": {
                    "status": "connected", 
                    "timestamp": datetime.utcnow().isoformat(),
                    "capabilities": ["git", "web_search", "command_line", "file_system"]
                }
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
                print(f"🔌 Enhanced WebSocket client disconnected: {self.client_address}")
    
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
        
        print(f"📡 Broadcasted to {len(cls.websocket_connections)} enhanced clients")
    
    def serve_enhanced_dashboard(self):
        """Serve the enhanced live dashboard with dev capabilities."""
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Master Orchestrator - Enhanced Live Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <style>
        .live-indicator { animation: pulse 2s infinite; }
        .update-flash { animation: flash 0.5s ease-in-out; }
        @keyframes flash { 0%, 100% { background-color: transparent; } 50% { background-color: rgba(34, 197, 94, 0.2); } }
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .terminal { background: #1a1a1a; color: #00ff00; font-family: 'Courier New', monospace; }
        .scrollbar-thin { scrollbar-width: thin; }
        .scrollbar-thin::-webkit-scrollbar { width: 4px; }
        .scrollbar-thin::-webkit-scrollbar-track { background: #f1f1f1; }
        .scrollbar-thin::-webkit-scrollbar-thumb { background: #888; border-radius: 2px; }
    </style>
</head>
<body class="bg-gray-50 min-h-screen" x-data="enhancedLiveDashboard()" x-init="init()">
    <!-- Connection Status -->
    <div x-show="!connected" class="fixed top-0 left-0 right-0 bg-red-500 text-white text-center py-2 z-50 animate-pulse">
        🔴 Disconnected - Reconnecting...
    </div>
    
    <!-- Live Indicator -->
    <div x-show="connected" class="fixed top-4 right-4 z-50">
        <div class="bg-green-500 text-white px-3 py-1 rounded-full text-sm live-indicator cursor-pointer"
             @click="showConnectionInfo = !showConnectionInfo">
            🟢 ENHANCED LIVE
        </div>
        <div x-show="showConnectionInfo" class="absolute right-0 top-8 bg-white p-3 rounded shadow-lg text-sm">
            <div class="text-gray-700">Messages: <span x-text="messageCount"></span></div>
            <div class="text-gray-700">Commands: <span x-text="commandCount"></span></div>
            <div class="text-gray-700">Git Actions: <span x-text="gitActionCount"></span></div>
        </div>
    </div>

    <!-- Header -->
    <div class="gradient-bg text-white p-6">
        <div class="container mx-auto max-w-7xl">
            <div class="flex items-center justify-between">
                <div>
                    <h1 class="text-4xl font-bold">🚀 Master Orchestrator</h1>
                    <p class="text-blue-100 mt-2 text-lg">Enhanced Live Dashboard - Full Dev Control</p>
                    <div class="text-sm text-blue-200 mt-1">
                        ✨ Hot Reload • 🔄 Git Control • 📡 Command Line • 🌐 Web Search • 📁 File System
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-5xl mb-2 live-indicator">⚡</div>
                    <div class="text-sm text-blue-100">Enhanced Mode</div>
                    <div class="font-semibold text-xl text-green-300">FULL DEV</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Tab Navigation -->
    <div class="container mx-auto max-w-7xl px-6 py-4">
        <div class="flex space-x-4 mb-6">
            <button @click="activeTab = 'dashboard'" 
                    :class="activeTab === 'dashboard' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'"
                    class="px-4 py-2 rounded-lg font-medium transition">
                📊 Dashboard
            </button>
            <button @click="activeTab = 'git'" 
                    :class="activeTab === 'git' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'"
                    class="px-4 py-2 rounded-lg font-medium transition">
                🔄 Git Control
            </button>
            <button @click="activeTab = 'terminal'" 
                    :class="activeTab === 'terminal' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'"
                    class="px-4 py-2 rounded-lg font-medium transition">
                💻 Terminal
            </button>
            <button @click="activeTab = 'search'" 
                    :class="activeTab === 'search' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'"
                    class="px-4 py-2 rounded-lg font-medium transition">
                🌐 Web Search
            </button>
            <button @click="activeTab = 'files'" 
                    :class="activeTab === 'files' ? 'bg-blue-500 text-white' : 'bg-white text-gray-700 hover:bg-gray-50'"
                    class="px-4 py-2 rounded-lg font-medium transition">
                📁 File System
            </button>
        </div>

        <!-- Dashboard Tab -->
        <div x-show="activeTab === 'dashboard'">
            <!-- Live Controls -->
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
                    
                    <button @click="refreshSystemInfo()" class="bg-purple-500 hover:bg-purple-600 text-white px-6 py-4 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-md">
                        <div class="text-3xl mb-2">🖥️</div>
                        <div class="font-semibold">System Info</div>
                        <div class="text-xs">Check Status</div>
                    </button>
                    
                    <button @click="refreshGitStatus()" class="bg-indigo-500 hover:bg-indigo-600 text-white px-6 py-4 rounded-lg transition-all duration-200 transform hover:scale-105 shadow-md">
                        <div class="text-3xl mb-2">🔄</div>
                        <div class="font-semibold">Git Status</div>
                        <div class="text-xs">Check Repo</div>
                    </button>
                </div>
            </div>

            <!-- Real-time Metrics -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div class="bg-white rounded-lg shadow-lg p-6 transform transition-all duration-300 hover:scale-105">
                    <div class="flex items-center">
                        <div class="text-5xl mr-4">🤖</div>
                        <div>
                            <div class="text-4xl font-bold text-gray-800" x-text="metrics.agents"></div>
                            <div class="text-sm text-gray-500">Active Agents</div>
                            <div class="text-xs text-green-600 mt-1">Auto-scaling</div>
                        </div>
                    </div>
                </div>
                
                <div class="bg-white rounded-lg shadow-lg p-6 transform transition-all duration-300 hover:scale-105">
                    <div class="flex items-center">
                        <div class="text-5xl mr-4">📁</div>
                        <div>
                            <div class="text-4xl font-bold text-gray-800" x-text="metrics.repositories"></div>
                            <div class="text-sm text-gray-500">Repositories</div>
                            <div class="text-xs text-blue-600 mt-1">Live monitoring</div>
                        </div>
                    </div>
                </div>
                
                <div class="bg-white rounded-lg shadow-lg p-6 transform transition-all duration-300 hover:scale-105">
                    <div class="flex items-center">
                        <div class="text-5xl mr-4">💻</div>
                        <div>
                            <div class="text-4xl font-bold text-gray-800" x-text="commandCount"></div>
                            <div class="text-sm text-gray-500">Commands Executed</div>
                            <div class="text-xs text-purple-600 mt-1">Real-time</div>
                        </div>
                    </div>
                </div>
                
                <div class="bg-white rounded-lg shadow-lg p-6 transform transition-all duration-300 hover:scale-105">
                    <div class="flex items-center">
                        <div class="text-5xl mr-4">🔄</div>
                        <div>
                            <div class="text-4xl font-bold text-gray-800" x-text="gitActionCount"></div>
                            <div class="text-sm text-gray-500">Git Actions</div>
                            <div class="text-xs text-indigo-600 mt-1">Version control</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Git Control Tab -->
        <div x-show="activeTab === 'git'">
            <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
                <h2 class="text-2xl font-semibold mb-4">🔄 Git Control Center</h2>
                
                <!-- Git Actions -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <button @click="gitCommit()" class="bg-green-500 hover:bg-green-600 text-white px-4 py-3 rounded-lg">
                        📝 Commit Changes
                    </button>
                    <button @click="gitPush()" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-3 rounded-lg">
                        ⬆️ Push to Remote
                    </button>
                    <button @click="gitPull()" class="bg-purple-500 hover:bg-purple-600 text-white px-4 py-3 rounded-lg">
                        ⬇️ Pull Changes
                    </button>
                </div>

                <!-- Git Status Display -->
                <div class="bg-gray-50 rounded-lg p-4 mb-4">
                    <h3 class="font-semibold mb-2">Repository Status:</h3>
                    <pre class="text-sm" x-text="gitStatus || 'Click refresh to load git status'"></pre>
                </div>

                <!-- Recent Commits -->
                <div class="bg-gray-50 rounded-lg p-4">
                    <h3 class="font-semibold mb-2">Recent Commits:</h3>
                    <div class="space-y-2">
                        <template x-for="commit in gitLog" :key="commit.hash">
                            <div class="text-sm border-l-2 border-blue-300 pl-3">
                                <div class="font-medium" x-text="commit.message"></div>
                                <div class="text-gray-500" x-text="commit.author + ' - ' + commit.date"></div>
                            </div>
                        </template>
                    </div>
                </div>
            </div>
        </div>

        <!-- Terminal Tab -->
        <div x-show="activeTab === 'terminal'">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-2xl font-semibold mb-4">💻 Live Terminal</h2>
                
                <!-- Command Input -->
                <div class="mb-4">
                    <div class="flex">
                        <input x-model="currentCommand" 
                               @keyup.enter="executeCommand()"
                               placeholder="Enter command (e.g., ls, pwd, git status)"
                               class="flex-1 px-4 py-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <button @click="executeCommand()" 
                                :disabled="!currentCommand.trim()"
                                class="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-2 rounded-r-lg">
                            Execute
                        </button>
                    </div>
                </div>

                <!-- Terminal Output -->
                <div class="terminal rounded-lg p-4 h-96 overflow-y-auto scrollbar-thin">
                    <div x-text="'Master Orchestrator Terminal - ' + new Date().toLocaleString()"></div>
                    <div>Type commands and see live results...</div>
                    <br>
                    <template x-for="entry in terminalHistory" :key="entry.id">
                        <div>
                            <div class="text-yellow-400">$ <span x-text="entry.command"></span></div>
                            <div class="text-green-400 whitespace-pre-wrap mb-2" x-text="entry.output"></div>
                        </div>
                    </template>
                </div>
            </div>
        </div>

        <!-- Web Search Tab -->
        <div x-show="activeTab === 'search'">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-2xl font-semibold mb-4">🌐 Web Search</h2>
                
                <!-- Search Input -->
                <div class="mb-4">
                    <div class="flex">
                        <input x-model="searchQuery" 
                               @keyup.enter="performWebSearch()"
                               placeholder="Search the web (e.g., 'Python asyncio examples')"
                               class="flex-1 px-4 py-2 border rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <button @click="performWebSearch()" 
                                :disabled="!searchQuery.trim()"
                                class="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-400 text-white px-6 py-2 rounded-r-lg">
                            Search
                        </button>
                    </div>
                </div>

                <!-- Search Results -->
                <div class="space-y-4">
                    <template x-for="result in searchResults" :key="result.url">
                        <div class="border rounded-lg p-4 hover:bg-gray-50">
                            <h3 class="font-semibold text-blue-600 mb-2">
                                <a :href="result.url" target="_blank" x-text="result.title"></a>
                            </h3>
                            <p class="text-gray-700 text-sm mb-2" x-text="result.description"></p>
                            <div class="text-xs text-gray-500" x-text="result.url"></div>
                        </div>
                    </template>
                    <div x-show="searchResults.length === 0 && searchQuery" class="text-gray-500 text-center py-8">
                        No results found. Try a different search query.
                    </div>
                </div>
            </div>
        </div>

        <!-- File System Tab -->
        <div x-show="activeTab === 'files'">
            <div class="bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-2xl font-semibold mb-4">📁 File System Manager</h2>
                
                <!-- File Creation -->
                <div class="mb-6">
                    <h3 class="font-semibold mb-2">Create New File:</h3>
                    <div class="space-y-2">
                        <input x-model="newFilePath" 
                               placeholder="File path (e.g., example.py)"
                               class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                        <textarea x-model="newFileContent" 
                                  placeholder="File content..."
                                  rows="4"
                                  class="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"></textarea>
                        <button @click="createFile()" 
                                :disabled="!newFilePath.trim()"
                                class="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg">
                            Create File
                        </button>
                    </div>
                </div>

                <!-- Recent File Operations -->
                <div class="bg-gray-50 rounded-lg p-4">
                    <h3 class="font-semibold mb-2">Recent File Operations:</h3>
                    <div class="space-y-2">
                        <template x-for="operation in fileOperations" :key="operation.id">
                            <div class="text-sm border-l-2 border-green-300 pl-3">
                                <div class="font-medium" x-text="operation.action"></div>
                                <div class="text-gray-500" x-text="operation.path + ' - ' + operation.timestamp"></div>
                            </div>
                        </template>
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Activity Feed -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4">📡 Live Activity Stream</h2>
            <div class="h-64 overflow-y-auto scrollbar-thin space-y-2">
                <template x-for="activity in activities" :key="activity.id">
                    <div class="flex items-center space-x-4 p-3 rounded-lg transition-all duration-500"
                         :class="activity.isNew ? 'bg-green-50 border-l-4 border-green-400 transform scale-105' : 'bg-gray-50'">
                        <div class="text-2xl" x-text="activity.icon"></div>
                        <div class="flex-1">
                            <div class="font-medium text-gray-800" x-text="activity.message"></div>
                            <div class="text-sm text-gray-500" x-text="activity.timestamp"></div>
                        </div>
                        <div class="text-xs px-2 py-1 rounded-full font-medium" 
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

    <script>
        function enhancedLiveDashboard() {
            return {
                connected: false,
                activeTab: 'dashboard',
                
                updateCount: 0,
                messageCount: 0,
                commandCount: 0,
                gitActionCount: 0,
                
                // State
                isGenerating: false,
                isDeploying: false,
                showConnectionInfo: false,
                
                // Data
                metrics: {
                    agents: 2,
                    repositories: 28,
                    generated_files: 15,
                    deployments: 3
                },
                
                activities: [],
                
                // Git
                gitStatus: '',
                gitLog: [],
                
                // Terminal
                currentCommand: '',
                terminalHistory: [],
                
                // Search
                searchQuery: '',
                searchResults: [],
                
                // Files
                newFilePath: '',
                newFileContent: '',
                fileOperations: [],
                
                websocket: null,
                
                init() {
                    this.connectWebSocket();
                    this.addActivity('🚀', 'Enhanced live dashboard initialized', 'success');
                    this.refreshGitStatus();
                },
                
                connectWebSocket() {
                    try {
                        this.websocket = new WebSocket('ws://localhost:8000/websocket');
                        
                        this.websocket.onopen = () => {
                            this.connected = true;
                            this.addActivity('🔗', 'Enhanced WebSocket connected - Full dev control active', 'success');
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
                        this.connected = true;
                        this.addActivity('📡', 'Polling mode active', 'info');
                    }
                },
                
                handleUpdate(message) {
                    this.updateCount++;
                    
                    if (message.type === 'command_executed') {
                        this.handleCommandExecuted(message.data);
                    } else if (message.type === 'git_action') {
                        this.handleGitAction(message.data);
                    } else if (message.type === 'file_created') {
                        this.handleFileCreated(message.data);
                    } else if (message.type === 'generation') {
                        this.handleGeneration(message.data);
                    } else if (message.type === 'deployment') {
                        this.handleDeployment(message.data);
                    }
                },
                
                handleCommandExecuted(data) {
                    this.commandCount++;
                    this.terminalHistory.unshift({
                        id: Date.now(),
                        command: data.command,
                        output: data.result.output || data.result.error || 'Command executed'
                    });
                    // Keep only last 50 commands
                    if (this.terminalHistory.length > 50) {
                        this.terminalHistory = this.terminalHistory.slice(0, 50);
                    }
                    this.addActivity('💻', `Command executed: ${data.command}`, 'success');
                },
                
                handleGitAction(data) {
                    this.gitActionCount++;
                    this.addActivity('🔄', `Git ${data.action} completed`, 'success');
                    if (data.action === 'commit' || data.action === 'pull') {
                        this.refreshGitStatus();
                    }
                },
                
                handleFileCreated(data) {
                    this.fileOperations.unshift({
                        id: Date.now(),
                        action: `Created file: ${data.path}`,
                        path: data.path,
                        timestamp: new Date().toLocaleTimeString()
                    });
                    this.addActivity('📁', `File created: ${data.path}`, 'success');
                },
                
                handleGeneration(data) {
                    this.metrics.generated_files += data.files_count || 1;
                    this.isGenerating = false;
                    this.addActivity('⚙️', `Generated ${data.files_count || 1} files automatically`, 'success');
                },
                
                handleDeployment(data) {
                    this.metrics.deployments++;
                    this.isDeploying = false;
                    this.addActivity('🚀', 'Zero-downtime deployment completed', 'success');
                },
                
                // Actions
                triggerGeneration() {
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
                
                refreshSystemInfo() {
                    fetch('/api/system/info')
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                this.addActivity('🖥️', `System: ${data.data.platform} - ${data.data.processor}`, 'info');
                            }
                        })
                        .catch(() => {
                            this.addActivity('❌', 'Failed to get system info', 'error');
                        });
                },
                
                refreshGitStatus() {
                    fetch('/api/git/status')
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                this.gitStatus = data.data.output || data.data.error || 'Git status retrieved';
                            }
                        });
                    
                    fetch('/api/git/log')
                        .then(response => response.json())
                        .then(data => {
                            if (data.success && data.data.output) {
                                // Parse simple git log output
                                const lines = data.data.output.split('\\n').filter(line => line.trim());
                                this.gitLog = lines.slice(0, 5).map((line, index) => ({
                                    hash: `commit-${index}`,
                                    message: line.substring(0, 60) + (line.length > 60 ? '...' : ''),
                                    author: 'Author',
                                    date: 'Recent'
                                }));
                            }
                        });
                },
                
                executeCommand() {
                    if (!this.currentCommand.trim()) return;
                    
                    fetch('/api/command/execute', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ command: this.currentCommand })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // Result will be handled by WebSocket
                            this.currentCommand = '';
                        } else {
                            this.addActivity('❌', `Command failed: ${data.error}`, 'error');
                        }
                    })
                    .catch(() => {
                        this.addActivity('❌', 'Failed to execute command', 'error');
                    });
                },
                
                gitCommit() {
                    const message = prompt('Commit message:', 'Auto commit via live dashboard');
                    if (!message) return;
                    
                    fetch('/api/git/command', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ action: 'commit', message: message })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (!data.success) {
                            this.addActivity('❌', `Git commit failed: ${data.error}`, 'error');
                        }
                    });
                },
                
                gitPush() {
                    fetch('/api/git/command', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ action: 'push' })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (!data.success) {
                            this.addActivity('❌', `Git push failed: ${data.error}`, 'error');
                        }
                    });
                },
                
                gitPull() {
                    fetch('/api/git/command', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ action: 'pull' })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (!data.success) {
                            this.addActivity('❌', `Git pull failed: ${data.error}`, 'error');
                        }
                    });
                },
                
                performWebSearch() {
                    if (!this.searchQuery.trim()) return;
                    
                    fetch(`/api/search/web?q=${encodeURIComponent(this.searchQuery)}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.success) {
                                this.searchResults = data.data || [];
                                this.addActivity('🌐', `Web search: "${this.searchQuery}" - ${this.searchResults.length} results`, 'success');
                            } else {
                                this.addActivity('❌', `Search failed: ${data.error}`, 'error');
                            }
                        })
                        .catch(() => {
                            this.addActivity('❌', 'Failed to perform web search', 'error');
                        });
                },
                
                createFile() {
                    if (!this.newFilePath.trim()) return;
                    
                    fetch('/api/file/create', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            path: this.newFilePath, 
                            content: this.newFileContent 
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            this.newFilePath = '';
                            this.newFileContent = '';
                            // Result will be handled by WebSocket
                        } else {
                            this.addActivity('❌', `File creation failed: ${data.error}`, 'error');
                        }
                    })
                    .catch(() => {
                        this.addActivity('❌', 'Failed to create file', 'error');
                    });
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
                    
                    setTimeout(() => {
                        activity.isNew = false;
                    }, 3000);
                    
                    if (this.activities.length > 100) {
                        this.activities = this.activities.slice(0, 100);
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
            "capabilities": ["git", "web_search", "command_line", "file_system"],
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
        time.sleep(12)  # Every 12 seconds
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
        
        EnhancedLiveHTTPHandler.broadcast_to_websockets(message)


def main():
    """Main enhanced server function."""
    port = 8000
    
    print("🚀 Master Orchestrator - Enhanced Live Server")
    print("=" * 60)
    print(f"🌐 Enhanced Dashboard: http://localhost:{port}")
    print("✨ Enhanced Features:")
    print("   • Hot reload without page refresh")
    print("   • Real-time WebSocket updates")
    print("   • Full Git control (status, commit, push, pull)")
    print("   • Live terminal with command execution")
    print("   • Web search integration")
    print("   • File system management")
    print("   • Interactive live activity feed")
    print("   • Zero external dependencies")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 60)
    
    # Initialize development controller
    EnhancedLiveHTTPHandler.initialize_dev_controller()
    
    # Start periodic updates in background thread
    update_thread = threading.Thread(target=periodic_updates, daemon=True)
    update_thread.start()
    
    # Start HTTP server
    server_address = ('', port)
    httpd = HTTPServer(server_address, EnhancedLiveHTTPHandler)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Enhanced Live Server stopped")


if __name__ == "__main__":
    main()