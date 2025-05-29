#!/usr/bin/env python3
"""
AG-UI Next.js Frontend Orchestrator - Complete Frontend System
Uses Next.js, TailwindCSS, AG-UI standardization with static assets
Provides comprehensive UI for monitoring, managing, and creating projects
Human-in-the-loop async capabilities with real-time updates
"""

import asyncio
import json
import yaml
import logging
import subprocess
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
import aiofiles
import aiohttp
from datetime import datetime, timedelta
import websockets
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import psutil
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrontendComponent(BaseModel):
    """Pydantic model for frontend components"""
    id: str
    name: str
    component_type: str  # dashboard, monitor, control, create, manage
    data_sources: List[str]
    update_frequency: int  # seconds
    config: Dict[str, Any]
    is_active: bool = True

class UIState(BaseModel):
    """Pydantic model for UI state management"""
    current_view: str
    active_projects: List[str]
    selected_components: List[str]
    user_preferences: Dict[str, Any]
    notification_settings: Dict[str, Any]
    human_in_loop_queue: List[Dict[str, Any]]

class AGUINextJSFrontendOrchestrator:
    """
    Complete Next.js Frontend System with AG-UI standardization
    """
    
    def __init__(self):
        self.base_dir = Path("foundation_data")
        self.frontend_dir = self.base_dir / "agui_frontend"
        self.static_assets_dir = self.frontend_dir / "static"
        self.components_dir = self.frontend_dir / "components"
        self.pages_dir = self.frontend_dir / "pages"
        self.api_dir = self.frontend_dir / "api"
        
        # FastAPI backend for API and WebSocket
        self.app = FastAPI(title="AG-UI Master Orchestrator Frontend")
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.ui_state = UIState(
            current_view="dashboard",
            active_projects=[],
            selected_components=[],
            user_preferences={},
            notification_settings={},
            human_in_loop_queue=[]
        )
        
        # Frontend components
        self.frontend_components: Dict[str, FrontendComponent] = {}
        
        # Static assets optimization
        self.static_assets_cache = {}
        self.asset_optimization_enabled = True
        
        self._initialize_directories()
        self._setup_fastapi_app()
        
    def _initialize_directories(self):
        """Initialize all frontend directories"""
        directories = [
            self.frontend_dir,
            self.static_assets_dir,
            self.components_dir,
            self.pages_dir,
            self.api_dir,
            self.frontend_dir / "styles",
            self.frontend_dir / "public",
            self.frontend_dir / "lib",
            self.frontend_dir / "hooks",
            self.frontend_dir / "utils",
            self.static_assets_dir / "images",
            self.static_assets_dir / "icons",
            self.static_assets_dir / "css",
            self.static_assets_dir / "js"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _setup_fastapi_app(self):
        """Setup FastAPI application with all routes and middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(self.static_assets_dir)), name="static")
        
        # Setup routes
        self._setup_api_routes()
        self._setup_websocket_routes()
        
    def _setup_api_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def root():
            return FileResponse(str(self.frontend_dir / "index.html"))
            
        @self.app.get("/api/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
            
        @self.app.get("/api/ui-state")
        async def get_ui_state():
            return self.ui_state.dict()
            
        @self.app.post("/api/ui-state")
        async def update_ui_state(state_update: Dict[str, Any]):
            for key, value in state_update.items():
                if hasattr(self.ui_state, key):
                    setattr(self.ui_state, key, value)
            await self._broadcast_ui_update("ui_state", self.ui_state.dict())
            return {"success": True}
            
        @self.app.get("/api/projects")
        async def get_projects():
            # This would integrate with the main orchestrator
            return {"projects": [], "count": 0}
            
        @self.app.get("/api/mcp-servers")
        async def get_mcp_servers():
            # This would integrate with the MCP orchestrator
            return {"mcp_servers": [], "count": 0}
            
        @self.app.get("/api/system-metrics")
        async def get_system_metrics():
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "timestamp": datetime.now().isoformat()
            }
            
        @self.app.post("/api/create-project")
        async def create_project(project_data: Dict[str, Any]):
            # This would integrate with project creation system
            project_id = str(uuid.uuid4())
            await self._broadcast_ui_update("project_created", {"id": project_id, **project_data})
            return {"success": True, "project_id": project_id}
            
        @self.app.post("/api/human-in-loop")
        async def submit_human_feedback(feedback: Dict[str, Any]):
            self.ui_state.human_in_loop_queue.append({
                "id": str(uuid.uuid4()),
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            await self._broadcast_ui_update("human_feedback", feedback)
            return {"success": True}
            
    def _setup_websocket_routes(self):
        """Setup WebSocket routes for real-time communication"""
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            await websocket.accept()
            self.websocket_connections[client_id] = websocket
            
            try:
                # Send initial state
                await websocket.send_json({
                    "type": "initial_state",
                    "data": self.ui_state.dict()
                })
                
                # Keep connection alive and handle messages
                while True:
                    message = await websocket.receive_json()
                    await self._handle_websocket_message(client_id, message)
                    
            except WebSocketDisconnect:
                if client_id in self.websocket_connections:
                    del self.websocket_connections[client_id]
                    
    async def _handle_websocket_message(self, client_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = message.get("type")
        data = message.get("data", {})
        
        if message_type == "ping":
            await self.websocket_connections[client_id].send_json({"type": "pong"})
            
        elif message_type == "subscribe":
            # Handle subscription to specific data streams
            await self._handle_subscription(client_id, data)
            
        elif message_type == "ui_action":
            # Handle UI actions
            await self._handle_ui_action(client_id, data)
            
        elif message_type == "human_input":
            # Handle human-in-the-loop input
            await self._handle_human_input(client_id, data)
            
    async def _broadcast_ui_update(self, update_type: str, data: Any):
        """Broadcast updates to all connected clients"""
        message = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected_clients = []
        for client_id, websocket in self.websocket_connections.items():
            try:
                await websocket.send_json(message)
            except:
                disconnected_clients.append(client_id)
                
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            if client_id in self.websocket_connections:
                del self.websocket_connections[client_id]
                
    async def initialize(self):
        """Initialize the frontend orchestrator"""
        logger.info("🎨 Initializing AG-UI Next.js Frontend...")
        
        # Setup Next.js project
        await self._setup_nextjs_project()
        
        # Generate AG-UI components
        await self._generate_agui_components()
        
        # Optimize static assets
        await self._optimize_static_assets()
        
        # Create dashboard components
        await self._create_dashboard_components()
        
        # Setup real-time data streams
        await self._setup_data_streams()
        
        logger.info("✅ AG-UI Next.js Frontend initialized")
        
    async def _setup_nextjs_project(self):
        """Setup Next.js project with all dependencies"""
        logger.info("📦 Setting up Next.js project...")
        
        # Check if Node.js is installed
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Node.js not found")
        except:
            logger.error("Node.js is required but not found. Please install Node.js")
            return
            
        # Create package.json
        package_json = {
            "name": "agui-master-orchestrator",
            "version": "1.0.0",
            "private": True,
            "scripts": {
                "dev": "next dev",
                "build": "next build",
                "start": "next start",
                "lint": "next lint",
                "export": "next export"
            },
            "dependencies": {
                "next": "^14.0.0",
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "@ag-grid-community/react": "^31.0.0",
                "@ag-grid-community/core": "^31.0.0",
                "@ag-grid-community/client-side-row-model": "^31.0.0",
                "tailwindcss": "^3.3.0",
                "autoprefixer": "^10.4.0",
                "postcss": "^8.4.0",
                "@headlessui/react": "^1.7.0",
                "@heroicons/react": "^2.0.0",
                "recharts": "^2.8.0",
                "react-flow-renderer": "^10.3.0",
                "socket.io-client": "^4.7.0",
                "date-fns": "^2.30.0",
                "lucide-react": "^0.292.0",
                "clsx": "^2.0.0",
                "framer-motion": "^10.16.0"
            },
            "devDependencies": {
                "typescript": "^5.2.0",
                "@types/node": "^20.8.0",
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "eslint": "^8.52.0",
                "eslint-config-next": "^14.0.0"
            }
        }
        
        # Write package.json
        async with aiofiles.open(self.frontend_dir / "package.json", "w") as f:
            await f.write(json.dumps(package_json, indent=2))
            
        # Create Next.js config
        nextjs_config = '''/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  experimental: {
    appDir: true
  },
  webpack: (config) => {
    config.experiments = {
      ...config.experiments,
      topLevelAwait: true,
    }
    return config
  }
}

module.exports = nextConfig
'''
        
        async with aiofiles.open(self.frontend_dir / "next.config.js", "w") as f:
            await f.write(nextjs_config)
            
        # Create TailwindCSS config
        tailwind_config = '''/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'ag-blue': '#007acc',
        'ag-dark': '#1a202c',
        'ag-gray': '#2d3748',
        'ag-green': '#38a169',
        'ag-red': '#e53e3e',
        'ag-yellow': '#d69e2e'
      },
      fontFamily: {
        'mono': ['Monaco', 'Menlo', 'monospace'],
        'sans': ['Inter', 'system-ui', 'sans-serif']
      }
    },
  },
  plugins: [],
}
'''
        
        async with aiofiles.open(self.frontend_dir / "tailwind.config.js", "w") as f:
            await f.write(tailwind_config)
            
        # Create PostCSS config
        postcss_config = '''module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
'''
        
        async with aiofiles.open(self.frontend_dir / "postcss.config.js", "w") as f:
            await f.write(postcss_config)
            
        # Install dependencies
        original_cwd = os.getcwd()
        try:
            os.chdir(self.frontend_dir)
            result = subprocess.run(["npm", "install"], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"npm install failed: {result.stderr}")
        finally:
            os.chdir(original_cwd)
            
    async def _generate_agui_components(self):
        """Generate AG-UI standardized components"""
        logger.info("🧩 Generating AG-UI components...")
        
        # Create main layout component
        layout_component = '''import React from 'react'
import Head from 'next/head'
import { Inter } from 'next/font/google'
import Navigation from './Navigation'
import Sidebar from './Sidebar'
import StatusBar from './StatusBar'

const inter = Inter({ subsets: ['latin'] })

interface LayoutProps {
  children: React.ReactNode
  title?: string
}

export default function Layout({ children, title = 'AG-UI Master Orchestrator' }: LayoutProps) {
  return (
    <div className={`min-h-screen bg-ag-dark text-white ${inter.className}`}>
      <Head>
        <title>{title}</title>
        <meta name="description" content="AG-UI Master Orchestrator - Comprehensive AI System Management" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <div className="flex h-screen">
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Navigation />
          <main className="flex-1 overflow-auto p-6">
            {children}
          </main>
          <StatusBar />
        </div>
      </div>
    </div>
  )
}
'''
        
        async with aiofiles.open(self.components_dir / "Layout.tsx", "w") as f:
            await f.write(layout_component)
            
        # Create Navigation component
        navigation_component = '''import React, { useState, useEffect } from 'react'
import { BellIcon, CogIcon, UserIcon } from '@heroicons/react/24/outline'
import { useWebSocket } from '../hooks/useWebSocket'

export default function Navigation() {
  const [notifications, setNotifications] = useState(0)
  const { isConnected, lastMessage } = useWebSocket()
  
  useEffect(() => {
    if (lastMessage?.type === 'notification') {
      setNotifications(prev => prev + 1)
    }
  }, [lastMessage])
  
  return (
    <nav className="bg-ag-gray border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h1 className="text-xl font-bold text-ag-blue">AG-UI Master Orchestrator</h1>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-ag-green' : 'bg-ag-red'}`} />
          <span className="text-sm text-gray-400">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="relative p-2 text-gray-400 hover:text-white">
            <BellIcon className="w-6 h-6" />
            {notifications > 0 && (
              <span className="absolute -top-1 -right-1 bg-ag-red text-xs rounded-full w-5 h-5 flex items-center justify-center">
                {notifications > 9 ? '9+' : notifications}
              </span>
            )}
          </button>
          
          <button className="p-2 text-gray-400 hover:text-white">
            <CogIcon className="w-6 h-6" />
          </button>
          
          <button className="p-2 text-gray-400 hover:text-white">
            <UserIcon className="w-6 h-6" />
          </button>
        </div>
      </div>
    </nav>
  )
}
'''
        
        async with aiofiles.open(self.components_dir / "Navigation.tsx", "w") as f:
            await f.write(navigation_component)
            
        # Create Sidebar component
        sidebar_component = '''import React, { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/router'
import {
  HomeIcon,
  ServerIcon,
  CubeIcon,
  ChartBarIcon,
  CogIcon,
  FolderIcon,
  UserGroupIcon,
  BeakerIcon
} from '@heroicons/react/24/outline'

const menuItems = [
  { name: 'Dashboard', href: '/', icon: HomeIcon },
  { name: 'Projects', href: '/projects', icon: FolderIcon },
  { name: 'MCP Servers', href: '/mcp-servers', icon: ServerIcon },
  { name: 'Components', href: '/components', icon: CubeIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
  { name: 'Testing', href: '/testing', icon: BeakerIcon },
  { name: 'Teams', href: '/teams', icon: UserGroupIcon },
  { name: 'Settings', href: '/settings', icon: CogIcon },
]

export default function Sidebar() {
  const router = useRouter()
  const [collapsed, setCollapsed] = useState(false)
  
  return (
    <aside className={`bg-ag-gray border-r border-gray-700 transition-all duration-300 ${
      collapsed ? 'w-16' : 'w-64'
    }`}>
      <div className="p-4">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full text-left text-ag-blue hover:text-blue-300 mb-6"
        >
          {collapsed ? '→' : '←'}
        </button>
        
        <nav className="space-y-2">
          {menuItems.map((item) => {
            const isActive = router.pathname === item.href
            const Icon = item.icon
            
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-ag-blue text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                <Icon className="w-5 h-5 mr-3" />
                {!collapsed && item.name}
              </Link>
            )
          })}
        </nav>
      </div>
    </aside>
  )
}
'''
        
        async with aiofiles.open(self.components_dir / "Sidebar.tsx", "w") as f:
            await f.write(sidebar_component)
            
        # Create WebSocket hook
        websocket_hook = '''import { useState, useEffect, useRef } from 'react'

interface WebSocketMessage {
  type: string
  data: any
  timestamp: string
}

export function useWebSocket(url?: string) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null)
  
  const wsUrl = url || `ws://localhost:8000/ws/${Math.random().toString(36).substr(2, 9)}`
  
  const connect = () => {
    try {
      ws.current = new WebSocket(wsUrl)
      
      ws.current.onopen = () => {
        setIsConnected(true)
        setConnectionError(null)
        console.log('WebSocket connected')
      }
      
      ws.current.onmessage = (event) => {
        const message = JSON.parse(event.data)
        setLastMessage(message)
      }
      
      ws.current.onclose = () => {
        setIsConnected(false)
        console.log('WebSocket disconnected')
        
        // Attempt to reconnect after 3 seconds
        reconnectTimeout.current = setTimeout(() => {
          connect()
        }, 3000)
      }
      
      ws.current.onerror = (error) => {
        setConnectionError('WebSocket connection error')
        console.error('WebSocket error:', error)
      }
    } catch (error) {
      setConnectionError('Failed to create WebSocket connection')
      console.error('WebSocket creation error:', error)
    }
  }
  
  useEffect(() => {
    connect()
    
    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current)
      }
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [])
  
  const sendMessage = (message: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message))
    }
  }
  
  return {
    isConnected,
    lastMessage,
    connectionError,
    sendMessage
  }
}
'''
        
        async with aiofiles.open(self.frontend_dir / "hooks" / "useWebSocket.ts", "w") as f:
            await f.write(websocket_hook)
            
    async def _create_dashboard_components(self):
        """Create comprehensive dashboard components"""
        logger.info("📊 Creating dashboard components...")
        
        # Main dashboard page
        dashboard_page = '''import React, { useState, useEffect } from 'react'
import Layout from '../components/Layout'
import SystemMetrics from '../components/dashboard/SystemMetrics'
import ProjectsOverview from '../components/dashboard/ProjectsOverview'
import MCPServersStatus from '../components/dashboard/MCPServersStatus'
import TaskExecution from '../components/dashboard/TaskExecution'
import HumanInLoopQueue from '../components/dashboard/HumanInLoopQueue'
import { useWebSocket } from '../hooks/useWebSocket'

export default function Dashboard() {
  const [systemData, setSystemData] = useState(null)
  const { lastMessage } = useWebSocket()
  
  useEffect(() => {
    if (lastMessage?.type === 'system_metrics') {
      setSystemData(lastMessage.data)
    }
  }, [lastMessage])
  
  return (
    <Layout title="Dashboard - AG-UI Master Orchestrator">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-white">Dashboard</h1>
          <div className="flex space-x-4">
            <button className="bg-ag-blue text-white px-4 py-2 rounded-lg hover:bg-blue-600">
              Create Project
            </button>
            <button className="bg-ag-green text-white px-4 py-2 rounded-lg hover:bg-green-600">
              Run Task
            </button>
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          <SystemMetrics data={systemData} />
          <ProjectsOverview />
          <MCPServersStatus />
        </div>
        
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <TaskExecution />
          <HumanInLoopQueue />
        </div>
      </div>
    </Layout>
  )
}
'''
        
        async with aiofiles.open(self.pages_dir / "index.tsx", "w") as f:
            await f.write(dashboard_page)
            
        # Create dashboard components directory
        dashboard_components_dir = self.components_dir / "dashboard"
        dashboard_components_dir.mkdir(exist_ok=True)
        
        # System Metrics component
        system_metrics_component = '''import React from 'react'
import { motion } from 'framer-motion'

interface SystemMetricsProps {
  data: any
}

export default function SystemMetrics({ data }: SystemMetricsProps) {
  const metrics = data || {
    cpu_usage: 0,
    memory_usage: 0,
    disk_usage: 0
  }
  
  const getUsageColor = (usage: number) => {
    if (usage > 80) return 'text-ag-red'
    if (usage > 60) return 'text-ag-yellow'
    return 'text-ag-green'
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-ag-gray rounded-lg p-6 border border-gray-700"
    >
      <h3 className="text-lg font-semibold mb-4 text-white">System Metrics</h3>
      
      <div className="space-y-4">
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300">CPU Usage</span>
            <span className={`font-mono ${getUsageColor(metrics.cpu_usage)}`}>
              {metrics.cpu_usage.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-ag-blue h-2 rounded-full transition-all duration-300"
              style={{ width: `${metrics.cpu_usage}%` }}
            />
          </div>
        </div>
        
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300">Memory Usage</span>
            <span className={`font-mono ${getUsageColor(metrics.memory_usage)}`}>
              {metrics.memory_usage.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-ag-green h-2 rounded-full transition-all duration-300"
              style={{ width: `${metrics.memory_usage}%` }}
            />
          </div>
        </div>
        
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300">Disk Usage</span>
            <span className={`font-mono ${getUsageColor(metrics.disk_usage)}`}>
              {metrics.disk_usage.toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-ag-yellow h-2 rounded-full transition-all duration-300"
              style={{ width: `${metrics.disk_usage}%` }}
            />
          </div>
        </div>
      </div>
    </motion.div>
  )
}
'''
        
        async with aiofiles.open(dashboard_components_dir / "SystemMetrics.tsx", "w") as f:
            await f.write(system_metrics_component)
            
    async def _optimize_static_assets(self):
        """Optimize static assets for performance"""
        logger.info("⚡ Optimizing static assets...")
        
        # Create optimized CSS
        optimized_css = '''@tailwind base;
@tailwind components;
@tailwind utilities;

/* AG-UI Custom Styles */
:root {
  --ag-blue: #007acc;
  --ag-dark: #1a202c;
  --ag-gray: #2d3748;
  --ag-green: #38a169;
  --ag-red: #e53e3e;
  --ag-yellow: #d69e2e;
}

.ag-card {
  @apply bg-ag-gray rounded-lg p-6 border border-gray-700 shadow-lg;
}

.ag-button {
  @apply px-4 py-2 rounded-lg font-medium transition-colors duration-200;
}

.ag-button-primary {
  @apply ag-button bg-ag-blue text-white hover:bg-blue-600;
}

.ag-button-secondary {
  @apply ag-button bg-gray-600 text-white hover:bg-gray-700;
}

.ag-input {
  @apply w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-ag-blue;
}

.ag-grid-theme {
  --ag-header-background-color: var(--ag-gray);
  --ag-header-foreground-color: white;
  --ag-background-color: var(--ag-dark);
  --ag-foreground-color: white;
  --ag-border-color: #4a5568;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--ag-gray);
}

::-webkit-scrollbar-thumb {
  background: var(--ag-blue);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #0066aa;
}

/* Loading animations */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.ag-loading {
  animation: pulse 2s infinite;
}

/* Responsive utilities */
@screen sm {
  .ag-grid-responsive {
    font-size: 0.875rem;
  }
}
'''
        
        async with aiofiles.open(self.static_assets_dir / "css" / "globals.css", "w") as f:
            await f.write(optimized_css)
            
        # Create utility JavaScript
        utility_js = '''// AG-UI Utility Functions
window.AGUI = {
  // WebSocket connection utilities
  createWebSocketConnection: (url, onMessage, onError) => {
    const ws = new WebSocket(url);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };
    ws.onerror = onError;
    return ws;
  },
  
  // Local storage utilities
  storage: {
    set: (key, value) => {
      try {
        localStorage.setItem(`agui_${key}`, JSON.stringify(value));
      } catch (error) {
        console.error('Storage set error:', error);
      }
    },
    get: (key, defaultValue = null) => {
      try {
        const value = localStorage.getItem(`agui_${key}`);
        return value ? JSON.parse(value) : defaultValue;
      } catch (error) {
        console.error('Storage get error:', error);
        return defaultValue;
      }
    },
    remove: (key) => {
      localStorage.removeItem(`agui_${key}`);
    }
  },
  
  // Performance monitoring
  performance: {
    mark: (name) => {
      if (performance.mark) {
        performance.mark(name);
      }
    },
    measure: (name, startMark, endMark) => {
      if (performance.measure) {
        performance.measure(name, startMark, endMark);
      }
    }
  },
  
  // Theme utilities
  theme: {
    toggle: () => {
      const current = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.classList.toggle('dark');
      AGUI.storage.set('theme', next);
    },
    apply: (theme) => {
      if (theme === 'dark') {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      AGUI.storage.set('theme', theme);
    }
  }
};

// Initialize theme on load
document.addEventListener('DOMContentLoaded', () => {
  const savedTheme = AGUI.storage.get('theme', 'dark');
  AGUI.theme.apply(savedTheme);
});
'''
        
        async with aiofiles.open(self.static_assets_dir / "js" / "agui-utils.js", "w") as f:
            await f.write(utility_js)
            
    async def _setup_data_streams(self):
        """Setup real-time data streams"""
        logger.info("📡 Setting up real-time data streams...")
        
        # Start background tasks for data streaming
        asyncio.create_task(self._stream_system_metrics())
        asyncio.create_task(self._stream_project_updates())
        asyncio.create_task(self._stream_mcp_status())
        asyncio.create_task(self._stream_task_execution())
        
    async def _stream_system_metrics(self):
        """Stream system metrics to connected clients"""
        while True:
            try:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = {
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "disk_usage": disk.percent,
                    "timestamp": datetime.now().isoformat()
                }
                
                await self._broadcast_ui_update("system_metrics", metrics)
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error streaming system metrics: {e}")
                await asyncio.sleep(10)
                
    async def _stream_project_updates(self):
        """Stream project updates to connected clients"""
        while True:
            try:
                # This would integrate with the main orchestrator to get project updates
                projects_data = {
                    "active_projects": len(self.ui_state.active_projects),
                    "total_projects": 0,  # Would get from database
                    "recent_activity": [],
                    "timestamp": datetime.now().isoformat()
                }
                
                await self._broadcast_ui_update("projects_update", projects_data)
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error streaming project updates: {e}")
                await asyncio.sleep(60)
                
    async def _stream_mcp_status(self):
        """Stream MCP server status to connected clients"""
        while True:
            try:
                # This would integrate with MCP orchestrator
                mcp_data = {
                    "total_servers": 0,
                    "active_servers": 0,
                    "failed_servers": 0,
                    "server_details": [],
                    "timestamp": datetime.now().isoformat()
                }
                
                await self._broadcast_ui_update("mcp_status", mcp_data)
                await asyncio.sleep(15)  # Update every 15 seconds
                
            except Exception as e:
                logger.error(f"Error streaming MCP status: {e}")
                await asyncio.sleep(30)
                
    async def _stream_task_execution(self):
        """Stream task execution updates to connected clients"""
        while True:
            try:
                # This would integrate with task execution system
                task_data = {
                    "running_tasks": 0,
                    "completed_tasks": 0,
                    "failed_tasks": 0,
                    "queue_length": 0,
                    "recent_executions": [],
                    "timestamp": datetime.now().isoformat()
                }
                
                await self._broadcast_ui_update("task_execution", task_data)
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error streaming task execution: {e}")
                await asyncio.sleep(20)
                
    async def start_frontend_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the frontend server"""
        logger.info(f"🌐 Starting AG-UI frontend server on {host}:{port}")
        
        # Start Next.js development server in background
        nextjs_process = None
        try:
            original_cwd = os.getcwd()
            os.chdir(self.frontend_dir)
            
            # Start Next.js dev server
            nextjs_process = subprocess.Popen(
                ["npm", "run", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Start FastAPI server
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Error starting frontend server: {e}")
        finally:
            if nextjs_process:
                nextjs_process.terminate()
            os.chdir(original_cwd)

async def main():
    """Main execution function"""
    frontend = AGUINextJSFrontendOrchestrator()
    await frontend.initialize()
    await frontend.start_frontend_server()

if __name__ == "__main__":
    asyncio.run(main())