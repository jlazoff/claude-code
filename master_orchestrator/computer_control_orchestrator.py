#!/usr/bin/env python3
"""
Computer Control Orchestrator - Master system for physical computer control
Integrates OpenHands, Codex, Aider with mouse/keyboard control and screen observation
"""

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import cv2
import numpy as np
import pyautogui
import pynput
from pynput import mouse, keyboard
import websockets
import aiohttp
from aiohttp import web
import psutil
import requests
from PIL import Image, ImageDraw
import base64
import io
import threading
import queue
import time

from unified_config import SecureConfigManager
from litellm_manager import LiteLLMManager

# Configure pyautogui for safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

class ScreenObserver:
    """Real-time screen monitoring and analysis"""
    
    def __init__(self):
        self.observers = []
        self.is_monitoring = False
        self.screenshot_queue = queue.Queue(maxsize=10)
        self.current_screenshot = None
        
    async def start_monitoring(self):
        """Start continuous screen monitoring"""
        self.is_monitoring = True
        monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitoring_thread.start()
        logging.info("Screen monitoring started")
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                screenshot = pyautogui.screenshot()
                self.current_screenshot = screenshot
                
                # Add to queue for processing
                if not self.screenshot_queue.full():
                    self.screenshot_queue.put(screenshot)
                    
                time.sleep(0.5)  # Monitor every 500ms
            except Exception as e:
                logging.error(f"Screenshot error: {e}")
                time.sleep(1)
                
    def get_current_screen(self) -> Optional[Image.Image]:
        """Get current screenshot"""
        return self.current_screenshot
        
    def find_element_on_screen(self, template_path: str) -> Optional[Tuple[int, int]]:
        """Find UI element using template matching"""
        if not self.current_screenshot:
            return None
            
        try:
            # Convert PIL to OpenCV format
            screen_cv = cv2.cvtColor(np.array(self.current_screenshot), cv2.COLOR_RGB2BGR)
            template = cv2.imread(template_path, cv2.IMREAD_COLOR)
            
            result = cv2.matchTemplate(screen_cv, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.8:  # High confidence threshold
                return max_loc
                
        except Exception as e:
            logging.error(f"Template matching error: {e}")
            
        return None
        
    def get_screen_analysis(self) -> Dict[str, Any]:
        """Analyze current screen for context"""
        if not self.current_screenshot:
            return {"status": "no_screen"}
            
        # Convert to base64 for LLM analysis
        buffer = io.BytesIO()
        self.current_screenshot.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "resolution": self.current_screenshot.size,
            "image_data": img_data,
            "format": "base64_png"
        }

class PhysicalController:
    """Direct mouse and keyboard control"""
    
    def __init__(self):
        self.mouse_controller = pynput.mouse.Controller()
        self.keyboard_controller = pynput.keyboard.Controller()
        self.action_history = []
        
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1):
        """Click at coordinates"""
        try:
            pyautogui.click(x, y, clicks=clicks, button=button)
            self._log_action("click", {"x": x, "y": y, "button": button, "clicks": clicks})
        except Exception as e:
            logging.error(f"Click error: {e}")
            
    def type_text(self, text: str, interval: float = 0.01):
        """Type text with specified interval"""
        try:
            pyautogui.write(text, interval=interval)
            self._log_action("type", {"text": text[:50] + "..." if len(text) > 50 else text})
        except Exception as e:
            logging.error(f"Type error: {e}")
            
    def key_combination(self, *keys):
        """Press key combination (e.g., cmd+c)"""
        try:
            pyautogui.hotkey(*keys)
            self._log_action("hotkey", {"keys": list(keys)})
        except Exception as e:
            logging.error(f"Hotkey error: {e}")
            
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 1.0):
        """Drag from start to end coordinates"""
        try:
            pyautogui.drag(end_x - start_x, end_y - start_y, duration=duration, button='left')
            self._log_action("drag", {"start": (start_x, start_y), "end": (end_x, end_y)})
        except Exception as e:
            logging.error(f"Drag error: {e}")
            
    def scroll(self, x: int, y: int, clicks: int = 3):
        """Scroll at position"""
        try:
            pyautogui.scroll(clicks, x, y)
            self._log_action("scroll", {"x": x, "y": y, "clicks": clicks})
        except Exception as e:
            logging.error(f"Scroll error: {e}")
            
    def _log_action(self, action_type: str, params: Dict):
        """Log performed action"""
        action_record = {
            "timestamp": datetime.now().isoformat(),
            "type": action_type,
            "params": params
        }
        self.action_history.append(action_record)
        
        # Keep only last 100 actions
        if len(self.action_history) > 100:
            self.action_history.pop(0)

class ToolIntegrator:
    """Integration with OpenHands, Codex, and Aider"""
    
    def __init__(self, config_manager: SecureConfigManager):
        self.config = config_manager
        self.openai_api_key = config_manager.get_api_key('openai')
        self.session = None
        
    async def initialize(self):
        """Initialize tool connections"""
        self.session = aiohttp.ClientSession()
        logging.info("Tool integrator initialized")
        
    async def query_codex(self, prompt: str, language: str = "python") -> Dict[str, Any]:
        """Query OpenAI Codex for code generation"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system", 
                        "content": f"You are a code generation assistant. Generate {language} code based on the request."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    return {
                        "success": True,
                        "code": result["choices"][0]["message"]["content"],
                        "model": result.get("model", "gpt-4")
                    }
                else:
                    return {"success": False, "error": result}
                    
        except Exception as e:
            logging.error(f"Codex query error: {e}")
            return {"success": False, "error": str(e)}
            
    async def run_aider_command(self, command: str, cwd: str = None) -> Dict[str, Any]:
        """Execute Aider command"""
        try:
            if not cwd:
                cwd = Path.cwd()
                
            process = await asyncio.create_subprocess_exec(
                "aider", *command.split(),
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "return_code": process.returncode
            }
            
        except Exception as e:
            logging.error(f"Aider command error: {e}")
            return {"success": False, "error": str(e)}
            
    async def launch_openhands(self, workspace_path: str) -> Dict[str, Any]:
        """Launch OpenHands environment"""
        try:
            # Check if OpenHands is available
            check_process = await asyncio.create_subprocess_exec(
                "which", "openhands",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await check_process.communicate()
            
            if check_process.returncode != 0:
                return {"success": False, "error": "OpenHands not found in PATH"}
                
            # Launch OpenHands
            process = await asyncio.create_subprocess_exec(
                "openhands", "--workspace", workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            return {
                "success": True,
                "pid": process.pid,
                "workspace": workspace_path
            }
            
        except Exception as e:
            logging.error(f"OpenHands launch error: {e}")
            return {"success": False, "error": str(e)}

class ComputerControlOrchestrator:
    """Main orchestrator for computer control system"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.llm_manager = LiteLLMManager(self.config)
        self.screen_observer = ScreenObserver()
        self.physical_controller = PhysicalController()
        self.tool_integrator = ToolIntegrator(self.config)
        self.websocket_clients = set()
        self.is_running = False
        
    async def initialize(self):
        """Initialize all components"""
        await self.config.initialize()
        await self.llm_manager.initialize()
        await self.tool_integrator.initialize()
        await self.screen_observer.start_monitoring()
        
        logging.info("Computer Control Orchestrator initialized")
        
    async def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming command"""
        try:
            command_type = command.get("type")
            params = command.get("params", {})
            
            if command_type == "click":
                self.physical_controller.click(
                    params["x"], params["y"], 
                    params.get("button", "left"),
                    params.get("clicks", 1)
                )
                return {"success": True, "action": "click_executed"}
                
            elif command_type == "type":
                self.physical_controller.type_text(params["text"])
                return {"success": True, "action": "text_typed"}
                
            elif command_type == "hotkey":
                self.physical_controller.key_combination(*params["keys"])
                return {"success": True, "action": "hotkey_executed"}
                
            elif command_type == "analyze_screen":
                analysis = self.screen_observer.get_screen_analysis()
                
                # Send to LLM for interpretation
                llm_response = await self.llm_manager.generate_response(
                    "You are analyzing a computer screen. Describe what you see and suggest possible actions.",
                    model="gpt-4-vision-preview"
                )
                
                analysis["llm_interpretation"] = llm_response.get("content", "")
                return {"success": True, "analysis": analysis}
                
            elif command_type == "generate_code":
                result = await self.tool_integrator.query_codex(
                    params["prompt"], 
                    params.get("language", "python")
                )
                return result
                
            elif command_type == "aider_command":
                result = await self.tool_integrator.run_aider_command(
                    params["command"],
                    params.get("cwd")
                )
                return result
                
            elif command_type == "launch_openhands":
                result = await self.tool_integrator.launch_openhands(
                    params["workspace_path"]
                )
                return result
                
            else:
                return {"success": False, "error": f"Unknown command type: {command_type}"}
                
        except Exception as e:
            logging.error(f"Command processing error: {e}")
            return {"success": False, "error": str(e)}
            
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time chat"""
        self.websocket_clients.add(websocket)
        logging.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "chat_message":
                        # Process chat message and generate response
                        response = await self.llm_manager.generate_response(
                            data["message"],
                            context={
                                "screen_analysis": self.screen_observer.get_screen_analysis(),
                                "recent_actions": self.physical_controller.action_history[-5:],
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        
                        await self.broadcast_message({
                            "type": "chat_response",
                            "message": response.get("content", ""),
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    elif data.get("type") == "command":
                        # Execute computer control command
                        result = await self.process_command(data)
                        await websocket.send(json.dumps({
                            "type": "command_result",
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        }))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    logging.error(f"WebSocket message error: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logging.info("WebSocket client disconnected")
        finally:
            self.websocket_clients.discard(websocket)
            
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        if self.websocket_clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.websocket_clients],
                return_exceptions=True
            )
            
    async def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time communication"""
        server = await websockets.serve(self.websocket_handler, host, port)
        logging.info(f"WebSocket server started on ws://{host}:{port}")
        return server
        
    async def run_background_service(self):
        """Run as background service with continuous monitoring"""
        self.is_running = True
        logging.info("Computer Control Orchestrator running in background")
        
        while self.is_running:
            try:
                # Periodic health checks and optimizations
                await self._health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Background service error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    async def _health_check(self):
        """Perform system health checks"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Check if monitoring is active
            monitoring_active = self.screen_observer.is_monitoring
            
            # Check WebSocket connections
            active_connections = len(self.websocket_clients)
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "monitoring_active": monitoring_active,
                "websocket_connections": active_connections,
                "status": "healthy" if cpu_percent < 80 and memory_percent < 80 else "warning"
            }
            
            # Broadcast status to connected clients
            await self.broadcast_message({
                "type": "health_status",
                "status": status
            })
            
        except Exception as e:
            logging.error(f"Health check error: {e}")

async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = ComputerControlOrchestrator()
    await orchestrator.initialize()
    
    # Start WebSocket server
    websocket_server = await orchestrator.start_websocket_server()
    
    # Start background service
    background_task = asyncio.create_task(orchestrator.run_background_service())
    
    try:
        # Keep running
        await asyncio.gather(websocket_server.wait_closed(), background_task)
    except KeyboardInterrupt:
        logging.info("Shutting down Computer Control Orchestrator")
        orchestrator.is_running = False
        websocket_server.close()
        await websocket_server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())