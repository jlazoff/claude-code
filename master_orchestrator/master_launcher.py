#!/usr/bin/env python3
"""
Master Launcher - Complete System Initialization and Autonomous Generation
This is the central command to start the entire Master Orchestrator ecosystem
"""

import asyncio
import logging
import sys
import os
import signal
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess

# Import all major components
from unified_config import SecureConfigManager
from parallel_llm_orchestrator import ParallelLLMOrchestrator
from enterprise_agent_ecosystem import EnterpriseAgentEcosystem
from frontend_orchestrator import FrontendOrchestrator
from computer_control_orchestrator import ComputerControlOrchestrator
from content_analyzer_deployer import ContentAnalyzerDeployer
from github_integration import AutomatedDevelopmentWorkflow
from conversation_project_initiator import ConversationProjectInitiator
from autonomous_code_generator import AutonomousCodeGenerator
from one_click_deploy import OneClickDeployer

class MasterLauncher:
    """Master launcher for the complete ecosystem"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.components = {}
        self.services = {}
        self.is_running = False
        self.startup_time = None
        
        # Component initialization order (dependencies first)
        self.component_order = [
            ("config", SecureConfigManager, {}),
            ("llm_orchestrator", ParallelLLMOrchestrator, {}),
            ("agent_ecosystem", EnterpriseAgentEcosystem, {}),
            ("computer_control", ComputerControlOrchestrator, {}),
            ("content_analyzer", ContentAnalyzerDeployer, {}),
            ("dev_workflow", AutomatedDevelopmentWorkflow, {}),
            ("project_initiator", ConversationProjectInitiator, {}),
            ("frontend", FrontendOrchestrator, {}),
            ("autonomous_generator", AutonomousCodeGenerator, {}),
        ]
        
    async def initialize_all_systems(self):
        """Initialize all system components"""
        self.startup_time = datetime.now()
        
        logging.info("🚀 MASTER ORCHESTRATOR INITIALIZATION STARTING")
        logging.info("=" * 80)
        
        total_components = len(self.component_order)
        
        for i, (name, component_class, init_args) in enumerate(self.component_order, 1):
            try:
                logging.info(f"📦 [{i}/{total_components}] Initializing {name}...")
                
                if name == "config":
                    component = self.config
                else:
                    component = component_class()
                    
                await component.initialize()
                self.components[name] = component
                
                logging.info(f"✅ {name} initialized successfully")
                
            except Exception as e:
                logging.error(f"❌ Failed to initialize {name}: {e}")
                # Continue with other components
                
        logging.info("=" * 80)
        logging.info(f"🎯 Initialization completed: {len(self.components)}/{total_components} components ready")
        
    async def start_all_services(self):
        """Start all background services"""
        logging.info("🔧 STARTING ALL SERVICES")
        logging.info("-" * 50)
        
        # Start frontend server
        if "frontend" in self.components:
            try:
                frontend_runner = await self.components["frontend"].start_server(
                    host="0.0.0.0", 
                    port=8080
                )
                self.services["frontend"] = frontend_runner
                logging.info("🌐 Frontend server started on http://0.0.0.0:8080")
            except Exception as e:
                logging.error(f"❌ Failed to start frontend: {e}")
                
        # Start computer control WebSocket server
        if "computer_control" in self.components:
            try:
                websocket_server = await self.components["computer_control"].start_websocket_server(
                    host="0.0.0.0",
                    port=8765
                )
                self.services["websocket"] = websocket_server
                logging.info("🔌 WebSocket server started on ws://0.0.0.0:8765")
            except Exception as e:
                logging.error(f"❌ Failed to start WebSocket server: {e}")
                
        # Start autonomous code generation
        if "autonomous_generator" in self.components:
            try:
                generation_task = asyncio.create_task(
                    self.components["autonomous_generator"].start_autonomous_generation()
                )
                self.services["autonomous_generation"] = generation_task
                logging.info("🤖 Autonomous code generation started")
            except Exception as e:
                logging.error(f"❌ Failed to start autonomous generation: {e}")
                
        # Start background monitoring
        monitoring_task = asyncio.create_task(self._system_monitoring_loop())
        self.services["monitoring"] = monitoring_task
        
        logging.info("-" * 50)
        logging.info(f"⚡ All services started: {len(self.services)} services running")
        
    async def _system_monitoring_loop(self):
        """System-wide monitoring and health checks"""
        while self.is_running:
            try:
                # Collect system status
                status = await self._collect_system_status()
                
                # Log periodic status
                active_components = sum(1 for comp in self.components.values() if hasattr(comp, 'is_running') and comp.is_running)
                
                logging.info(f"💓 System Health Check: {active_components}/{len(self.components)} components active")
                
                # Check for issues and auto-recover
                await self._auto_recovery_check()
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
                
    async def _collect_system_status(self) -> Dict[str, Any]:
        """Collect comprehensive system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
            "components": {},
            "services": {},
            "overall_health": "healthy"
        }
        
        # Check component status
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_status'):
                    component_status = component.get_status()
                elif hasattr(component, 'is_running'):
                    component_status = {"running": component.is_running}
                else:
                    component_status = {"initialized": True}
                    
                status["components"][name] = component_status
                
            except Exception as e:
                status["components"][name] = {"error": str(e)}
                status["overall_health"] = "degraded"
                
        # Check service status
        for name, service in self.services.items():
            try:
                if asyncio.iscoroutine(service) or asyncio.isfuture(service):
                    status["services"][name] = {
                        "running": not service.done() if hasattr(service, 'done') else True
                    }
                else:
                    status["services"][name] = {"running": True}
                    
            except Exception as e:
                status["services"][name] = {"error": str(e)}
                status["overall_health"] = "degraded"
                
        return status
        
    async def _auto_recovery_check(self):
        """Check for failed components and attempt recovery"""
        try:
            # Check if frontend is still responding
            if "frontend" in self.services:
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get("http://localhost:8080/health", timeout=5) as response:
                            if response.status != 200:
                                logging.warning("Frontend health check failed, attempting restart...")
                                await self._restart_frontend()
                except Exception:
                    logging.warning("Frontend not responding, attempting restart...")
                    await self._restart_frontend()
                    
        except Exception as e:
            logging.error(f"Auto-recovery error: {e}")
            
    async def _restart_frontend(self):
        """Restart frontend service"""
        try:
            if "frontend" in self.services:
                # Stop existing service
                old_runner = self.services["frontend"]
                if hasattr(old_runner, 'cleanup'):
                    await old_runner.cleanup()
                    
                # Start new service
                frontend_runner = await self.components["frontend"].start_server(
                    host="0.0.0.0", 
                    port=8080
                )
                self.services["frontend"] = frontend_runner
                logging.info("🔄 Frontend service restarted")
                
        except Exception as e:
            logging.error(f"Frontend restart failed: {e}")
            
    async def run_interactive_mode(self):
        """Run in interactive mode with user commands"""
        self.is_running = True
        
        print("\n" + "=" * 80)
        print("🎯 MASTER ORCHESTRATOR - INTERACTIVE MODE")
        print("=" * 80)
        print("\nAvailable commands:")
        print("  status    - Show system status")
        print("  generate  - Trigger manual code generation")
        print("  deploy    - Deploy a project")
        print("  commit    - Commit current changes")
        print("  optimize  - Run system optimization")
        print("  logs      - Show recent logs")
        print("  stop      - Stop all services")
        print("  help      - Show this help")
        print("\nPress Ctrl+C to exit gracefully")
        print("-" * 80)
        
        try:
            while self.is_running:
                # Show status periodically
                await asyncio.sleep(10)
                
                # Check for user interrupt
                if not self.is_running:
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Graceful shutdown initiated...")
            await self.graceful_shutdown()
            
    async def run_autonomous_mode(self):
        """Run in fully autonomous mode"""
        self.is_running = True
        
        print("\n" + "=" * 80)
        print("🤖 MASTER ORCHESTRATOR - AUTONOMOUS MODE")
        print("=" * 80)
        print("Running in autonomous mode. The system will:")
        print("  ✓ Generate code continuously")
        print("  ✓ Monitor and optimize performance")
        print("  ✓ Commit improvements automatically")
        print("  ✓ Deploy updates seamlessly")
        print("  ✓ Self-improve and adapt")
        print("\nPress Ctrl+C to stop")
        print("-" * 80)
        
        try:
            # Run indefinitely in autonomous mode
            while self.is_running:
                await asyncio.sleep(60)
                
                # Periodic status report
                status = await self._collect_system_status()
                uptime_hours = status["uptime_seconds"] / 3600
                
                print(f"\n⚡ System running autonomously for {uptime_hours:.1f} hours")
                print(f"   Components: {len(status['components'])} active")
                print(f"   Services: {len([s for s in status['services'].values() if s.get('running')])} running")
                print(f"   Health: {status['overall_health']}")
                
                if "autonomous_generator" in self.components:
                    gen_status = self.components["autonomous_generator"].get_status()
                    print(f"   Generated: {gen_status.get('completed_projects', 0)} projects")
                    print(f"   Active: {gen_status.get('active_generations', 0)} generations")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Autonomous mode stopping...")
            await self.graceful_shutdown()
            
    async def graceful_shutdown(self):
        """Gracefully shutdown all services"""
        self.is_running = False
        
        logging.info("🛑 GRACEFUL SHUTDOWN INITIATED")
        logging.info("-" * 50)
        
        # Stop autonomous generation
        if "autonomous_generator" in self.components:
            try:
                await self.components["autonomous_generator"].stop()
                logging.info("✅ Autonomous generation stopped")
            except Exception as e:
                logging.error(f"❌ Error stopping autonomous generation: {e}")
                
        # Stop services
        for name, service in self.services.items():
            try:
                if hasattr(service, 'close'):
                    service.close()
                    if hasattr(service, 'wait_closed'):
                        await service.wait_closed()
                elif hasattr(service, 'cancel'):
                    service.cancel()
                    
                logging.info(f"✅ {name} service stopped")
                
            except Exception as e:
                logging.error(f"❌ Error stopping {name}: {e}")
                
        # Cleanup components
        for name, component in self.components.items():
            try:
                if hasattr(component, 'cleanup'):
                    await component.cleanup()
                logging.info(f"✅ {name} component cleaned up")
            except Exception as e:
                logging.error(f"❌ Error cleaning up {name}: {e}")
                
        logging.info("-" * 50)
        logging.info("🔚 Shutdown completed")
        
    async def deploy_with_one_click(self):
        """Deploy the entire system with one click"""
        deployer = OneClickDeployer()
        
        print("\n🚀 ONE-CLICK DEPLOYMENT STARTING")
        print("=" * 50)
        
        result = await deployer.deploy()
        
        if result["success"]:
            print("\n✅ DEPLOYMENT SUCCESSFUL!")
            print("🌐 Access your system at: http://localhost:8080")
        else:
            print("\n❌ DEPLOYMENT FAILED")
            print("Check deployment.log for details")
            
        return result

async def handle_signal(signum, frame):
    """Handle shutdown signals"""
    logging.info(f"Received signal {signum}")
    # Signal will be handled by the main loop

def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"master_orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from some loggers
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)

async def main():
    """Main entry point"""
    setup_logging()
    
    # Handle command line arguments
    mode = "interactive"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
    # Setup signal handlers
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda s, f: asyncio.create_task(handle_signal(s, f)))
        
    # Create and run launcher
    launcher = MasterLauncher()
    
    try:
        if mode == "deploy":
            # One-click deployment mode
            await launcher.deploy_with_one_click()
            
        elif mode == "autonomous":
            # Full autonomous mode
            await launcher.initialize_all_systems()
            await launcher.start_all_services()
            await launcher.run_autonomous_mode()
            
        else:
            # Interactive mode (default)
            await launcher.initialize_all_systems()
            await launcher.start_all_services()
            await launcher.run_interactive_mode()
            
    except Exception as e:
        logging.error(f"Critical error: {e}")
        await launcher.graceful_shutdown()
    
if __name__ == "__main__":
    print("""
    
╔══════════════════════════════════════════════════════════════════════════════╗
║                           MASTER ORCHESTRATOR                                ║
║                    AI-Powered Development Platform                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🤖 Autonomous Code Generation    🌐 Real-time Frontend                      ║
║  🔧 Enterprise Agent Ecosystem    ⚡ Parallel LLM Processing                ║
║  🚀 One-Click Deployment         📊 Continuous Monitoring                   ║
║  🔄 Auto-Optimization           🛠️  Computer Control                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
  python master_launcher.py                 # Interactive mode
  python master_launcher.py autonomous      # Autonomous mode  
  python master_launcher.py deploy          # One-click deployment

""")
    
    asyncio.run(main())