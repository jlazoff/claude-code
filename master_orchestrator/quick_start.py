#!/usr/bin/env python3
"""
Quick Start - Simplified launcher for immediate functionality
"""

import asyncio
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

# Import only the core components we need immediately
from unified_config import SecureConfigManager
from parallel_llm_orchestrator import ParallelLLMOrchestrator
from frontend_orchestrator import FrontendOrchestrator

class QuickStarter:
    """Quick starter for immediate functionality"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.llm_orchestrator = ParallelLLMOrchestrator()
        self.frontend = FrontendOrchestrator()
        self.is_running = False
        
    async def initialize(self):
        """Initialize core components"""
        logging.info("🚀 Quick Start - Initializing Core Components")
        
        try:
            await self.config.initialize()
            logging.info("✅ Configuration initialized")
        except Exception as e:
            logging.warning(f"⚠️ Config initialization: {e}")
            
        try:
            await self.llm_orchestrator.initialize()
            logging.info("✅ LLM Orchestrator initialized")
        except Exception as e:
            logging.warning(f"⚠️ LLM initialization: {e}")
            
        try:
            await self.frontend.initialize()
            logging.info("✅ Frontend initialized")
        except Exception as e:
            logging.warning(f"⚠️ Frontend initialization: {e}")
            
        logging.info("🎯 Core components ready!")
        
    async def start_services(self):
        """Start essential services"""
        logging.info("🔧 Starting Essential Services")
        
        try:
            # Start frontend server
            frontend_runner = await self.frontend.start_server(
                host="0.0.0.0", 
                port=8080
            )
            logging.info("🌐 Frontend server started on http://0.0.0.0:8080")
            
            self.is_running = True
            
            # Demo generation
            await self._run_demo_generation()
            
        except Exception as e:
            logging.error(f"❌ Service startup error: {e}")
            
    async def _run_demo_generation(self):
        """Run a demo code generation"""
        logging.info("🤖 Running demo code generation...")
        
        demo_prompt = """
Create a simple FastAPI web service with the following features:
1. Health check endpoint
2. User registration endpoint
3. Basic authentication
4. Simple data storage
5. Error handling and logging

Make it production-ready with proper structure.
"""
        
        try:
            result = await self.llm_orchestrator.generate_code_parallel(
                demo_prompt, 
                "comprehensive"
            )
            
            if result.get("success"):
                # Save generated code
                demo_dir = Path("generated_demo")
                demo_dir.mkdir(exist_ok=True)
                
                with open(demo_dir / "demo_api.py", 'w') as f:
                    f.write(result["merged_code"])
                    
                logging.info(f"✅ Demo code generated and saved to {demo_dir}/demo_api.py")
                
                # Broadcast to frontend
                await self.frontend._broadcast_websocket({
                    "type": "demo_complete",
                    "message": "Demo code generation completed successfully!",
                    "file": str(demo_dir / "demo_api.py"),
                    "providers": result.get("source_providers", [])
                })
            else:
                logging.error(f"❌ Demo generation failed: {result.get('error')}")
                
        except Exception as e:
            logging.error(f"❌ Demo generation error: {e}")
            
    async def run(self):
        """Run the quick starter"""
        self.is_running = True
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MASTER ORCHESTRATOR - QUICK START                     ║
║                         Essential Features Running                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

🌐 Frontend: http://localhost:8080
🔧 Core LLM processing enabled
🤖 Demo code generation included

""")
        
        try:
            while self.is_running:
                await asyncio.sleep(30)
                logging.info("💓 System running - Access http://localhost:8080")
                
        except KeyboardInterrupt:
            logging.info("⏹️ Shutting down...")
            self.is_running = False

async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    starter = QuickStarter()
    
    try:
        await starter.initialize()
        await starter.start_services()
        await starter.run()
        
    except Exception as e:
        logging.error(f"Critical error: {e}")

if __name__ == "__main__":
    print("Starting Master Orchestrator Quick Start...")
    asyncio.run(main())