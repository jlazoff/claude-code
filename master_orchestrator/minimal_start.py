#!/usr/bin/env python3
"""
Minimal Start - Get the system running immediately
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from aiohttp import web
import aiohttp_cors

class MinimalOrchestrator:
    """Minimal orchestrator for immediate functionality"""
    
    def __init__(self):
        self.app = web.Application()
        self.websocket_clients = set()
        self.generated_projects = []
        
    async def initialize(self):
        """Initialize minimal system"""
        # Setup routes
        self.app.router.add_get('/', self.serve_main_page)
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_post('/api/generate', self.generate_code)
        self.app.router.add_get('/api/status', self.get_status)
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
            
        logging.info("✅ Minimal orchestrator initialized")
        
    async def serve_main_page(self, request):
        """Serve main interface"""
        html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Master Orchestrator - Running</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 40px; }
        .status { background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .generate-form { background: #2d2d2d; padding: 20px; border-radius: 8px; }
        .btn { background: #007acc; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #005fa3; }
        textarea { width: 100%; height: 150px; background: #1a1a1a; color: #fff; border: 1px solid #444; padding: 10px; }
        .output { background: #1a1a1a; padding: 15px; margin-top: 20px; border-radius: 4px; white-space: pre-wrap; font-family: monospace; }
        .success { color: #4caf50; }
        .error { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Master Orchestrator</h1>
            <h2>AI-Powered Development Platform</h2>
            <p>Status: <span class="success">✅ Running Successfully</span></p>
        </div>
        
        <div class="status">
            <h3>📊 System Status</h3>
            <p>✅ Core System: Online</p>
            <p>✅ API Endpoints: Active</p>
            <p>✅ Code Generation: Ready</p>
            <p>📈 Projects Generated: <span id="project-count">0</span></p>
        </div>
        
        <div class="generate-form">
            <h3>🤖 Generate Code</h3>
            <textarea id="prompt" placeholder="Describe what you want to build...">Create a FastAPI web service with user authentication, database models, and API endpoints for a task management system.</textarea>
            <br><br>
            <button class="btn" onclick="generateCode()">Generate Code</button>
            <div id="output" class="output" style="display: none;"></div>
        </div>
    </div>
    
    <script>
        async function generateCode() {
            const prompt = document.getElementById('prompt').value;
            const output = document.getElementById('output');
            
            output.style.display = 'block';
            output.innerHTML = 'Generating code...';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    output.innerHTML = 'Generated Code:\\n\\n' + result.code;
                    document.getElementById('project-count').textContent = parseInt(document.getElementById('project-count').textContent) + 1;
                } else {
                    output.innerHTML = 'Error: ' + (result.error || 'Unknown error');
                }
            } catch (error) {
                output.innerHTML = 'Error: ' + error.message;
            }
        }
        
        // Update status periodically
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                document.getElementById('project-count').textContent = status.projects_generated || 0;
            } catch (error) {
                console.log('Status update error:', error);
            }
        }, 5000);
    </script>
</body>
</html>
        '''
        return web.Response(text=html, content_type='text/html')
        
    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "minimal_orchestrator": "online",
                "api_endpoints": "active"
            }
        })
        
    async def generate_code(self, request):
        """Generate code endpoint"""
        try:
            data = await request.json()
            prompt = data.get('prompt', '')
            
            if not prompt:
                return web.json_response({"success": False, "error": "No prompt provided"})
                
            # Simple code generation (mock for now)
            generated_code = await self._mock_generate_code(prompt)
            
            # Save project
            project = {
                "id": len(self.generated_projects) + 1,
                "prompt": prompt,
                "code": generated_code,
                "timestamp": datetime.now().isoformat()
            }
            self.generated_projects.append(project)
            
            # Save to file
            project_dir = Path("generated_projects") / f"project_{project['id']}"
            project_dir.mkdir(parents=True, exist_ok=True)
            
            with open(project_dir / "main.py", 'w') as f:
                f.write(generated_code)
                
            with open(project_dir / "README.md", 'w') as f:
                f.write(f"# Generated Project {project['id']}\\n\\n{prompt}\\n\\nGenerated: {project['timestamp']}")
                
            return web.json_response({
                "success": True,
                "code": generated_code,
                "project_id": project["id"],
                "project_path": str(project_dir)
            })
            
        except Exception as e:
            logging.error(f"Code generation error: {e}")
            return web.json_response({"success": False, "error": str(e)})
            
    async def _mock_generate_code(self, prompt: str) -> str:
        """Mock code generation for demonstration"""
        # This is a simple template-based generator for demo purposes
        
        if "fastapi" in prompt.lower() or "api" in prompt.lower():
            return '''
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime

app = FastAPI(title="Generated API", description="Auto-generated FastAPI application")

# Data models
class User(BaseModel):
    id: Optional[int] = None
    username: str
    email: str
    created_at: Optional[datetime] = None

class Task(BaseModel):
    id: Optional[int] = None
    title: str
    description: str
    completed: bool = False
    user_id: int
    created_at: Optional[datetime] = None

# In-memory storage (use database in production)
users_db = []
tasks_db = []

# Endpoints
@app.get("/")
async def root():
    return {"message": "Generated API is running!", "timestamp": datetime.now()}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/users/", response_model=User)
async def create_user(user: User):
    user.id = len(users_db) + 1
    user.created_at = datetime.now()
    users_db.append(user)
    return user

@app.get("/users/", response_model=List[User])
async def get_users():
    return users_db

@app.post("/tasks/", response_model=Task)
async def create_task(task: Task):
    task.id = len(tasks_db) + 1
    task.created_at = datetime.now()
    tasks_db.append(task)
    return task

@app.get("/tasks/", response_model=List[Task])
async def get_tasks():
    return tasks_db

@app.put("/tasks/{task_id}")
async def update_task(task_id: int, task: Task):
    for i, existing_task in enumerate(tasks_db):
        if existing_task.id == task_id:
            task.id = task_id
            tasks_db[i] = task
            return task
    raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        elif "react" in prompt.lower() or "frontend" in prompt.lower():
            return '''
import React, { useState, useEffect } from 'react';

const App = () => {
    const [tasks, setTasks] = useState([]);
    const [newTask, setNewTask] = useState('');

    useEffect(() => {
        // Fetch tasks from API
        fetchTasks();
    }, []);

    const fetchTasks = async () => {
        try {
            const response = await fetch('/api/tasks');
            const data = await response.json();
            setTasks(data);
        } catch (error) {
            console.error('Error fetching tasks:', error);
        }
    };

    const addTask = async () => {
        if (!newTask.trim()) return;
        
        try {
            const response = await fetch('/api/tasks', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTask, description: '', completed: false })
            });
            
            if (response.ok) {
                setNewTask('');
                fetchTasks();
            }
        } catch (error) {
            console.error('Error adding task:', error);
        }
    };

    return (
        <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
            <h1>Generated Task Manager</h1>
            
            <div style={{ marginBottom: '20px' }}>
                <input
                    type="text"
                    value={newTask}
                    onChange={(e) => setNewTask(e.target.value)}
                    placeholder="Enter new task..."
                    style={{ padding: '10px', width: '300px', marginRight: '10px' }}
                />
                <button onClick={addTask} style={{ padding: '10px 20px' }}>
                    Add Task
                </button>
            </div>
            
            <div>
                <h2>Tasks ({tasks.length})</h2>
                {tasks.map(task => (
                    <div key={task.id} style={{ 
                        padding: '10px', 
                        border: '1px solid #ddd', 
                        marginBottom: '10px',
                        backgroundColor: task.completed ? '#f0f8f0' : '#fff'
                    }}>
                        <h3>{task.title}</h3>
                        <p>{task.description}</p>
                        <small>Created: {new Date(task.created_at).toLocaleString()}</small>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default App;
'''
        else:
            return f'''
# Generated Python Application
# Based on: {prompt}

import logging
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeneratedApplication:
    """Auto-generated application class"""
    
    def __init__(self):
        self.created_at = datetime.now()
        self.data = []
        logger.info("Generated application initialized")
    
    def process_data(self, input_data: Any) -> Dict[str, Any]:
        """Process input data and return results"""
        try:
            result = {{
                "processed_at": datetime.now().isoformat(),
                "input": str(input_data),
                "status": "success"
            }}
            
            self.data.append(result)
            logger.info(f"Processed data: {{input_data}}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing data: {{e}}")
            return {{"error": str(e), "status": "failed"}}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get application statistics"""
        return {{
            "created_at": self.created_at.isoformat(),
            "total_processed": len(self.data),
            "last_activity": self.data[-1]["processed_at"] if self.data else None
        }}

def main():
    """Main application entry point"""
    app = GeneratedApplication()
    
    # Example usage
    print("Generated Application Running!")
    print(f"Created: {{app.created_at}}")
    
    # Process some sample data
    sample_data = ["sample1", "sample2", "sample3"]
    for item in sample_data:
        result = app.process_data(item)
        print(f"Processed: {{result}}")
    
    print(f"Final stats: {{app.get_stats()}}")

if __name__ == "__main__":
    main()
'''
        
    async def get_status(self, request):
        """Get system status"""
        return web.json_response({
            "status": "running",
            "projects_generated": len(self.generated_projects),
            "uptime": "online",
            "timestamp": datetime.now().isoformat()
        })
        
    async def start_server(self, host="0.0.0.0", port=8080):
        """Start the server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logging.info(f"🌐 Server started on http://{host}:{port}")
        return runner

async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        MASTER ORCHESTRATOR - MINIMAL                         ║
║                              Quick Start Mode                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Starting essential services...
""")
    
    orchestrator = MinimalOrchestrator()
    await orchestrator.initialize()
    
    runner = await orchestrator.start_server()
    
    print(f"""
✅ System Started Successfully!

🌐 Access your platform: http://localhost:8080
📊 Health check: http://localhost:8080/health
🤖 Code generation: Available through web interface

Features available:
- Real-time code generation
- Project creation and management
- Web-based interface
- API endpoints

Press Ctrl+C to stop the server
""")
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(3600)  # Sleep for 1 hour
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())