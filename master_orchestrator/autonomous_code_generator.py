#!/usr/bin/env python3
"""
Autonomous Code Generator
Self-generating system that creates comprehensive code and systems automatically
"""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import hashlib
import subprocess
import importlib.util

from unified_config import SecureConfigManager
from parallel_llm_orchestrator import ParallelLLMOrchestrator
from enterprise_agent_ecosystem import EnterpriseAgentEcosystem
from frontend_orchestrator import FrontendOrchestrator
from github_integration import AutomatedDevelopmentWorkflow
from conversation_project_initiator import ConversationProjectInitiator

class AutonomousCodeGenerator:
    """Self-generating code system with full automation"""
    
    def __init__(self):
        self.config = SecureConfigManager()
        self.llm_orchestrator = ParallelLLMOrchestrator()
        self.agent_ecosystem = EnterpriseAgentEcosystem()
        self.frontend = FrontendOrchestrator()
        self.dev_workflow = AutomatedDevelopmentWorkflow()
        self.project_initiator = ConversationProjectInitiator()
        
        self.generation_queue = asyncio.Queue()
        self.active_generations = {}
        self.completed_projects = []
        self.is_running = False
        
        # Pre-defined project templates for self-generation
        self.project_templates = self._get_project_templates()
        
    async def initialize(self):
        """Initialize all components for autonomous generation"""
        logging.info("🚀 Initializing Autonomous Code Generator...")
        
        # Initialize all systems
        await self.config.initialize()
        await self.llm_orchestrator.initialize()
        await self.agent_ecosystem.initialize()
        await self.frontend.initialize()
        await self.dev_workflow.initialize()
        await self.project_initiator.initialize()
        
        # Create necessary directories
        self._create_directory_structure()
        
        # Load any existing conversation data
        await self._load_conversation_projects()
        
        # Start autonomous generation
        await self._queue_initial_projects()
        
        logging.info("✅ Autonomous Code Generator initialized and ready")
        
    def _create_directory_structure(self):
        """Create comprehensive directory structure"""
        directories = [
            "generated_projects",
            "generated_apis",
            "generated_frontends",
            "generated_microservices",
            "generated_tools",
            "generated_dashboards",
            "generated_tests",
            "generated_docs",
            "templates",
            "configs",
            "deployments",
            "monitoring",
            "logs/generation",
            "backups"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def _get_project_templates(self) -> List[Dict[str, Any]]:
        """Get comprehensive project templates for self-generation"""
        return [
            {
                "name": "Enterprise API Gateway",
                "description": "Complete API gateway with authentication, rate limiting, and monitoring",
                "type": "microservice",
                "priority": "high",
                "technologies": ["FastAPI", "Redis", "JWT", "PostgreSQL", "Docker"],
                "features": [
                    "JWT authentication",
                    "Rate limiting",
                    "Request/response logging",
                    "Health checks",
                    "API documentation",
                    "Metrics collection",
                    "Circuit breaker pattern"
                ]
            },
            {
                "name": "Real-time Analytics Dashboard",
                "description": "Interactive dashboard with real-time data visualization",
                "type": "frontend",
                "priority": "high",
                "technologies": ["React", "D3.js", "WebSocket", "Material-UI"],
                "features": [
                    "Real-time charts",
                    "Interactive filters",
                    "Responsive design",
                    "Dark/light theme",
                    "Export capabilities",
                    "Alert notifications"
                ]
            },
            {
                "name": "AI-Powered Content Management System",
                "description": "CMS with AI content generation and optimization",
                "type": "fullstack",
                "priority": "high",
                "technologies": ["Next.js", "Node.js", "MongoDB", "OpenAI", "AWS S3"],
                "features": [
                    "AI content generation",
                    "SEO optimization",
                    "Media management",
                    "User roles",
                    "Publishing workflow",
                    "Analytics integration"
                ]
            },
            {
                "name": "Microservices Communication Hub",
                "description": "Service mesh for microservices communication and monitoring",
                "type": "infrastructure",
                "priority": "medium",
                "technologies": ["Go", "gRPC", "Kubernetes", "Prometheus", "Jaeger"],
                "features": [
                    "Service discovery",
                    "Load balancing",
                    "Circuit breaker",
                    "Distributed tracing",
                    "Metrics collection",
                    "Health monitoring"
                ]
            },
            {
                "name": "ML Model Deployment Platform",
                "description": "Platform for deploying and managing ML models",
                "type": "platform",
                "priority": "high",
                "technologies": ["Python", "FastAPI", "MLflow", "Docker", "Kubernetes"],
                "features": [
                    "Model versioning",
                    "A/B testing",
                    "Model monitoring",
                    "Auto-scaling",
                    "Feature store",
                    "Inference pipelines"
                ]
            },
            {
                "name": "Blockchain Integration Service",
                "description": "Service for blockchain interactions and smart contracts",
                "type": "service",
                "priority": "medium",
                "technologies": ["Python", "Web3.py", "Ethereum", "FastAPI", "Redis"],
                "features": [
                    "Smart contract interaction",
                    "Transaction monitoring",
                    "Wallet management",
                    "Gas optimization",
                    "Event listening",
                    "Security auditing"
                ]
            },
            {
                "name": "IoT Data Processing Pipeline",
                "description": "Real-time pipeline for IoT data processing and analytics",
                "type": "pipeline",
                "priority": "medium",
                "technologies": ["Apache Kafka", "Python", "InfluxDB", "Grafana", "Docker"],
                "features": [
                    "Stream processing",
                    "Data transformation",
                    "Anomaly detection",
                    "Real-time alerts",
                    "Time series storage",
                    "Visualization"
                ]
            },
            {
                "name": "Enterprise Chat Platform",
                "description": "Secure enterprise chat with file sharing and integrations",
                "type": "platform",
                "priority": "high",
                "technologies": ["React", "Socket.io", "Node.js", "MongoDB", "Redis"],
                "features": [
                    "Real-time messaging",
                    "File sharing",
                    "Video calls",
                    "Channel management",
                    "Integrations",
                    "Enterprise SSO"
                ]
            },
            {
                "name": "Automated Testing Framework",
                "description": "Comprehensive testing framework with AI-powered test generation",
                "type": "tool",
                "priority": "high",
                "technologies": ["Python", "Selenium", "Pytest", "Docker", "Jenkins"],
                "features": [
                    "Test generation",
                    "Cross-browser testing",
                    "API testing",
                    "Performance testing",
                    "Visual regression",
                    "CI/CD integration"
                ]
            },
            {
                "name": "Cloud Resource Optimizer",
                "description": "Tool for optimizing cloud resource usage and costs",
                "type": "tool",
                "priority": "medium",
                "technologies": ["Python", "AWS SDK", "Azure SDK", "GCP SDK", "Terraform"],
                "features": [
                    "Resource monitoring",
                    "Cost analysis",
                    "Usage optimization",
                    "Automated scaling",
                    "Compliance checking",
                    "Reporting"
                ]
            }
        ]
        
    async def _load_conversation_projects(self):
        """Load projects from conversation data if available"""
        conversation_files = [
            "conversations.json",
            "chatgpt_conversations.json",
            "project_ideas.json"
        ]
        
        for file_name in conversation_files:
            file_path = Path(file_name)
            if file_path.exists():
                try:
                    result = await self.project_initiator.process_conversations_file(str(file_path))
                    if result.get("success"):
                        logging.info(f"Loaded {result.get('total_projects_found', 0)} projects from {file_name}")
                except Exception as e:
                    logging.warning(f"Could not load {file_name}: {e}")
                    
    async def _queue_initial_projects(self):
        """Queue initial projects for generation"""
        # Add template projects to queue
        for template in self.project_templates:
            await self.generation_queue.put({
                "type": "template_project",
                "project": template,
                "priority": template.get("priority", "medium")
            })
            
        # Add dynamic project generation tasks
        dynamic_tasks = [
            {
                "type": "analyze_and_generate",
                "description": "Analyze current tech trends and generate relevant projects",
                "priority": "medium"
            },
            {
                "type": "optimize_existing",
                "description": "Optimize and enhance existing code",
                "priority": "low"
            },
            {
                "type": "create_utilities",
                "description": "Generate useful development utilities and tools",
                "priority": "medium"
            }
        ]
        
        for task in dynamic_tasks:
            await self.generation_queue.put(task)
            
        logging.info(f"Queued {self.generation_queue.qsize()} initial generation tasks")
        
    async def start_autonomous_generation(self):
        """Start autonomous code generation process"""
        if self.is_running:
            logging.warning("Autonomous generation is already running")
            return
            
        self.is_running = True
        logging.info("🎯 Starting autonomous code generation...")
        
        # Start multiple generation workers
        workers = []
        for i in range(3):  # 3 concurrent workers
            worker = asyncio.create_task(self._generation_worker(f"worker_{i}"))
            workers.append(worker)
            
        # Start monitoring and optimization
        monitor_task = asyncio.create_task(self._monitoring_loop())
        optimization_task = asyncio.create_task(self._optimization_loop())
        
        try:
            # Run until stopped
            await asyncio.gather(*workers, monitor_task, optimization_task)
        except Exception as e:
            logging.error(f"Autonomous generation error: {e}")
        finally:
            self.is_running = False
            
    async def _generation_worker(self, worker_id: str):
        """Individual generation worker"""
        logging.info(f"🔧 Generation worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get next generation task
                task = await asyncio.wait_for(
                    self.generation_queue.get(), 
                    timeout=10.0
                )
                
                # Process the task
                await self._process_generation_task(task, worker_id)
                
                # Mark task as done
                self.generation_queue.task_done()
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(5)
                
    async def _process_generation_task(self, task: Dict[str, Any], worker_id: str):
        """Process a specific generation task"""
        task_type = task.get("type")
        task_id = hashlib.md5(json.dumps(task, sort_keys=True).encode()).hexdigest()[:8]
        
        logging.info(f"🔨 Worker {worker_id} processing task: {task_type} ({task_id})")
        
        try:
            self.active_generations[task_id] = {
                "task": task,
                "worker_id": worker_id,
                "started_at": datetime.now().isoformat(),
                "status": "in_progress"
            }
            
            if task_type == "template_project":
                result = await self._generate_template_project(task["project"])
            elif task_type == "analyze_and_generate":
                result = await self._analyze_and_generate_projects()
            elif task_type == "optimize_existing":
                result = await self._optimize_existing_code()
            elif task_type == "create_utilities":
                result = await self._create_development_utilities()
            else:
                result = {"success": False, "error": f"Unknown task type: {task_type}"}
                
            # Update status
            self.active_generations[task_id]["status"] = "completed" if result.get("success") else "failed"
            self.active_generations[task_id]["result"] = result
            self.active_generations[task_id]["completed_at"] = datetime.now().isoformat()
            
            # Move to completed projects
            if result.get("success"):
                self.completed_projects.append({
                    "task_id": task_id,
                    "task": task,
                    "result": result,
                    "worker_id": worker_id,
                    "completed_at": datetime.now().isoformat()
                })
                
                # Commit to git if successful
                await self._commit_generated_code(task, result)
                
            logging.info(f"✅ Task {task_id} completed by worker {worker_id}")
            
        except Exception as e:
            logging.error(f"❌ Task {task_id} failed: {e}")
            self.active_generations[task_id]["status"] = "failed"
            self.active_generations[task_id]["error"] = str(e)
            
    async def _generate_template_project(self, project_template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete project from template"""
        project_name = project_template["name"]
        logging.info(f"Generating project: {project_name}")
        
        # Use agent ecosystem for comprehensive generation
        generation_task = {
            "type": "full_project_generation",
            "description": f"Generate complete {project_name}: {project_template['description']}",
            "prompt": f"""
Create a complete, production-ready implementation of: {project_name}

Description: {project_template['description']}
Type: {project_template['type']}
Technologies: {', '.join(project_template['technologies'])}

Features to implement:
{chr(10).join(f"- {feature}" for feature in project_template['features'])}

Requirements:
1. Complete working code with all features
2. Proper project structure and organization
3. Configuration files and environment setup
4. Comprehensive documentation
5. Unit and integration tests
6. Docker containerization
7. CI/CD pipeline configuration
8. Error handling and logging
9. Security best practices
10. Performance optimization

Generate a full project structure with all necessary files.
"""
        }
        
        # Execute with all available agents
        result = await self.agent_ecosystem.execute_task_parallel(generation_task)
        
        if result.get("success"):
            # Create project structure
            project_id = hashlib.md5(project_name.encode()).hexdigest()[:8]
            project_dir = Path("generated_projects") / project_id
            
            # Save generated code
            await self._save_project_code(project_dir, project_template, result)
            
            # Generate additional components
            await self._generate_project_components(project_dir, project_template)
            
            return {
                "success": True,
                "project_id": project_id,
                "project_dir": str(project_dir),
                "project_name": project_name,
                "generation_result": result
            }
        else:
            return result
            
    async def _save_project_code(self, project_dir: Path, template: Dict[str, Any], result: Dict[str, Any]):
        """Save generated project code to filesystem"""
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Main application code
        main_code = result.get("merged_result", {}).get("merged_content", "")
        if main_code:
            # Try to split into multiple files based on content
            files = await self._split_code_into_files(main_code, template)
            
            for file_name, file_content in files.items():
                file_path = project_dir / file_name
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(file_content)
                    
        # Create project metadata
        metadata = {
            "name": template["name"],
            "description": template["description"],
            "type": template["type"],
            "technologies": template["technologies"],
            "features": template["features"],
            "generated_at": datetime.now().isoformat(),
            "agent_frameworks": result.get("merged_result", {}).get("source_frameworks", [])
        }
        
        async with aiofiles.open(project_dir / "project_metadata.json", 'w') as f:
            await f.write(json.dumps(metadata, indent=2))
            
    async def _split_code_into_files(self, code: str, template: Dict[str, Any]) -> Dict[str, str]:
        """Split generated code into appropriate files"""
        files = {}
        
        # Use LLM to intelligently split code
        split_prompt = f"""
Analyze this generated code and split it into appropriate files for a {template['type']} project:

Code:
{code}

Project type: {template['type']}
Technologies: {', '.join(template['technologies'])}

Split the code into logical files with appropriate names and structure.
Return as JSON with filename: content mapping.
Include proper imports and dependencies in each file.
"""
        
        try:
            split_result = await self.llm_orchestrator.generate_code_parallel(
                split_prompt, "comprehensive"
            )
            
            if split_result.get("success"):
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', split_result["merged_code"], re.DOTALL)
                if json_match:
                    files_data = json.loads(json_match.group())
                    files.update(files_data)
                    
        except Exception as e:
            logging.warning(f"Could not split code intelligently: {e}")
            
        # Fallback: create basic file structure
        if not files:
            files = self._create_basic_file_structure(code, template)
            
        return files
        
    def _create_basic_file_structure(self, code: str, template: Dict[str, Any]) -> Dict[str, str]:
        """Create basic file structure as fallback"""
        files = {}
        
        # Main application file
        files["main.py"] = code
        
        # Requirements file
        tech_to_packages = {
            "FastAPI": "fastapi==0.104.1\nuvicorn==0.24.0",
            "React": "# React dependencies in package.json",
            "Django": "django==4.2.7",
            "Flask": "flask==2.3.3",
            "Next.js": "# Next.js dependencies in package.json",
            "PostgreSQL": "psycopg2-binary==2.9.7",
            "MongoDB": "pymongo==4.5.0",
            "Redis": "redis==5.0.1",
            "Docker": "# Docker configuration in Dockerfile"
        }
        
        requirements = []
        for tech in template.get("technologies", []):
            if tech in tech_to_packages:
                requirements.append(tech_to_packages[tech])
                
        files["requirements.txt"] = "\n".join(requirements)
        
        # README file
        files["README.md"] = f"""# {template['name']}

{template['description']}

## Features

{chr(10).join(f"- {feature}" for feature in template.get('features', []))}

## Technologies

{chr(10).join(f"- {tech}" for tech in template.get('technologies', []))}

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Generated automatically by Master Orchestrator.
"""

        # Basic configuration
        files["config.py"] = """
import os
from typing import Optional

class Config:
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', 'localhost')
    PORT = int(os.getenv('PORT', 8000))
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # Add more configuration as needed
"""

        return files
        
    async def _generate_project_components(self, project_dir: Path, template: Dict[str, Any]):
        """Generate additional project components"""
        # Generate tests
        await self._generate_project_tests(project_dir, template)
        
        # Generate Docker configuration
        await self._generate_docker_config(project_dir, template)
        
        # Generate CI/CD configuration
        await self._generate_cicd_config(project_dir, template)
        
        # Generate documentation
        await self._generate_project_docs(project_dir, template)
        
    async def _generate_project_tests(self, project_dir: Path, template: Dict[str, Any]):
        """Generate comprehensive tests for the project"""
        test_prompt = f"""
Generate comprehensive tests for the {template['name']} project:

Project type: {template['type']}
Technologies: {', '.join(template['technologies'])}
Features: {', '.join(template['features'])}

Create:
1. Unit tests for all components
2. Integration tests for API endpoints
3. End-to-end tests for main workflows
4. Performance tests
5. Security tests

Use appropriate testing frameworks for the technologies used.
"""
        
        test_result = await self.agent_ecosystem.execute_task_parallel({
            "type": "test_generation",
            "prompt": test_prompt
        })
        
        if test_result.get("success"):
            test_content = test_result.get("merged_result", {}).get("merged_content", "")
            if test_content:
                tests_dir = project_dir / "tests"
                tests_dir.mkdir(exist_ok=True)
                
                async with aiofiles.open(tests_dir / "test_main.py", 'w') as f:
                    await f.write(test_content)
                    
    async def _generate_docker_config(self, project_dir: Path, template: Dict[str, Any]):
        """Generate Docker configuration"""
        docker_prompt = f"""
Generate Docker configuration for {template['name']}:

Technologies: {', '.join(template['technologies'])}
Type: {template['type']}

Create:
1. Dockerfile for the application
2. docker-compose.yml for full stack
3. .dockerignore file
4. Multi-stage build if appropriate

Optimize for production use with security and performance best practices.
"""
        
        docker_result = await self.agent_ecosystem.execute_task_parallel({
            "type": "docker_configuration",
            "prompt": docker_prompt
        })
        
        if docker_result.get("success"):
            docker_content = docker_result.get("merged_result", {}).get("merged_content", "")
            
            # Parse and save Docker files
            # For now, create basic Dockerfile
            dockerfile_content = f"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
"""
            
            async with aiofiles.open(project_dir / "Dockerfile", 'w') as f:
                await f.write(dockerfile_content)
                
    async def _generate_cicd_config(self, project_dir: Path, template: Dict[str, Any]):
        """Generate CI/CD configuration"""
        # Create GitHub Actions workflow
        workflow_dir = project_dir / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_content = f"""
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        pytest tests/
        
    - name: Build Docker image
      run: |
        docker build -t {template['name'].lower().replace(' ', '-')} .
        
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to production
      run: |
        echo "Deployment would happen here"
"""
        
        async with aiofiles.open(workflow_dir / "ci.yml", 'w') as f:
            await f.write(workflow_content)
            
    async def _generate_project_docs(self, project_dir: Path, template: Dict[str, Any]):
        """Generate comprehensive project documentation"""
        docs_dir = project_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # API documentation
        if "API" in template["name"] or template["type"] in ["microservice", "service"]:
            api_docs = f"""# API Documentation

## {template['name']}

{template['description']}

### Endpoints

Generated API endpoints for {template['name']}.

### Authentication

Security implementation details.

### Examples

Usage examples and code samples.
"""
            
            async with aiofiles.open(docs_dir / "api.md", 'w') as f:
                await f.write(api_docs)
                
        # User guide
        user_guide = f"""# User Guide

## {template['name']}

{template['description']}

## Features

{chr(10).join(f"### {feature}" + chr(10) + f"Description of {feature} functionality." + chr(10) for feature in template.get('features', []))}

## Getting Started

Step-by-step guide to get started with {template['name']}.

## Configuration

Configuration options and environment variables.
"""
        
        async with aiofiles.open(docs_dir / "user_guide.md", 'w') as f:
            await f.write(user_guide)
            
    async def _analyze_and_generate_projects(self) -> Dict[str, Any]:
        """Analyze trends and generate relevant projects"""
        analysis_prompt = """
Analyze current technology trends and generate 3-5 innovative project ideas that would be valuable to implement:

Consider:
1. Emerging technologies (AI, blockchain, IoT, edge computing)
2. Market demands and business needs
3. Developer productivity tools
4. Automation and optimization opportunities
5. Integration and platform solutions

For each project idea, provide:
- Name and description
- Target use case and value proposition
- Technology stack
- Key features
- Implementation complexity
- Market potential

Focus on practical, implementable solutions that solve real problems.
"""
        
        result = await self.agent_ecosystem.execute_task_parallel({
            "type": "trend_analysis",
            "prompt": analysis_prompt
        })
        
        if result.get("success"):
            # Parse the response and queue new projects
            content = result.get("merged_result", {}).get("merged_content", "")
            
            # Queue generated projects for implementation
            # This would parse the content and create new project templates
            logging.info("Generated new project ideas from trend analysis")
            
        return result
        
    async def _optimize_existing_code(self) -> Dict[str, Any]:
        """Optimize existing code in the system"""
        # Find existing Python files to optimize
        python_files = list(Path(".").glob("**/*.py"))
        optimized_files = []
        
        for file_path in python_files[:5]:  # Limit to 5 files per run
            try:
                optimization_result = await self.llm_orchestrator.optimize_existing_code(str(file_path))
                if optimization_result.get("success"):
                    optimized_files.append({
                        "file": str(file_path),
                        "result": optimization_result
                    })
            except Exception as e:
                logging.warning(f"Could not optimize {file_path}: {e}")
                
        return {
            "success": True,
            "optimized_files": optimized_files,
            "total_files": len(optimized_files)
        }
        
    async def _create_development_utilities(self) -> Dict[str, Any]:
        """Create useful development utilities and tools"""
        utilities = [
            {
                "name": "Code Quality Checker",
                "description": "Tool to analyze code quality and suggest improvements",
                "type": "utility"
            },
            {
                "name": "API Performance Monitor",
                "description": "Monitor and analyze API performance metrics",
                "type": "monitoring"
            },
            {
                "name": "Database Migration Tool",
                "description": "Tool for managing database schema migrations",
                "type": "database"
            }
        ]
        
        created_utilities = []
        
        for utility in utilities:
            try:
                result = await self._generate_template_project(utility)
                if result.get("success"):
                    created_utilities.append(result)
            except Exception as e:
                logging.error(f"Failed to create utility {utility['name']}: {e}")
                
        return {
            "success": True,
            "utilities_created": len(created_utilities),
            "utilities": created_utilities
        }
        
    async def _commit_generated_code(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Commit generated code to git repository"""
        try:
            if result.get("success") and "project_dir" in result:
                project_dir = result["project_dir"]
                project_name = result.get("project_name", "Generated Project")
                
                # Use the automated development workflow
                commit_result = await self.dev_workflow.github_integrator.stage_and_commit_changes(
                    commit_message=f"feat: Generate {project_name}\n\nAuto-generated complete project implementation",
                    files=[project_dir]
                )
                
                if commit_result.get("success"):
                    logging.info(f"Successfully committed {project_name} to git")
                else:
                    logging.warning(f"Failed to commit {project_name}: {commit_result.get('error')}")
                    
        except Exception as e:
            logging.error(f"Error committing generated code: {e}")
            
    async def _monitoring_loop(self):
        """Monitor generation progress and system health"""
        while self.is_running:
            try:
                # Log current status
                active_count = len(self.active_generations)
                completed_count = len(self.completed_projects)
                queue_size = self.generation_queue.qsize()
                
                logging.info(f"📊 Generation Status: {active_count} active, {completed_count} completed, {queue_size} queued")
                
                # Broadcast status to frontend
                await self.frontend._broadcast_websocket({
                    "type": "generation_status",
                    "data": {
                        "active_generations": active_count,
                        "completed_projects": completed_count,
                        "queue_size": queue_size,
                        "is_running": self.is_running,
                        "timestamp": datetime.now().isoformat()
                    }
                })
                
                # Clean up old active generations
                current_time = datetime.now()
                stale_generations = []
                
                for task_id, generation in self.active_generations.items():
                    started_at = datetime.fromisoformat(generation["started_at"])
                    if (current_time - started_at).total_seconds() > 3600:  # 1 hour timeout
                        stale_generations.append(task_id)
                        
                for task_id in stale_generations:
                    del self.active_generations[task_id]
                    logging.warning(f"Removed stale generation: {task_id}")
                    
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
                
    async def _optimization_loop(self):
        """Continuous optimization and improvement"""
        while self.is_running:
            try:
                # Queue optimization tasks periodically
                await asyncio.sleep(1800)  # Every 30 minutes
                
                if self.generation_queue.qsize() < 5:  # Only if queue is not too full
                    await self.generation_queue.put({
                        "type": "optimize_existing",
                        "description": "Periodic code optimization",
                        "priority": "low"
                    })
                    
                    await self.generation_queue.put({
                        "type": "analyze_and_generate",
                        "description": "Periodic trend analysis",
                        "priority": "low"
                    })
                    
            except Exception as e:
                logging.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)
                
    def get_status(self) -> Dict[str, Any]:
        """Get current generation status"""
        return {
            "is_running": self.is_running,
            "active_generations": len(self.active_generations),
            "completed_projects": len(self.completed_projects),
            "queue_size": self.generation_queue.qsize(),
            "recent_completions": self.completed_projects[-5:] if self.completed_projects else [],
            "system_health": "healthy" if self.is_running else "stopped"
        }
        
    async def stop(self):
        """Stop autonomous generation"""
        self.is_running = False
        logging.info("🛑 Autonomous code generation stopped")

async def main():
    """Main entry point for autonomous code generation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and initialize autonomous generator
    generator = AutonomousCodeGenerator()
    await generator.initialize()
    
    # Start autonomous generation
    try:
        await generator.start_autonomous_generation()
    except KeyboardInterrupt:
        logging.info("Received interrupt signal")
        await generator.stop()

if __name__ == "__main__":
    # Import aiofiles
    try:
        import aiofiles
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiofiles"])
        import aiofiles
    
    asyncio.run(main())