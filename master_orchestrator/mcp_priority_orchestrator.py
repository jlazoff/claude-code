#!/usr/bin/env python3
"""
MCP Priority Orchestrator - Comprehensive MCP Server Integration System
Prioritizes MCP implementations, tests completeness, and integrates optimally
"""

import asyncio
import json
import yaml
import logging
import subprocess
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiofiles
import aiohttp
from datetime import datetime
import hashlib
import sqlite3
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    path: str
    implementation_type: str  # "native", "wrapper", "proxy"
    completeness_score: float
    capabilities: List[str]
    dependencies: List[str]
    test_results: Dict[str, Any]
    integration_methods: List[str]
    deployment_configs: Dict[str, Any]
    performance_metrics: Dict[str, float]
    automation_options: List[str]

@dataclass
class TaskApproach:
    """Different approaches to complete a task"""
    name: str
    description: str
    tools_required: List[str]
    estimated_time: float
    success_probability: float
    automation_level: float
    test_coverage: float
    integration_complexity: float

@dataclass
class ExecutionResult:
    """Result of executing a task approach"""
    approach_name: str
    success: bool
    execution_time: float
    output: Any
    metrics: Dict[str, float]
    lessons_learned: List[str]
    optimization_suggestions: List[str]

class MCPPriorityOrchestrator:
    """
    Comprehensive MCP Server Integration and Multi-Approach Task Execution System
    """
    
    def __init__(self):
        self.base_dir = Path("foundation_data")
        self.github_dir = Path("/Users/jlazoff/Documents/GitHub")
        self.mcp_servers: Dict[str, MCPServerConfig] = {}
        self.task_approaches: Dict[str, List[TaskApproach]] = {}
        self.execution_history: List[ExecutionResult] = []
        self.automation_registry: Dict[str, callable] = {}
        self.learning_database = None
        
        # Initialize directories
        self.base_dir.mkdir(exist_ok=True)
        (self.base_dir / "mcp_configs").mkdir(exist_ok=True)
        (self.base_dir / "test_results").mkdir(exist_ok=True)
        (self.base_dir / "automation_scripts").mkdir(exist_ok=True)
        (self.base_dir / "deployment_configs").mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize the orchestrator system"""
        logger.info("🚀 Initializing MCP Priority Orchestrator...")
        
        # Initialize database
        await self._init_learning_database()
        
        # Discover all MCP servers
        await self._discover_mcp_servers()
        
        # Load existing configurations
        await self._load_existing_configs()
        
        # Register automation functions
        await self._register_automation_functions()
        
        logger.info("✅ MCP Priority Orchestrator initialized")
        
    async def _init_learning_database(self):
        """Initialize SQLite database for learning and optimization"""
        db_path = self.base_dir / "learning.db"
        
        self.learning_database = sqlite3.connect(str(db_path))
        cursor = self.learning_database.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mcp_servers (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                path TEXT,
                implementation_type TEXT,
                completeness_score REAL,
                capabilities TEXT,
                last_tested TIMESTAMP,
                integration_status TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_executions (
                id INTEGER PRIMARY KEY,
                task_name TEXT,
                approach_name TEXT,
                success BOOLEAN,
                execution_time REAL,
                metrics TEXT,
                lessons_learned TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_insights (
                id INTEGER PRIMARY KEY,
                component TEXT,
                insight TEXT,
                impact_score REAL,
                applied BOOLEAN DEFAULT FALSE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.learning_database.commit()
        
    async def _discover_mcp_servers(self):
        """Discover all MCP server implementations in GitHub repositories"""
        logger.info("🔍 Discovering MCP servers...")
        
        mcp_patterns = [
            "**/mcp-*",
            "**/*mcp*",
            "**/server.py",
            "**/mcp_server.py",
            "**/__main__.py"
        ]
        
        discovered_servers = {}
        
        for pattern in mcp_patterns:
            for repo_path in self.github_dir.rglob("*"):
                if repo_path.is_dir() and (repo_path / ".git").exists():
                    await self._analyze_repo_for_mcp(repo_path, discovered_servers)
        
        # Test and score each discovered server
        for server_name, server_info in discovered_servers.items():
            mcp_config = await self._test_mcp_completeness(server_info)
            if mcp_config:
                self.mcp_servers[server_name] = mcp_config
                
        logger.info(f"✅ Discovered {len(self.mcp_servers)} MCP servers")
        
    async def _analyze_repo_for_mcp(self, repo_path: Path, discovered_servers: Dict):
        """Analyze a repository for MCP server implementations"""
        try:
            # Check for MCP-related files
            mcp_files = []
            
            # Look for obvious MCP files
            for pattern in ["*mcp*", "*server*", "*main*"]:
                mcp_files.extend(list(repo_path.rglob(pattern + ".py")))
                mcp_files.extend(list(repo_path.rglob(pattern + ".js")))
                mcp_files.extend(list(repo_path.rglob(pattern + ".ts")))
            
            # Check package.json or pyproject.toml for MCP dependencies
            package_files = list(repo_path.glob("package.json")) + list(repo_path.glob("pyproject.toml")) + list(repo_path.glob("requirements.txt"))
            
            has_mcp_deps = False
            for package_file in package_files:
                try:
                    content = package_file.read_text()
                    if "mcp" in content.lower() or "model-context-protocol" in content.lower():
                        has_mcp_deps = True
                        break
                except:
                    continue
            
            if mcp_files or has_mcp_deps:
                server_info = {
                    "name": repo_path.name,
                    "path": str(repo_path),
                    "mcp_files": [str(f) for f in mcp_files],
                    "has_mcp_deps": has_mcp_deps,
                    "repo_type": self._detect_repo_type(repo_path)
                }
                discovered_servers[repo_path.name] = server_info
                
        except Exception as e:
            logger.debug(f"Error analyzing {repo_path}: {e}")
            
    def _detect_repo_type(self, repo_path: Path) -> str:
        """Detect the type of repository (Python, Node.js, etc.)"""
        if (repo_path / "package.json").exists():
            return "nodejs"
        elif (repo_path / "pyproject.toml").exists() or (repo_path / "setup.py").exists():
            return "python"
        elif (repo_path / "Cargo.toml").exists():
            return "rust"
        elif (repo_path / "go.mod").exists():
            return "go"
        else:
            return "unknown"
            
    async def _test_mcp_completeness(self, server_info: Dict) -> Optional[MCPServerConfig]:
        """Test MCP server completeness and generate configuration"""
        try:
            repo_path = Path(server_info["path"])
            
            # Read README and documentation
            capabilities = await self._extract_capabilities(repo_path)
            
            # Test basic functionality
            test_results = await self._run_mcp_tests(repo_path, server_info["repo_type"])
            
            # Calculate completeness score
            completeness_score = self._calculate_completeness_score(
                capabilities, test_results, server_info
            )
            
            # Generate deployment configurations
            deployment_configs = await self._generate_deployment_configs(repo_path, server_info)
            
            # Generate automation options
            automation_options = await self._generate_automation_options(repo_path, capabilities)
            
            config = MCPServerConfig(
                name=server_info["name"],
                path=server_info["path"],
                implementation_type=self._determine_implementation_type(server_info),
                completeness_score=completeness_score,
                capabilities=capabilities,
                dependencies=await self._extract_dependencies(repo_path),
                test_results=test_results,
                integration_methods=await self._determine_integration_methods(repo_path, capabilities),
                deployment_configs=deployment_configs,
                performance_metrics=await self._measure_performance(repo_path),
                automation_options=automation_options
            )
            
            return config
            
        except Exception as e:
            logger.error(f"Error testing MCP server {server_info['name']}: {e}")
            return None
            
    async def _extract_capabilities(self, repo_path: Path) -> List[str]:
        """Extract capabilities from README and source code"""
        capabilities = []
        
        # Read README files
        readme_files = list(repo_path.glob("README*")) + list(repo_path.glob("readme*"))
        for readme in readme_files:
            try:
                content = readme.read_text().lower()
                
                # Common MCP capabilities
                capability_keywords = {
                    "file operations": ["file", "read", "write", "filesystem"],
                    "web browsing": ["browser", "web", "http", "requests"],
                    "database": ["database", "sql", "mongodb", "postgres"],
                    "memory": ["memory", "store", "recall", "remember"],
                    "tools": ["tools", "functions", "execute"],
                    "search": ["search", "find", "query"],
                    "git": ["git", "repository", "commit"],
                    "ai": ["ai", "llm", "model", "inference"]
                }
                
                for capability, keywords in capability_keywords.items():
                    if any(keyword in content for keyword in keywords):
                        capabilities.append(capability)
                        
            except:
                continue
                
        return list(set(capabilities))
        
    async def _run_mcp_tests(self, repo_path: Path, repo_type: str) -> Dict[str, Any]:
        """Run tests on MCP server implementation"""
        test_results = {
            "basic_import": False,
            "server_start": False,
            "capabilities_list": False,
            "tool_execution": False,
            "error_handling": False,
            "performance": {}
        }
        
        try:
            if repo_type == "python":
                # Test Python MCP server
                test_results.update(await self._test_python_mcp(repo_path))
            elif repo_type == "nodejs":
                # Test Node.js MCP server
                test_results.update(await self._test_nodejs_mcp(repo_path))
                
        except Exception as e:
            logger.error(f"Error running tests for {repo_path}: {e}")
            
        return test_results
        
    async def _test_python_mcp(self, repo_path: Path) -> Dict[str, Any]:
        """Test Python MCP server"""
        test_results = {}
        
        try:
            # Try to import and run basic tests
            original_cwd = os.getcwd()
            os.chdir(repo_path)
            
            # Look for main entry point
            entry_points = ["server.py", "main.py", "__main__.py", "app.py"]
            entry_point = None
            
            for ep in entry_points:
                if (repo_path / ep).exists():
                    entry_point = ep
                    break
                    
            if entry_point:
                # Try basic syntax check
                result = subprocess.run([
                    sys.executable, "-m", "py_compile", entry_point
                ], capture_output=True, text=True, timeout=30)
                
                test_results["syntax_check"] = result.returncode == 0
                
                # Try to run with --help or --version
                for flag in ["--help", "--version", "-h"]:
                    try:
                        result = subprocess.run([
                            sys.executable, entry_point, flag
                        ], capture_output=True, text=True, timeout=10)
                        
                        if result.returncode == 0:
                            test_results["help_available"] = True
                            break
                    except:
                        continue
                        
        except Exception as e:
            logger.debug(f"Python test error: {e}")
        finally:
            os.chdir(original_cwd)
            
        return test_results
        
    async def _test_nodejs_mcp(self, repo_path: Path) -> Dict[str, Any]:
        """Test Node.js MCP server"""
        test_results = {}
        
        try:
            original_cwd = os.getcwd()
            os.chdir(repo_path)
            
            # Check package.json
            package_json = repo_path / "package.json"
            if package_json.exists():
                package_data = json.loads(package_json.read_text())
                
                # Check if npm install works
                result = subprocess.run([
                    "npm", "install"
                ], capture_output=True, text=True, timeout=120)
                
                test_results["npm_install"] = result.returncode == 0
                
                # Try to run npm start or main script
                if "scripts" in package_data:
                    for script in ["start", "dev", "serve"]:
                        if script in package_data["scripts"]:
                            try:
                                result = subprocess.run([
                                    "npm", "run", script, "--", "--help"
                                ], capture_output=True, text=True, timeout=10)
                                test_results[f"script_{script}"] = result.returncode == 0
                            except:
                                continue
                                
        except Exception as e:
            logger.debug(f"Node.js test error: {e}")
        finally:
            os.chdir(original_cwd)
            
        return test_results
        
    def _calculate_completeness_score(self, capabilities: List[str], test_results: Dict, server_info: Dict) -> float:
        """Calculate completeness score for MCP server"""
        score = 0.0
        
        # Base score for having MCP-related code
        if server_info.get("has_mcp_deps", False):
            score += 0.2
            
        # Score for capabilities
        score += min(len(capabilities) * 0.1, 0.3)
        
        # Score for test results
        passing_tests = sum(1 for result in test_results.values() if result is True)
        total_tests = len([v for v in test_results.values() if isinstance(v, bool)])
        if total_tests > 0:
            score += (passing_tests / total_tests) * 0.3
            
        # Score for documentation
        repo_path = Path(server_info["path"])
        if list(repo_path.glob("README*")):
            score += 0.1
            
        # Score for proper structure
        if server_info["repo_type"] != "unknown":
            score += 0.1
            
        return min(score, 1.0)
        
    async def _generate_deployment_configs(self, repo_path: Path, server_info: Dict) -> Dict[str, Any]:
        """Generate deployment configurations for different environments"""
        configs = {
            "local": {},
            "docker": {},
            "kubernetes": {},
            "cloud": {}
        }
        
        repo_type = server_info["repo_type"]
        
        # Local deployment
        if repo_type == "python":
            configs["local"] = {
                "command": f"cd {repo_path} && python server.py",
                "requirements": "pip install -r requirements.txt",
                "port": 8000
            }
        elif repo_type == "nodejs":
            configs["local"] = {
                "command": f"cd {repo_path} && npm start",
                "requirements": "npm install",
                "port": 3000
            }
            
        # Docker deployment
        dockerfile_content = self._generate_dockerfile(repo_type, repo_path)
        configs["docker"] = {
            "dockerfile": dockerfile_content,
            "build_command": f"docker build -t mcp-{repo_path.name} .",
            "run_command": f"docker run -p 8000:8000 mcp-{repo_path.name}"
        }
        
        # Kubernetes deployment
        configs["kubernetes"] = self._generate_k8s_config(repo_path.name, repo_type)
        
        return configs
        
    def _generate_dockerfile(self, repo_type: str, repo_path: Path) -> str:
        """Generate Dockerfile for MCP server"""
        if repo_type == "python":
            return f"""
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "server.py"]
"""
        elif repo_type == "nodejs":
            return f"""
FROM node:18-slim

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
EXPOSE 3000

CMD ["npm", "start"]
"""
        else:
            return "# Generic Dockerfile - needs customization"
            
    def _generate_k8s_config(self, name: str, repo_type: str) -> Dict[str, Any]:
        """Generate Kubernetes configuration"""
        port = 8000 if repo_type == "python" else 3000
        
        return {
            "deployment": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": f"mcp-{name}"},
                "spec": {
                    "replicas": 1,
                    "selector": {"matchLabels": {"app": f"mcp-{name}"}},
                    "template": {
                        "metadata": {"labels": {"app": f"mcp-{name}"}},
                        "spec": {
                            "containers": [{
                                "name": f"mcp-{name}",
                                "image": f"mcp-{name}:latest",
                                "ports": [{"containerPort": port}]
                            }]
                        }
                    }
                }
            },
            "service": {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": f"mcp-{name}-service"},
                "spec": {
                    "selector": {"app": f"mcp-{name}"},
                    "ports": [{
                        "protocol": "TCP",
                        "port": 80,
                        "targetPort": port
                    }]
                }
            }
        }
        
    async def _generate_automation_options(self, repo_path: Path, capabilities: List[str]) -> List[str]:
        """Generate automation options for the MCP server"""
        options = []
        
        # Basic automation options
        options.extend([
            "auto_start_on_boot",
            "auto_restart_on_failure",
            "health_check_monitoring",
            "log_rotation",
            "performance_monitoring"
        ])
        
        # Capability-specific automation
        if "file operations" in capabilities:
            options.extend([
                "auto_backup_monitoring",
                "file_change_notifications",
                "directory_cleanup"
            ])
            
        if "database" in capabilities:
            options.extend([
                "auto_database_backup",
                "connection_pool_monitoring",
                "query_performance_tracking"
            ])
            
        if "web browsing" in capabilities:
            options.extend([
                "session_management",
                "cookie_cleanup",
                "browser_process_monitoring"
            ])
            
        return options
        
    async def execute_task_with_multiple_approaches(self, task_name: str, task_description: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task using multiple approaches in parallel and analyze results
        """
        logger.info(f"🔄 Executing task: {task_name}")
        
        # Generate multiple approaches for the task
        approaches = await self._generate_task_approaches(task_name, task_description, **kwargs)
        
        # Execute approaches in parallel
        execution_results = await self._execute_approaches_parallel(approaches, **kwargs)
        
        # Analyze results and select best approach
        analysis = await self._analyze_execution_results(execution_results)
        
        # Learn from results
        await self._learn_from_execution(task_name, execution_results, analysis)
        
        # Generate automation options for future use
        automation_options = await self._generate_task_automation(task_name, analysis)
        
        return {
            "task_name": task_name,
            "approaches_tried": len(approaches),
            "execution_results": execution_results,
            "analysis": analysis,
            "automation_options": automation_options,
            "recommendation": analysis.get("best_approach"),
            "next_steps": analysis.get("optimization_suggestions", [])
        }
        
    async def _generate_task_approaches(self, task_name: str, task_description: str, **kwargs) -> List[TaskApproach]:
        """Generate multiple approaches to complete a task"""
        approaches = []
        
        # MCP-based approach
        relevant_mcps = [
            mcp for mcp in self.mcp_servers.values()
            if any(cap in task_description.lower() for cap in mcp.capabilities)
        ]
        
        if relevant_mcps:
            approaches.append(TaskApproach(
                name="mcp_based",
                description=f"Use MCP servers: {[mcp.name for mcp in relevant_mcps[:3]]}",
                tools_required=[mcp.name for mcp in relevant_mcps[:3]],
                estimated_time=5.0,
                success_probability=0.8,
                automation_level=0.9,
                test_coverage=0.7,
                integration_complexity=0.3
            ))
            
        # Direct implementation approach
        approaches.append(TaskApproach(
            name="direct_implementation",
            description="Direct implementation using available libraries",
            tools_required=["python", "standard_libraries"],
            estimated_time=15.0,
            success_probability=0.9,
            automation_level=0.6,
            test_coverage=0.8,
            integration_complexity=0.5
        ))
        
        # Hybrid approach
        if len(relevant_mcps) > 0:
            approaches.append(TaskApproach(
                name="hybrid",
                description="Combine MCP servers with custom implementation",
                tools_required=[relevant_mcps[0].name, "custom_code"],
                estimated_time=10.0,
                success_probability=0.85,
                automation_level=0.8,
                test_coverage=0.9,
                integration_complexity=0.4
            ))
            
        # External service approach
        approaches.append(TaskApproach(
            name="external_service",
            description="Use external APIs and services",
            tools_required=["http_client", "api_keys"],
            estimated_time=8.0,
            success_probability=0.7,
            automation_level=0.9,
            test_coverage=0.6,
            integration_complexity=0.6
        ))
        
        return approaches
        
    async def _execute_approaches_parallel(self, approaches: List[TaskApproach], **kwargs) -> List[ExecutionResult]:
        """Execute multiple approaches in parallel"""
        results = []
        
        async def execute_approach(approach: TaskApproach) -> ExecutionResult:
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Simulate approach execution (replace with actual implementation)
                if approach.name == "mcp_based":
                    result = await self._execute_mcp_approach(approach, **kwargs)
                elif approach.name == "direct_implementation":
                    result = await self._execute_direct_approach(approach, **kwargs)
                elif approach.name == "hybrid":
                    result = await self._execute_hybrid_approach(approach, **kwargs)
                else:
                    result = await self._execute_external_approach(approach, **kwargs)
                    
                execution_time = asyncio.get_event_loop().time() - start_time
                
                return ExecutionResult(
                    approach_name=approach.name,
                    success=result.get("success", False),
                    execution_time=execution_time,
                    output=result.get("output"),
                    metrics=result.get("metrics", {}),
                    lessons_learned=result.get("lessons_learned", []),
                    optimization_suggestions=result.get("optimization_suggestions", [])
                )
                
            except Exception as e:
                execution_time = asyncio.get_event_loop().time() - start_time
                return ExecutionResult(
                    approach_name=approach.name,
                    success=False,
                    execution_time=execution_time,
                    output=None,
                    metrics={"error": str(e)},
                    lessons_learned=[f"Failed with error: {e}"],
                    optimization_suggestions=["Add better error handling", "Validate inputs"]
                )
                
        # Execute all approaches concurrently
        tasks = [execute_approach(approach) for approach in approaches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [r for r in results if isinstance(r, ExecutionResult)]
        
    async def _execute_mcp_approach(self, approach: TaskApproach, **kwargs) -> Dict[str, Any]:
        """Execute approach using MCP servers"""
        # This would implement actual MCP server execution
        return {
            "success": True,
            "output": "MCP-based execution completed",
            "metrics": {"latency": 0.5, "accuracy": 0.9},
            "lessons_learned": ["MCP servers provide good abstraction"],
            "optimization_suggestions": ["Cache MCP responses", "Parallel MCP calls"]
        }
        
    async def _execute_direct_approach(self, approach: TaskApproach, **kwargs) -> Dict[str, Any]:
        """Execute direct implementation approach"""
        return {
            "success": True,
            "output": "Direct implementation completed",
            "metrics": {"latency": 1.2, "accuracy": 0.95},
            "lessons_learned": ["Direct control allows optimization"],
            "optimization_suggestions": ["Add async processing", "Implement caching"]
        }
        
    async def _execute_hybrid_approach(self, approach: TaskApproach, **kwargs) -> Dict[str, Any]:
        """Execute hybrid approach"""
        return {
            "success": True,
            "output": "Hybrid execution completed",
            "metrics": {"latency": 0.8, "accuracy": 0.92},
            "lessons_learned": ["Best of both worlds approach"],
            "optimization_suggestions": ["Balance MCP vs custom code", "Optimize handoffs"]
        }
        
    async def _execute_external_approach(self, approach: TaskApproach, **kwargs) -> Dict[str, Any]:
        """Execute external service approach"""
        return {
            "success": True,
            "output": "External service execution completed",
            "metrics": {"latency": 2.0, "accuracy": 0.85},
            "lessons_learned": ["External dependencies add latency"],
            "optimization_suggestions": ["Add retry logic", "Implement fallbacks"]
        }
        
    async def _analyze_execution_results(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Analyze execution results and determine best approach"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                "best_approach": None,
                "success_rate": 0.0,
                "average_time": sum(r.execution_time for r in results) / len(results),
                "optimization_suggestions": ["All approaches failed - investigate requirements"]
            }
            
        # Score each approach
        scored_results = []
        for result in successful_results:
            score = (
                (1.0 if result.success else 0.0) * 0.4 +
                (1.0 / max(result.execution_time, 0.1)) * 0.3 +
                result.metrics.get("accuracy", 0.8) * 0.3
            )
            scored_results.append((score, result))
            
        # Sort by score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        best_result = scored_results[0][1]
        
        return {
            "best_approach": best_result.approach_name,
            "success_rate": len(successful_results) / len(results),
            "average_time": sum(r.execution_time for r in successful_results) / len(successful_results),
            "best_score": scored_results[0][0],
            "optimization_suggestions": self._aggregate_optimization_suggestions(results),
            "lessons_learned": self._aggregate_lessons_learned(results)
        }
        
    def _aggregate_optimization_suggestions(self, results: List[ExecutionResult]) -> List[str]:
        """Aggregate optimization suggestions from all results"""
        all_suggestions = []
        for result in results:
            all_suggestions.extend(result.optimization_suggestions)
        return list(set(all_suggestions))
        
    def _aggregate_lessons_learned(self, results: List[ExecutionResult]) -> List[str]:
        """Aggregate lessons learned from all results"""
        all_lessons = []
        for result in results:
            all_lessons.extend(result.lessons_learned)
        return list(set(all_lessons))
        
    async def _learn_from_execution(self, task_name: str, results: List[ExecutionResult], analysis: Dict[str, Any]):
        """Learn from execution results and update knowledge base"""
        cursor = self.learning_database.cursor()
        
        for result in results:
            cursor.execute("""
                INSERT INTO task_executions 
                (task_name, approach_name, success, execution_time, metrics, lessons_learned)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                task_name,
                result.approach_name,
                result.success,
                result.execution_time,
                json.dumps(result.metrics),
                json.dumps(result.lessons_learned)
            ))
            
        # Store optimization insights
        for suggestion in analysis.get("optimization_suggestions", []):
            cursor.execute("""
                INSERT INTO optimization_insights (component, insight, impact_score)
                VALUES (?, ?, ?)
            """, (task_name, suggestion, analysis.get("best_score", 0.5)))
            
        self.learning_database.commit()
        
    async def _generate_task_automation(self, task_name: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate automation options for the task"""
        automation_options = []
        
        best_approach = analysis.get("best_approach")
        if best_approach:
            automation_options.extend([
                f"auto_execute_using_{best_approach}",
                f"schedule_periodic_{task_name}",
                f"trigger_on_condition_{task_name}",
                f"batch_process_{task_name}",
                f"monitor_and_retry_{task_name}"
            ])
            
        # Add specific automation based on success rate
        success_rate = analysis.get("success_rate", 0.0)
        if success_rate > 0.8:
            automation_options.append(f"fully_automate_{task_name}")
        elif success_rate > 0.6:
            automation_options.append(f"semi_automate_with_confirmation_{task_name}")
        else:
            automation_options.append(f"manual_execution_with_assistance_{task_name}")
            
        return automation_options
        
    async def provide_automation_options(self, user_request: str) -> Dict[str, Any]:
        """
        For any user request, provide automation options
        """
        automation_options = {
            "immediate_automation": [],
            "scheduled_automation": [],
            "conditional_automation": [],
            "batch_automation": [],
            "learning_automation": []
        }
        
        # Analyze the request to determine automation possibilities
        request_lower = user_request.lower()
        
        # Immediate automation options
        if "test" in request_lower:
            automation_options["immediate_automation"].extend([
                "run_all_tests_now",
                "run_specific_test_suite",
                "run_parallel_tests"
            ])
            
        if "deploy" in request_lower:
            automation_options["immediate_automation"].extend([
                "deploy_to_staging",
                "deploy_to_production",
                "rollback_deployment"
            ])
            
        # Scheduled automation
        automation_options["scheduled_automation"].extend([
            "schedule_daily_execution",
            "schedule_weekly_analysis",
            "schedule_monthly_optimization"
        ])
        
        # Conditional automation
        automation_options["conditional_automation"].extend([
            "trigger_on_file_change",
            "trigger_on_performance_threshold",
            "trigger_on_error_rate",
            "trigger_on_user_activity"
        ])
        
        # Batch automation
        automation_options["batch_automation"].extend([
            "batch_process_all_repos",
            "batch_update_dependencies",
            "batch_run_security_scans"
        ])
        
        # Learning automation
        automation_options["learning_automation"].extend([
            "auto_optimize_based_on_usage",
            "auto_suggest_improvements",
            "auto_detect_patterns",
            "auto_update_configurations"
        ])
        
        return {
            "request": user_request,
            "automation_options": automation_options,
            "recommended_immediate": automation_options["immediate_automation"][:3],
            "setup_instructions": self._generate_automation_setup_instructions(automation_options)
        }
        
    def _generate_automation_setup_instructions(self, automation_options: Dict[str, List[str]]) -> Dict[str, str]:
        """Generate setup instructions for automation options"""
        instructions = {}
        
        for category, options in automation_options.items():
            if options:
                if category == "immediate_automation":
                    instructions[category] = "Run: ./setup_immediate_automation.sh"
                elif category == "scheduled_automation":
                    instructions[category] = "Setup cron jobs or systemd timers"
                elif category == "conditional_automation":
                    instructions[category] = "Configure file watchers and monitoring"
                elif category == "batch_automation":
                    instructions[category] = "Setup batch processing queues"
                elif category == "learning_automation":
                    instructions[category] = "Enable ML-based optimization"
                    
        return instructions
        
    async def continuous_optimization_loop(self):
        """Run continuous optimization and learning loop"""
        logger.info("🔄 Starting continuous optimization loop...")
        
        while True:
            try:
                # Scan for new MCP servers
                await self._discover_mcp_servers()
                
                # Optimize existing configurations
                await self._optimize_mcp_configurations()
                
                # Learn from execution history
                await self._analyze_performance_patterns()
                
                # Update automation recommendations
                await self._update_automation_recommendations()
                
                # Generate optimization reports
                await self._generate_optimization_report()
                
                logger.info("✅ Optimization cycle completed")
                
                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
                
    async def _optimize_mcp_configurations(self):
        """Optimize MCP server configurations based on performance data"""
        for mcp_name, mcp_config in self.mcp_servers.items():
            # Analyze performance metrics
            if mcp_config.performance_metrics:
                # Optimize based on metrics
                optimizations = await self._generate_optimization_recommendations(mcp_config)
                
                # Apply optimizations if beneficial
                if optimizations:
                    await self._apply_optimizations(mcp_name, optimizations)
                    
    async def _generate_optimization_recommendations(self, mcp_config: MCPServerConfig) -> List[str]:
        """Generate optimization recommendations for MCP server"""
        recommendations = []
        
        metrics = mcp_config.performance_metrics
        
        # Memory optimization
        if metrics.get("memory_usage", 0) > 0.8:
            recommendations.append("reduce_memory_footprint")
            
        # CPU optimization
        if metrics.get("cpu_usage", 0) > 0.7:
            recommendations.append("optimize_cpu_usage")
            
        # Response time optimization
        if metrics.get("avg_response_time", 0) > 1.0:
            recommendations.append("improve_response_time")
            
        return recommendations
        
    async def save_configurations(self):
        """Save all configurations to disk"""
        config_file = self.base_dir / "mcp_configurations.yaml"
        
        configs_data = {
            "mcp_servers": {
                name: asdict(config) for name, config in self.mcp_servers.items()
            },
            "automation_registry": list(self.automation_registry.keys()),
            "last_updated": datetime.now().isoformat()
        }
        
        async with aiofiles.open(config_file, 'w') as f:
            await f.write(yaml.dump(configs_data, default_flow_style=False))
            
        logger.info(f"✅ Configurations saved to {config_file}")

async def main():
    """Main execution function"""
    orchestrator = MCPPriorityOrchestrator()
    await orchestrator.initialize()
    
    # Example: Execute a task with multiple approaches
    result = await orchestrator.execute_task_with_multiple_approaches(
        "file_analysis",
        "Analyze GitHub repositories for MCP implementations"
    )
    
    print("Task Execution Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Example: Get automation options for a request
    automation = await orchestrator.provide_automation_options(
        "Test all MCP servers and deploy the best ones"
    )
    
    print("\nAutomation Options:")
    print(json.dumps(automation, indent=2))
    
    # Save configurations
    await orchestrator.save_configurations()
    
    # Start continuous optimization (comment out for single run)
    # await orchestrator.continuous_optimization_loop()

if __name__ == "__main__":
    asyncio.run(main())