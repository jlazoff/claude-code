#!/usr/bin/env python3
"""
Continuous Repository Scanner and Tool Catalog Optimizer
Scans GitHub repositories, creates a dynamic tool catalog, and provides optimization recommendations
"""

import os
import json
import yaml
import time
import logging
import asyncio
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import subprocess
import re
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repo_scanner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RepositoryMetadata:
    """Metadata for a repository"""
    path: str
    name: str
    description: str = ""
    primary_language: str = ""
    languages: Dict[str, int] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    last_updated: str = ""
    size_bytes: int = 0
    file_count: int = 0
    has_dockerfile: bool = False
    has_tests: bool = False
    has_ci: bool = False
    readme_content: str = ""
    license: str = ""
    topics: List[str] = field(default_factory=list)
    
@dataclass
class ToolProfile:
    """Profile for a tool/repository"""
    metadata: RepositoryMetadata
    category: str = ""
    primary_function: str = ""
    value_proposition: str = ""
    integration_points: List[str] = field(default_factory=list)
    dependencies_on: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    containerization_status: str = ""
    deployment_strategy: str = ""
    optimization_recommendations: List[str] = field(default_factory=list)
    synergies: List[str] = field(default_factory=list)
    redundancies: List[str] = field(default_factory=list)
    architecture_role: str = ""
    last_scanned: str = ""
    content_hash: str = ""

class RepositoryWatcher(FileSystemEventHandler):
    """Watches for changes in repository directories"""
    
    def __init__(self, scanner):
        self.scanner = scanner
        self.debounce_timers = {}
        
    def on_any_event(self, event):
        """Handle file system events"""
        if event.is_directory:
            return
            
        # Get repository path
        repo_path = self._get_repo_path(event.src_path)
        if not repo_path:
            return
            
        # Debounce events (wait 5 seconds before processing)
        if repo_path in self.debounce_timers:
            self.debounce_timers[repo_path].cancel()
            
        timer = threading.Timer(5.0, self.scanner.scan_repository, args=[repo_path])
        self.debounce_timers[repo_path] = timer
        timer.start()
        
    def _get_repo_path(self, file_path: str) -> Optional[str]:
        """Extract repository path from file path"""
        path = Path(file_path)
        while path.parent != path:
            if (path / '.git').exists():
                return str(path)
            path = path.parent
        return None

class ContinuousRepoScanner:
    """Main scanner class for continuous repository monitoring and analysis"""
    
    def __init__(self, base_path: str, db_path: str = "tool_catalog.db"):
        self.base_path = Path(base_path)
        self.db_path = db_path
        self.tools_catalog: Dict[str, ToolProfile] = {}
        self.scan_interval = 3600  # 1 hour
        self.running = False
        self._init_database()
        self._load_catalog()
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tools (
                path TEXT PRIMARY KEY,
                name TEXT,
                data TEXT,
                last_scanned TIMESTAMP,
                content_hash TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                total_repos INTEGER,
                new_repos INTEGER,
                updated_repos INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_catalog(self):
        """Load existing catalog from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT path, data FROM tools')
        for row in cursor.fetchall():
            path, data = row
            tool_data = json.loads(data)
            self.tools_catalog[path] = self._dict_to_tool_profile(tool_data)
            
        conn.close()
        logger.info(f"Loaded {len(self.tools_catalog)} tools from database")
        
    def _dict_to_tool_profile(self, data: dict) -> ToolProfile:
        """Convert dictionary to ToolProfile object"""
        metadata_data = data.pop('metadata', {})
        metadata = RepositoryMetadata(**metadata_data)
        return ToolProfile(metadata=metadata, **data)
        
    def _save_tool(self, path: str, tool: ToolProfile):
        """Save tool to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tool_dict = asdict(tool)
        data = json.dumps(tool_dict)
        
        cursor.execute('''
            INSERT OR REPLACE INTO tools (path, name, data, last_scanned, content_hash)
            VALUES (?, ?, ?, ?, ?)
        ''', (path, tool.metadata.name, data, tool.last_scanned, tool.content_hash))
        
        conn.commit()
        conn.close()
        
    def scan_repository(self, repo_path: str) -> Optional[ToolProfile]:
        """Scan a single repository and extract metadata"""
        try:
            logger.info(f"Scanning repository: {repo_path}")
            
            # Check if path exists and is a git repository
            if not os.path.exists(repo_path) or not os.path.exists(os.path.join(repo_path, '.git')):
                return None
                
            # Calculate content hash to detect changes
            content_hash = self._calculate_repo_hash(repo_path)
            
            # Check if already scanned and unchanged
            if repo_path in self.tools_catalog:
                existing_tool = self.tools_catalog[repo_path]
                if existing_tool.content_hash == content_hash:
                    logger.info(f"Repository {repo_path} unchanged, skipping")
                    return existing_tool
                    
            # Extract metadata
            metadata = self._extract_metadata(repo_path)
            
            # Create tool profile
            tool = ToolProfile(
                metadata=metadata,
                last_scanned=datetime.now().isoformat(),
                content_hash=content_hash
            )
            
            # Analyze tool characteristics
            self._analyze_tool(tool, repo_path)
            
            # Save to catalog
            self.tools_catalog[repo_path] = tool
            self._save_tool(repo_path, tool)
            
            return tool
            
        except Exception as e:
            logger.error(f"Error scanning repository {repo_path}: {e}")
            return None
            
    def _calculate_repo_hash(self, repo_path: str) -> str:
        """Calculate hash of repository contents for change detection"""
        hasher = hashlib.md5()
        
        # Hash important files
        important_files = ['README.md', 'setup.py', 'pyproject.toml', 'package.json', 
                          'Dockerfile', 'docker-compose.yml', 'requirements.txt']
        
        for file in important_files:
            file_path = os.path.join(repo_path, file)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
                    
        return hasher.hexdigest()
        
    def _extract_metadata(self, repo_path: str) -> RepositoryMetadata:
        """Extract repository metadata"""
        metadata = RepositoryMetadata(
            path=repo_path,
            name=os.path.basename(repo_path),
            last_updated=datetime.now().isoformat()
        )
        
        # Read README
        readme_paths = ['README.md', 'README.rst', 'README.txt', 'README']
        for readme in readme_paths:
            readme_path = os.path.join(repo_path, readme)
            if os.path.exists(readme_path):
                with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                    metadata.readme_content = f.read()[:5000]  # First 5000 chars
                break
                
        # Extract description from README
        if metadata.readme_content:
            lines = metadata.readme_content.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and i < 10:
                    metadata.description = line.strip()[:200]
                    break
                    
        # Check for various files
        metadata.has_dockerfile = os.path.exists(os.path.join(repo_path, 'Dockerfile'))
        metadata.has_tests = any(os.path.exists(os.path.join(repo_path, d)) 
                                for d in ['tests', 'test', '__tests__'])
        metadata.has_ci = any(os.path.exists(os.path.join(repo_path, f)) 
                             for f in ['.github/workflows', '.travis.yml', '.circleci'])
        
        # Count files and calculate size
        metadata.file_count = 0
        metadata.size_bytes = 0
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            metadata.file_count += len(files)
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    metadata.size_bytes += os.path.getsize(file_path)
                except:
                    pass
                    
        # Detect languages
        metadata.languages = self._detect_languages(repo_path)
        if metadata.languages:
            metadata.primary_language = max(metadata.languages, key=metadata.languages.get)
            
        # Extract dependencies
        metadata.dependencies = self._extract_dependencies(repo_path)
        
        # Extract license
        license_files = ['LICENSE', 'LICENSE.md', 'LICENSE.txt', 'COPYING']
        for license_file in license_files:
            license_path = os.path.join(repo_path, license_file)
            if os.path.exists(license_path):
                metadata.license = license_file
                break
                
        return metadata
        
    def _detect_languages(self, repo_path: str) -> Dict[str, int]:
        """Detect programming languages used in repository"""
        language_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.jl': 'Julia',
            '.m': 'MATLAB',
            '.sh': 'Shell',
            '.ps1': 'PowerShell'
        }
        
        language_counts = defaultdict(int)
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in language_extensions:
                    language_counts[language_extensions[ext]] += 1
                    
        return dict(language_counts)
        
    def _extract_dependencies(self, repo_path: str) -> List[str]:
        """Extract project dependencies"""
        dependencies = []
        
        # Python dependencies
        requirements_path = os.path.join(repo_path, 'requirements.txt')
        if os.path.exists(requirements_path):
            with open(requirements_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep = line.split('=')[0].split('>')[0].split('<')[0].strip()
                        if dep:
                            dependencies.append(f"python:{dep}")
                            
        # Python pyproject.toml
        pyproject_path = os.path.join(repo_path, 'pyproject.toml')
        if os.path.exists(pyproject_path):
            try:
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    # Simple regex to extract dependencies
                    deps = re.findall(r'"([^"]+)"', content)
                    for dep in deps:
                        if '=' not in dep and '>' not in dep and '<' not in dep:
                            dependencies.append(f"python:{dep}")
            except:
                pass
                
        # Node.js dependencies
        package_path = os.path.join(repo_path, 'package.json')
        if os.path.exists(package_path):
            try:
                with open(package_path, 'r') as f:
                    package_data = json.load(f)
                    for dep in package_data.get('dependencies', {}):
                        dependencies.append(f"npm:{dep}")
                    for dep in package_data.get('devDependencies', {}):
                        dependencies.append(f"npm:{dep}")
            except:
                pass
                
        return list(set(dependencies))
        
    def _analyze_tool(self, tool: ToolProfile, repo_path: str):
        """Analyze tool characteristics and categorize"""
        
        # Categorize based on keywords and structure
        categories = self._categorize_tool(tool)
        if categories:
            tool.category = categories[0]  # Primary category
            tool.topics = categories
            
        # Determine primary function
        tool.primary_function = self._determine_primary_function(tool)
        
        # Extract value proposition
        tool.value_proposition = self._extract_value_proposition(tool)
        
        # Identify integration points
        tool.integration_points = self._identify_integration_points(tool)
        
        # Analyze containerization
        tool.containerization_status = self._analyze_containerization(tool, repo_path)
        
        # Determine deployment strategy
        tool.deployment_strategy = self._determine_deployment_strategy(tool)
        
        # Analyze performance characteristics
        tool.performance_metrics = self._analyze_performance(tool)
        
        # Generate optimization recommendations
        tool.optimization_recommendations = self._generate_recommendations(tool)
        
        # Determine architecture role
        tool.architecture_role = self._determine_architecture_role(tool)
        
    def _categorize_tool(self, tool: ToolProfile) -> List[str]:
        """Categorize tool based on content analysis"""
        categories = []
        
        content = tool.metadata.readme_content.lower()
        name = tool.metadata.name.lower()
        
        # AI/ML categories
        if any(keyword in content or keyword in name for keyword in 
               ['agent', 'ai', 'llm', 'gpt', 'machine learning', 'neural', 'model']):
            categories.append('AI/ML')
            
        # Infrastructure
        if any(keyword in content or keyword in name for keyword in 
               ['docker', 'kubernetes', 'container', 'orchestrat', 'deploy', 'infrastructure']):
            categories.append('Infrastructure')
            
        # Development tools
        if any(keyword in content or keyword in name for keyword in 
               ['develop', 'ide', 'editor', 'debug', 'test', 'build']):
            categories.append('Development')
            
        # Data processing
        if any(keyword in content or keyword in name for keyword in 
               ['data', 'etl', 'pipeline', 'stream', 'batch', 'analytics']):
            categories.append('Data Processing')
            
        # API/Web services
        if any(keyword in content or keyword in name for keyword in 
               ['api', 'rest', 'graphql', 'server', 'web', 'http']):
            categories.append('Web Services')
            
        # Security
        if any(keyword in content or keyword in name for keyword in 
               ['security', 'auth', 'encrypt', 'secure', 'vulnerab']):
            categories.append('Security')
            
        # Monitoring
        if any(keyword in content or keyword in name for keyword in 
               ['monitor', 'observ', 'metric', 'log', 'trace', 'alert']):
            categories.append('Monitoring')
            
        # Communication
        if any(keyword in content or keyword in name for keyword in 
               ['chat', 'message', 'communication', 'slack', 'discord']):
            categories.append('Communication')
            
        return categories
        
    def _determine_primary_function(self, tool: ToolProfile) -> str:
        """Determine the primary function of the tool"""
        content = tool.metadata.readme_content.lower()
        name = tool.metadata.name.lower()
        
        # Look for function indicators
        if 'agent' in name or 'agent' in content[:500]:
            if 'autonomous' in content:
                return "Autonomous AI Agent"
            elif 'multi' in content:
                return "Multi-Agent System"
            else:
                return "AI Agent Framework"
                
        elif 'orchestrat' in name or 'orchestrat' in content[:500]:
            return "System Orchestrator"
            
        elif 'api' in name or 'server' in name:
            return "API Service"
            
        elif 'monitor' in name:
            return "Monitoring System"
            
        elif 'data' in name:
            return "Data Processing"
            
        elif 'test' in name:
            return "Testing Framework"
            
        else:
            # Extract from first meaningful line of README
            lines = tool.metadata.readme_content.split('\n')
            for line in lines[1:10]:  # Skip title
                if line.strip() and not line.startswith('#'):
                    return line.strip()[:100]
                    
        return "General Purpose Tool"
        
    def _extract_value_proposition(self, tool: ToolProfile) -> str:
        """Extract the unique value proposition"""
        # Look for value indicators in README
        content = tool.metadata.readme_content
        
        # Common patterns
        patterns = [
            r'(?:allows?|enables?|provides?|helps?)\s+(?:you\s+)?(.{20,100})',
            r'(?:designed|built|created)\s+(?:to|for)\s+(.{20,100})',
            r'(?:solution|tool|framework)\s+(?:for|that)\s+(.{20,100})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
                
        return tool.metadata.description
        
    def _identify_integration_points(self, tool: ToolProfile) -> List[str]:
        """Identify potential integration points"""
        integration_points = []
        
        # Check for common integration patterns
        if tool.metadata.has_dockerfile:
            integration_points.append("Docker containerization")
            
        if any('api' in dep or 'rest' in dep for dep in tool.metadata.dependencies):
            integration_points.append("REST API")
            
        if any('grpc' in dep for dep in tool.metadata.dependencies):
            integration_points.append("gRPC")
            
        if any('kafka' in dep or 'rabbitmq' in dep for dep in tool.metadata.dependencies):
            integration_points.append("Message Queue")
            
        if any('postgres' in dep or 'mysql' in dep or 'mongo' in dep 
               for dep in tool.metadata.dependencies):
            integration_points.append("Database")
            
        # Check for specific frameworks
        if 'fastapi' in tool.metadata.dependencies:
            integration_points.append("FastAPI endpoints")
            
        if 'flask' in tool.metadata.dependencies or 'django' in tool.metadata.dependencies:
            integration_points.append("Web framework")
            
        return integration_points
        
    def _analyze_containerization(self, tool: ToolProfile, repo_path: str) -> str:
        """Analyze containerization status"""
        if tool.metadata.has_dockerfile:
            # Check Dockerfile quality
            dockerfile_path = os.path.join(repo_path, 'Dockerfile')
            with open(dockerfile_path, 'r') as f:
                content = f.read()
                
            if 'multi-stage' in content:
                return "Multi-stage Docker build"
            elif 'FROM' in content:
                return "Basic Dockerfile present"
        
        compose_path = os.path.join(repo_path, 'docker-compose.yml')
        if os.path.exists(compose_path):
            return "Docker Compose configured"
            
        return "No containerization"
        
    def _determine_deployment_strategy(self, tool: ToolProfile) -> str:
        """Determine recommended deployment strategy"""
        # Based on tool characteristics
        if 'kubernetes' in tool.metadata.readme_content.lower():
            return "Kubernetes deployment"
            
        elif tool.metadata.has_dockerfile:
            if tool.category == "Web Services":
                return "Container with load balancer"
            else:
                return "Standalone container"
                
        elif tool.metadata.primary_language == "Python":
            if any('flask' in dep or 'fastapi' in dep or 'django' in dep 
                   for dep in tool.metadata.dependencies):
                return "WSGI/ASGI deployment"
            else:
                return "Python virtual environment"
                
        elif tool.metadata.primary_language == "JavaScript":
            return "Node.js deployment"
            
        return "Traditional deployment"
        
    def _analyze_performance(self, tool: ToolProfile) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        metrics = {
            'scalability': 'unknown',
            'resource_usage': 'unknown',
            'startup_time': 'unknown',
            'concurrency': 'unknown'
        }
        
        # Analyze based on dependencies and structure
        if any('async' in dep for dep in tool.metadata.dependencies):
            metrics['concurrency'] = 'async/await support'
            
        if tool.metadata.has_dockerfile:
            metrics['scalability'] = 'horizontally scalable'
            
        if tool.metadata.size_bytes < 10 * 1024 * 1024:  # < 10MB
            metrics['resource_usage'] = 'lightweight'
        elif tool.metadata.size_bytes < 100 * 1024 * 1024:  # < 100MB
            metrics['resource_usage'] = 'moderate'
        else:
            metrics['resource_usage'] = 'heavy'
            
        return metrics
        
    def _generate_recommendations(self, tool: ToolProfile) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Containerization recommendations
        if not tool.metadata.has_dockerfile:
            recommendations.append("Add Dockerfile for containerization")
            
        # Testing recommendations
        if not tool.metadata.has_tests:
            recommendations.append("Add unit tests for reliability")
            
        # CI/CD recommendations
        if not tool.metadata.has_ci:
            recommendations.append("Implement CI/CD pipeline")
            
        # Documentation recommendations
        if len(tool.metadata.readme_content) < 500:
            recommendations.append("Enhance documentation")
            
        # Performance recommendations
        if tool.metadata.primary_language == "Python" and \
           not any('async' in dep for dep in tool.metadata.dependencies):
            recommendations.append("Consider async support for better concurrency")
            
        # Security recommendations
        if not any('auth' in dep for dep in tool.metadata.dependencies) and \
           tool.category == "Web Services":
            recommendations.append("Implement authentication/authorization")
            
        return recommendations
        
    def _determine_architecture_role(self, tool: ToolProfile) -> str:
        """Determine the role in overall architecture"""
        name = tool.metadata.name.lower()
        function = tool.primary_function.lower()
        
        if 'orchestrat' in name or 'orchestrat' in function:
            return "System Coordinator"
        elif 'agent' in name or 'agent' in function:
            return "Autonomous Executor"
        elif 'api' in name or 'server' in name:
            return "Service Provider"
        elif 'monitor' in name:
            return "System Observer"
        elif 'data' in name:
            return "Data Processor"
        elif 'frontend' in name or 'ui' in name:
            return "User Interface"
        elif 'auth' in name:
            return "Security Layer"
        else:
            return "Supporting Component"
            
    def identify_synergies(self):
        """Identify synergies between tools"""
        for path1, tool1 in self.tools_catalog.items():
            synergies = []
            for path2, tool2 in self.tools_catalog.items():
                if path1 == path2:
                    continue
                    
                # Check for complementary functions
                if tool1.category == "AI/ML" and tool2.category == "Data Processing":
                    synergies.append(f"Data pipeline for {tool2.metadata.name}")
                    
                elif tool1.category == "Web Services" and tool2.category == "Security":
                    synergies.append(f"Security layer with {tool2.metadata.name}")
                    
                # Check for shared dependencies
                shared_deps = set(tool1.metadata.dependencies) & set(tool2.metadata.dependencies)
                if len(shared_deps) > 3:
                    synergies.append(f"Shared stack with {tool2.metadata.name}")
                    
            tool1.synergies = synergies[:5]  # Top 5 synergies
            
    def identify_redundancies(self):
        """Identify redundant tools"""
        # Group by category and function
        category_tools = defaultdict(list)
        
        for path, tool in self.tools_catalog.items():
            key = f"{tool.category}:{tool.primary_function}"
            category_tools[key].append((path, tool))
            
        # Find redundancies
        for key, tools in category_tools.items():
            if len(tools) > 1:
                for path, tool in tools:
                    others = [t[1].metadata.name for t in tools if t[0] != path]
                    tool.redundancies = [f"Similar to {name}" for name in others]
                    
    def generate_catalog_report(self) -> str:
        """Generate comprehensive catalog report"""
        report = "# Tool Catalog Report\n\n"
        report += f"Generated at: {datetime.now().isoformat()}\n"
        report += f"Total tools: {len(self.tools_catalog)}\n\n"
        
        # Category breakdown
        report += "## Category Breakdown\n\n"
        category_counts = defaultdict(int)
        for tool in self.tools_catalog.values():
            category_counts[tool.category] += 1
            
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            report += f"- {category}: {count}\n"
            
        # Architecture roles
        report += "\n## Architecture Roles\n\n"
        role_tools = defaultdict(list)
        for path, tool in self.tools_catalog.items():
            role_tools[tool.architecture_role].append(tool.metadata.name)
            
        for role, tools in sorted(role_tools.items()):
            report += f"### {role}\n"
            for tool in tools:
                report += f"- {tool}\n"
            report += "\n"
            
        # Integration opportunities
        report += "## Integration Opportunities\n\n"
        for path, tool in self.tools_catalog.items():
            if tool.synergies:
                report += f"### {tool.metadata.name}\n"
                for synergy in tool.synergies:
                    report += f"- {synergy}\n"
                report += "\n"
                
        # Optimization recommendations
        report += "## Optimization Recommendations\n\n"
        for path, tool in self.tools_catalog.items():
            if tool.optimization_recommendations:
                report += f"### {tool.metadata.name}\n"
                for rec in tool.optimization_recommendations:
                    report += f"- {rec}\n"
                report += "\n"
                
        return report
        
    def export_catalog(self, format: str = "json") -> str:
        """Export catalog in specified format"""
        catalog_data = {
            "version": "1.0",
            "generated": datetime.now().isoformat(),
            "total_tools": len(self.tools_catalog),
            "tools": {}
        }
        
        for path, tool in self.tools_catalog.items():
            catalog_data["tools"][path] = asdict(tool)
            
        if format == "json":
            return json.dumps(catalog_data, indent=2)
        elif format == "yaml":
            return yaml.dump(catalog_data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def generate_architecture_diagram(self) -> str:
        """Generate text-based architecture diagram"""
        diagram = """
# System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Master Orchestrator                           │
│                   (Coordination & Management Layer)                  │
└─────────────────────┬───────────────────────┬──────────────────────┘
                      │                       │
        ┌─────────────┴──────────┐ ┌─────────┴──────────────┐
        │   AI/ML Components     │ │  Infrastructure Layer   │
        ├────────────────────────┤ ├────────────────────────┤
"""
        
        # Group tools by role
        role_tools = defaultdict(list)
        for tool in self.tools_catalog.values():
            role_tools[tool.architecture_role].append(tool.metadata.name)
            
        # Add components to diagram
        for role, tools in role_tools.items():
            diagram += f"\n        │ {role:20} │\n"
            for tool in tools[:3]:  # Show top 3 per role
                diagram += f"        │  - {tool:17} │\n"
                
        diagram += """        └────────────────────────┘ └────────────────────────┘

## Data Flow

1. User Request → Master Orchestrator
2. Orchestrator → Appropriate Agent/Service
3. Agent/Service → Data Processing/Storage
4. Results → Aggregation → User Response

## Integration Points

- REST APIs for service communication
- Message queues for async processing
- Shared databases for state management
- Container orchestration for deployment
"""
        
        return diagram
        
    async def start_continuous_scan(self):
        """Start continuous scanning"""
        self.running = True
        logger.info("Starting continuous repository scanning...")
        
        # Initial scan
        await self.scan_all_repositories()
        
        # Set up file system watcher
        observer = Observer()
        handler = RepositoryWatcher(self)
        observer.schedule(handler, str(self.base_path), recursive=True)
        observer.start()
        
        # Periodic full scan
        try:
            while self.running:
                await asyncio.sleep(self.scan_interval)
                await self.scan_all_repositories()
        except KeyboardInterrupt:
            observer.stop()
            self.running = False
            
        observer.join()
        
    async def scan_all_repositories(self):
        """Scan all repositories in base path"""
        logger.info(f"Starting full scan of {self.base_path}")
        
        new_repos = 0
        updated_repos = 0
        
        # Find all git repositories
        repos = []
        for root, dirs, files in os.walk(self.base_path):
            if '.git' in dirs:
                repos.append(root)
                # Don't descend into git repo subdirectories
                dirs[:] = []
                
        logger.info(f"Found {len(repos)} repositories to scan")
        
        # Scan each repository
        for repo_path in repos:
            if repo_path not in self.tools_catalog:
                new_repos += 1
            else:
                # Check if updated
                old_hash = self.tools_catalog[repo_path].content_hash
                new_hash = self._calculate_repo_hash(repo_path)
                if old_hash != new_hash:
                    updated_repos += 1
                    
            self.scan_repository(repo_path)
            
        # Update synergies and redundancies
        self.identify_synergies()
        self.identify_redundancies()
        
        # Generate reports
        report = self.generate_catalog_report()
        with open('tool_catalog_report.md', 'w') as f:
            f.write(report)
            
        # Export catalog
        catalog_json = self.export_catalog('json')
        with open('tool_catalog.json', 'w') as f:
            f.write(catalog_json)
            
        catalog_yaml = self.export_catalog('yaml')
        with open('tool_catalog.yaml', 'w') as f:
            f.write(catalog_yaml)
            
        # Generate architecture diagram
        diagram = self.generate_architecture_diagram()
        with open('architecture_diagram.md', 'w') as f:
            f.write(diagram)
            
        # Log scan results
        self._log_scan_results(len(repos), new_repos, updated_repos)
        
        logger.info(f"Scan complete. New: {new_repos}, Updated: {updated_repos}")
        
    def _log_scan_results(self, total: int, new: int, updated: int):
        """Log scan results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO scan_history (timestamp, total_repos, new_repos, updated_repos)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now(), total, new, updated))
        
        conn.commit()
        conn.close()
        
    def stop(self):
        """Stop continuous scanning"""
        self.running = False
        logger.info("Stopping continuous repository scanning...")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Repository Scanner and Tool Catalog Optimizer')
    parser.add_argument('--base-path', default='/Users/jlazoff/Documents/GitHub', 
                       help='Base path to scan for repositories')
    parser.add_argument('--db-path', default='tool_catalog.db',
                       help='Path to SQLite database')
    parser.add_argument('--scan-interval', type=int, default=3600,
                       help='Scan interval in seconds (default: 3600)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    
    args = parser.parse_args()
    
    # Create scanner
    scanner = ContinuousRepoScanner(args.base_path, args.db_path)
    scanner.scan_interval = args.scan_interval
    
    if args.once:
        # Run once
        asyncio.run(scanner.scan_all_repositories())
    else:
        # Run continuously
        try:
            asyncio.run(scanner.start_continuous_scan())
        except KeyboardInterrupt:
            scanner.stop()
            

if __name__ == '__main__':
    main()