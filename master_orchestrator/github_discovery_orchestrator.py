#!/usr/bin/env python3
"""
GitHub Discovery Orchestrator
Automatically discovers, analyzes, and containerizes GitHub repositories
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
import tempfile
import docker
import git
import requests
from pydantic import BaseModel, Field

# Local framework integration
from local_agentic_framework import LocalAgenticFramework, KnowledgeGraphNode, DataLakeRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubRepository(BaseModel):
    """GitHub repository model"""
    name: str = Field(..., description="Repository name")
    full_name: str = Field(..., description="Full repository name (owner/repo)")
    url: str = Field(..., description="Repository URL")
    description: str = Field(default="", description="Repository description")
    language: str = Field(default="unknown", description="Primary language")
    stars: int = Field(default=0, description="Star count")
    forks: int = Field(default=0, description="Fork count")
    topics: List[str] = Field(default_factory=list, description="Repository topics")
    has_dockerfile: bool = Field(default=False, description="Contains Dockerfile")
    has_requirements: bool = Field(default=False, description="Contains requirements")
    has_package_json: bool = Field(default=False, description="Contains package.json")
    dependencies: List[str] = Field(default_factory=list, description="Detected dependencies")
    framework_type: str = Field(default="unknown", description="Detected framework type")
    containerization_status: str = Field(default="pending", description="Containerization status")
    analysis_results: Dict[str, Any] = Field(default_factory=dict, description="Analysis results")

class ContainerEnvironment(BaseModel):
    """Container environment model"""
    container_id: str = Field(..., description="Docker container ID")
    repository_name: str = Field(..., description="Source repository name")
    image_name: str = Field(..., description="Docker image name")
    port_mappings: Dict[int, int] = Field(default_factory=dict, description="Port mappings")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    status: str = Field(default="created", description="Container status")
    health_check_url: Optional[str] = Field(None, description="Health check URL")
    management_commands: List[str] = Field(default_factory=list, description="Management commands")

class GitHubDiscoveryOrchestrator:
    """Orchestrates GitHub repository discovery and containerization"""
    
    def __init__(self, framework: LocalAgenticFramework):
        self.framework = framework
        self.foundation_dir = Path("foundation_data")
        self.repos_dir = self.foundation_dir / "discovered_repos"
        self.containers_dir = self.foundation_dir / "containers"
        self.analysis_dir = self.foundation_dir / "repo_analysis"
        
        # Create directories
        for dir_path in [self.repos_dir, self.containers_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("✅ Docker client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Docker: {e}")
            self.docker_client = None
        
        # Repository database
        self.discovered_repos: Dict[str, GitHubRepository] = {}
        self.container_environments: Dict[str, ContainerEnvironment] = {}
        
        # Search patterns for different types of repositories
        self.search_patterns = {
            "ai_frameworks": ["machine learning", "deep learning", "neural network", "transformer", "llm"],
            "agent_frameworks": ["autonomous agent", "multi-agent", "agent framework", "agentic"],
            "automation_tools": ["automation", "workflow", "orchestration", "pipeline"],
            "development_tools": ["developer tools", "code generation", "ide", "editor"],
            "data_processing": ["data processing", "etl", "data pipeline", "analytics"],
            "web_frameworks": ["web framework", "api", "rest", "graphql"],
            "monitoring_tools": ["monitoring", "logging", "metrics", "observability"],
            "security_tools": ["security", "authentication", "encryption", "vulnerability"]
        }
        
        logger.info("GitHub Discovery Orchestrator initialized")

    async def discover_github_repositories(self) -> List[GitHubRepository]:
        """Discover relevant GitHub repositories"""
        logger.info("🔍 Starting GitHub repository discovery...")
        
        discovered = []
        
        # Search for repositories using different patterns
        for category, patterns in self.search_patterns.items():
            logger.info(f"🔎 Searching for {category} repositories...")
            
            for pattern in patterns:
                try:
                    repos = await self.search_github_repos(pattern, category)
                    discovered.extend(repos)
                    await asyncio.sleep(1)  # Rate limiting
                except Exception as e:
                    logger.warning(f"Search failed for '{pattern}': {e}")
        
        # Remove duplicates and store
        unique_repos = {}
        for repo in discovered:
            if repo.full_name not in unique_repos:
                unique_repos[repo.full_name] = repo
                self.discovered_repos[repo.full_name] = repo
        
        logger.info(f"✅ Discovered {len(unique_repos)} unique repositories")
        
        # Save discovery results
        discovery_file = self.analysis_dir / f"discovery_results_{int(time.time())}.json"
        with open(discovery_file, 'w') as f:
            json.dump([repo.model_dump() for repo in unique_repos.values()], f, indent=2, default=str)
        
        return list(unique_repos.values())

    async def search_github_repos(self, query: str, category: str) -> List[GitHubRepository]:
        """Search GitHub for repositories matching query"""
        url = "https://api.github.com/search/repositories"
        params = {
            "q": f"{query} language:python OR language:javascript OR language:typescript OR language:go OR language:rust",
            "sort": "stars",
            "order": "desc",
            "per_page": 20
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                repos = []
                
                for item in data.get("items", []):
                    repo = GitHubRepository(
                        name=item["name"],
                        full_name=item["full_name"],
                        url=item["html_url"],
                        description=item.get("description", ""),
                        language=item.get("language", "unknown"),
                        stars=item.get("stargazers_count", 0),
                        forks=item.get("forks_count", 0),
                        topics=item.get("topics", []),
                        framework_type=category
                    )
                    repos.append(repo)
                
                return repos
            else:
                logger.warning(f"GitHub API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to search GitHub: {e}")
            return []

    async def analyze_repository(self, repo: GitHubRepository) -> GitHubRepository:
        """Analyze a repository for containerization potential"""
        logger.info(f"🔬 Analyzing repository: {repo.full_name}")
        
        try:
            # Clone repository to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = Path(temp_dir) / repo.name
                
                # Clone repository
                git.Repo.clone_from(repo.url, repo_path, depth=1)
                
                # Analyze repository structure
                analysis = await self.analyze_repo_structure(repo_path)
                repo.analysis_results = analysis
                
                # Detect containerization files
                repo.has_dockerfile = (repo_path / "Dockerfile").exists()
                repo.has_requirements = (repo_path / "requirements.txt").exists()
                repo.has_package_json = (repo_path / "package.json").exists()
                
                # Extract dependencies
                repo.dependencies = await self.extract_dependencies(repo_path)
                
                # Store analysis in knowledge graph
                await self.store_repo_analysis(repo)
                
                logger.info(f"✅ Analysis complete for {repo.full_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to analyze {repo.full_name}: {e}")
            repo.analysis_results = {"error": str(e)}
        
        return repo

    async def analyze_repo_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze repository structure and characteristics"""
        analysis = {
            "file_types": {},
            "directory_structure": [],
            "config_files": [],
            "documentation": [],
            "tests": [],
            "size_metrics": {},
            "potential_entry_points": []
        }
        
        try:
            # Count file types
            for file_path in repo_path.rglob("*"):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    analysis["file_types"][suffix] = analysis["file_types"].get(suffix, 0) + 1
                    
                    # Identify important files
                    name = file_path.name.lower()
                    if name in ["dockerfile", "docker-compose.yml", "requirements.txt", "package.json", "setup.py", "pyproject.toml"]:
                        analysis["config_files"].append(str(file_path.relative_to(repo_path)))
                    elif name in ["readme.md", "readme.rst", "docs"]:
                        analysis["documentation"].append(str(file_path.relative_to(repo_path)))
                    elif "test" in name or file_path.parent.name.lower() == "tests":
                        analysis["tests"].append(str(file_path.relative_to(repo_path)))
                    elif name in ["main.py", "app.py", "server.py", "index.js", "main.js"]:
                        analysis["potential_entry_points"].append(str(file_path.relative_to(repo_path)))
            
            # Analyze directory structure
            for dir_path in repo_path.iterdir():
                if dir_path.is_dir() and not dir_path.name.startswith('.'):
                    analysis["directory_structure"].append(dir_path.name)
            
            # Calculate size metrics
            total_size = sum(f.stat().st_size for f in repo_path.rglob("*") if f.is_file())
            total_files = len(list(repo_path.rglob("*")))
            
            analysis["size_metrics"] = {
                "total_size_bytes": total_size,
                "total_files": total_files,
                "avg_file_size": total_size / max(total_files, 1)
            }
            
        except Exception as e:
            logger.warning(f"Error in structure analysis: {e}")
            analysis["error"] = str(e)
        
        return analysis

    async def extract_dependencies(self, repo_path: Path) -> List[str]:
        """Extract dependencies from various configuration files"""
        dependencies = set()
        
        # Python dependencies
        for req_file in ["requirements.txt", "setup.py", "pyproject.toml"]:
            req_path = repo_path / req_file
            if req_path.exists():
                try:
                    content = req_path.read_text(encoding='utf-8')
                    if req_file == "requirements.txt":
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                dep = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                                dependencies.add(dep)
                    elif req_file == "setup.py":
                        # Basic extraction from setup.py
                        if "install_requires" in content:
                            # This is a simplified extraction
                            import re
                            matches = re.findall(r"['\"]([a-zA-Z0-9\-_]+)['\"]", content)
                            dependencies.update(matches[:20])  # Limit to first 20
                except Exception as e:
                    logger.warning(f"Failed to parse {req_file}: {e}")
        
        # Node.js dependencies
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    deps = data.get("dependencies", {})
                    dev_deps = data.get("devDependencies", {})
                    dependencies.update(deps.keys())
                    dependencies.update(dev_deps.keys())
            except Exception as e:
                logger.warning(f"Failed to parse package.json: {e}")
        
        return list(dependencies)[:50]  # Limit to 50 dependencies

    async def store_repo_analysis(self, repo: GitHubRepository):
        """Store repository analysis in knowledge graph"""
        node = KnowledgeGraphNode(
            node_id=f"repo_{repo.name}_{int(time.time())}",
            node_type="github_repository",
            content={
                "repository": repo.model_dump(),
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            metadata={
                "source": "github_discovery",
                "framework_type": repo.framework_type,
                "containerizable": repo.has_dockerfile or repo.has_requirements or repo.has_package_json
            },
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        await self.framework.store_knowledge_node(node)

    async def create_container_environment(self, repo: GitHubRepository) -> Optional[ContainerEnvironment]:
        """Create isolated container environment for repository"""
        logger.info(f"📦 Creating container environment for {repo.full_name}")
        
        if not self.docker_client:
            logger.error("Docker client not available")
            return None
        
        try:
            # Generate Dockerfile if not present
            dockerfile_content = await self.generate_dockerfile(repo)
            
            # Create build context
            build_dir = self.containers_dir / repo.name
            build_dir.mkdir(exist_ok=True)
            
            # Write Dockerfile
            dockerfile_path = build_dir / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)
            
            # Clone repository into build context
            repo_build_path = build_dir / "app"
            if repo_build_path.exists():
                import shutil
                shutil.rmtree(repo_build_path)
            
            git.Repo.clone_from(repo.url, repo_build_path, depth=1)
            
            # Build Docker image
            image_name = f"discovered-repo-{repo.name.lower()}:latest"
            
            logger.info(f"🔨 Building Docker image: {image_name}")
            image, build_logs = self.docker_client.images.build(
                path=str(build_dir),
                tag=image_name,
                rm=True
            )
            
            # Start container with appropriate port mapping
            container_port = self.detect_service_port(repo)
            host_port = self.get_available_port()
            
            logger.info(f"🚀 Starting container on port {host_port}")
            container = self.docker_client.containers.run(
                image_name,
                detach=True,
                ports={f"{container_port}/tcp": host_port} if container_port else {},
                name=f"repo-{repo.name.lower()}-{int(time.time())}",
                restart_policy={"Name": "unless-stopped"}
            )
            
            # Create container environment record
            env = ContainerEnvironment(
                container_id=container.id,
                repository_name=repo.full_name,
                image_name=image_name,
                port_mappings={container_port: host_port} if container_port else {},
                status="running",
                health_check_url=f"http://localhost:{host_port}" if container_port else None,
                management_commands=[
                    f"docker logs {container.id}",
                    f"docker stop {container.id}",
                    f"docker restart {container.id}"
                ]
            )
            
            self.container_environments[repo.full_name] = env
            repo.containerization_status = "completed"
            
            # Store container info in data lake
            record = DataLakeRecord(
                record_id=f"container_{repo.name}_{int(time.time())}",
                table_name="container_environments",
                data=env.model_dump(),
                schema_version="1.0",
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            await self.framework.store_data_lake_record(record)
            
            logger.info(f"✅ Container environment created for {repo.full_name}")
            return env
            
        except Exception as e:
            logger.error(f"❌ Failed to create container for {repo.full_name}: {e}")
            repo.containerization_status = "failed"
            return None

    async def generate_dockerfile(self, repo: GitHubRepository) -> str:
        """Generate appropriate Dockerfile for repository"""
        
        # Check if repository already has Dockerfile
        if repo.has_dockerfile:
            # Try to get existing Dockerfile content
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo_path = Path(temp_dir) / repo.name
                    git.Repo.clone_from(repo.url, repo_path, depth=1)
                    dockerfile_path = repo_path / "Dockerfile"
                    if dockerfile_path.exists():
                        return dockerfile_path.read_text()
            except Exception as e:
                logger.warning(f"Failed to get existing Dockerfile: {e}")
        
        # Generate Dockerfile based on detected language and dependencies
        if repo.language.lower() == "python":
            return self.generate_python_dockerfile(repo)
        elif repo.language.lower() in ["javascript", "typescript"]:
            return self.generate_nodejs_dockerfile(repo)
        elif repo.language.lower() == "go":
            return self.generate_go_dockerfile(repo)
        else:
            return self.generate_generic_dockerfile(repo)

    def generate_python_dockerfile(self, repo: GitHubRepository) -> str:
        """Generate Dockerfile for Python repositories"""
        return f'''FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY app/requirements.txt* ./
COPY app/setup.py* ./
COPY app/pyproject.toml* ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt || \\
    pip install --no-cache-dir -e . || \\
    pip install --no-cache-dir {" ".join(repo.dependencies[:10])} || \\
    echo "No specific dependencies found"

# Copy application code
COPY app/ .

# Try to determine the entry point
EXPOSE 8000

# Multiple possible entry points
CMD python main.py || \\
    python app.py || \\
    python server.py || \\
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 || \\
    python -m flask run --host 0.0.0.0 --port 8000 || \\
    python -m http.server 8000 || \\
    bash
'''

    def generate_nodejs_dockerfile(self, repo: GitHubRepository) -> str:
        """Generate Dockerfile for Node.js repositories"""
        return f'''FROM node:18-slim

WORKDIR /app

# Copy package files
COPY app/package*.json ./

# Install dependencies
RUN npm ci --only=production || npm install

# Copy application code
COPY app/ .

EXPOSE 3000

# Multiple possible entry points
CMD npm start || \\
    node server.js || \\
    node index.js || \\
    node main.js || \\
    node app.js || \\
    npx serve -s . -l 3000 || \\
    bash
'''

    def generate_go_dockerfile(self, repo: GitHubRepository) -> str:
        """Generate Dockerfile for Go repositories"""
        return f'''FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY app/ .

RUN go mod download || go mod init {repo.name}
RUN go build -o main .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/

COPY --from=builder /app/main .

EXPOSE 8080

CMD ["./main"]
'''

    def generate_generic_dockerfile(self, repo: GitHubRepository) -> str:
        """Generate generic Dockerfile"""
        return f'''FROM ubuntu:22.04

WORKDIR /app

# Install common dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    git \\
    python3 \\
    python3-pip \\
    nodejs \\
    npm \\
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY app/ .

EXPOSE 8000

# Generic entry point
CMD bash -c "echo 'Repository: {repo.full_name}' && \\
    echo 'Language: {repo.language}' && \\
    echo 'Starting generic container...' && \\
    tail -f /dev/null"
'''

    def detect_service_port(self, repo: GitHubRepository) -> int:
        """Detect likely service port for repository"""
        if repo.language.lower() == "python":
            return 8000
        elif repo.language.lower() in ["javascript", "typescript"]:
            return 3000
        elif repo.language.lower() == "go":
            return 8080
        else:
            return 8000

    def get_available_port(self) -> int:
        """Get available port for container mapping"""
        import socket
        
        for port in range(9000, 9100):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        return 9000  # Fallback

    async def orchestrate_discovery_and_containerization(self) -> Dict[str, Any]:
        """Main orchestration method"""
        logger.info("🎯 Starting GitHub Discovery and Containerization Orchestration")
        
        start_time = time.time()
        
        # Phase 1: Discovery
        repositories = await self.discover_github_repositories()
        
        # Phase 2: Analysis
        logger.info("🔬 Analyzing discovered repositories...")
        analyzed_repos = []
        for repo in repositories[:10]:  # Limit to first 10 for testing
            analyzed_repo = await self.analyze_repository(repo)
            analyzed_repos.append(analyzed_repo)
            await asyncio.sleep(1)  # Rate limiting
        
        # Phase 3: Containerization
        logger.info("📦 Creating container environments...")
        containerized_repos = []
        for repo in analyzed_repos[:5]:  # Limit to first 5 for containerization
            if repo.has_dockerfile or repo.has_requirements or repo.has_package_json:
                container_env = await self.create_container_environment(repo)
                if container_env:
                    containerized_repos.append(repo)
            await asyncio.sleep(2)  # Give containers time to start
        
        # Compile results
        execution_time = time.time() - start_time
        
        results = {
            "execution_time": execution_time,
            "discovered_repositories": len(repositories),
            "analyzed_repositories": len(analyzed_repos),
            "containerized_repositories": len(containerized_repos),
            "repositories": [repo.model_dump() for repo in analyzed_repos],
            "container_environments": [env.model_dump() for env in self.container_environments.values()],
            "summary": {
                "languages": list(set(repo.language for repo in analyzed_repos)),
                "framework_types": list(set(repo.framework_type for repo in analyzed_repos)),
                "containerizable": len([repo for repo in analyzed_repos if repo.has_dockerfile or repo.has_requirements or repo.has_package_json]),
                "total_stars": sum(repo.stars for repo in analyzed_repos),
                "top_repositories": sorted(analyzed_repos, key=lambda r: r.stars, reverse=True)[:5]
            }
        }
        
        # Save orchestration results
        results_file = self.analysis_dir / f"orchestration_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Orchestration complete in {execution_time:.2f}s")
        logger.info(f"   Discovered: {len(repositories)} repositories")
        logger.info(f"   Analyzed: {len(analyzed_repos)} repositories") 
        logger.info(f"   Containerized: {len(containerized_repos)} repositories")
        
        return results

async def main():
    """Test the GitHub Discovery Orchestrator"""
    # Initialize framework
    framework = LocalAgenticFramework()
    await asyncio.sleep(5)  # Wait for framework initialization
    
    # Initialize orchestrator
    orchestrator = GitHubDiscoveryOrchestrator(framework)
    
    # Run orchestration
    results = await orchestrator.orchestrate_discovery_and_containerization()
    
    print(f"\n🎯 GitHub Discovery and Containerization Results:")
    print(f"   Execution Time: {results['execution_time']:.2f}s")
    print(f"   Discovered Repositories: {results['discovered_repositories']}")
    print(f"   Analyzed Repositories: {results['analyzed_repositories']}")
    print(f"   Containerized Repositories: {results['containerized_repositories']}")
    
    print(f"\n📊 Summary:")
    print(f"   Languages: {', '.join(results['summary']['languages'])}")
    print(f"   Framework Types: {', '.join(results['summary']['framework_types'])}")
    print(f"   Containerizable: {results['summary']['containerizable']}")
    print(f"   Total Stars: {results['summary']['total_stars']}")
    
    print(f"\n🏆 Top Repositories:")
    for repo in results['summary']['top_repositories']:
        print(f"   • {repo['full_name']} ({repo['stars']} stars) - {repo['language']}")

if __name__ == "__main__":
    asyncio.run(main())