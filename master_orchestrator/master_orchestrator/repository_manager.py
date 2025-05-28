"""Repository Management for Master Orchestrator."""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

import structlog
from pydantic import BaseModel, Field

from .config import RepositoryConfig
from .knowledge_graph import KnowledgeGraph

logger = structlog.get_logger()


class RepositoryInfo(BaseModel):
    """Repository information model."""
    
    name: str = Field(description="Repository name")
    path: str = Field(description="Repository path")
    languages: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    config_files: List[str] = Field(default_factory=list)
    file_count: int = Field(default=0)
    last_modified: Optional[datetime] = Field(default=None)
    size_bytes: int = Field(default=0)
    analysis_hash: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RepositoryManager:
    """
    Repository Manager for Master Orchestrator.
    
    Manages analysis, synchronization, and integration of multiple
    GitHub repositories in the ecosystem.
    """
    
    def __init__(self, config: RepositoryConfig, knowledge_graph: KnowledgeGraph):
        self.config = config
        self.knowledge_graph = knowledge_graph
        self.logger = structlog.get_logger("repository_manager")
        
        # Repository tracking
        self.connected_repositories: Dict[str, RepositoryInfo] = {}
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # File patterns for analysis
        self.language_patterns = {
            "python": [".py", ".pyx", ".pyi"],
            "javascript": [".js", ".jsx", ".mjs"],
            "typescript": [".ts", ".tsx"],
            "rust": [".rs"],
            "go": [".go"],
            "java": [".java"],
            "c++": [".cpp", ".cxx", ".cc", ".hpp", ".hxx"],
            "c": [".c", ".h"],
            "shell": [".sh", ".bash", ".zsh"],
            "yaml": [".yml", ".yaml"],
            "json": [".json"],
            "dockerfile": ["Dockerfile", ".dockerfile"],
            "makefile": ["Makefile", "makefile"],
        }
        
        self.technology_indicators = {
            "kubernetes": ["k8s", "kubernetes", ".kube", "helm"],
            "docker": ["docker", "compose", "dockerfile"],
            "terraform": [".tf", ".tfvars"],
            "ansible": ["ansible", ".yml", "playbook"],
            "nodejs": ["package.json", "node_modules"],
            "python": ["requirements.txt", "pyproject.toml", "setup.py", "poetry.lock"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "go": ["go.mod", "go.sum"],
            "frontend": ["src/", "public/", "build/", "dist/"],
            "backend": ["api/", "server/", "service/"],
            "database": ["migrations/", "schema/", "models/"],
            "ai": ["model", "train", "inference", "agent"],
            "ml": ["dataset", "features", "pipeline", "model"]
        }
    
    async def initialize(self) -> None:
        """Initialize repository manager."""
        self.logger.info("Initializing Repository Manager")
        
        try:
            # Discover repositories
            await self._discover_repositories()
            
            # Initial analysis of repositories
            if self.config.auto_sync:
                await self._analyze_all_repositories()
            
            self.logger.info(
                "Repository Manager initialized",
                repositories_found=len(self.connected_repositories)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Repository Manager: {e}")
            raise
    
    async def _discover_repositories(self) -> None:
        """Discover repositories in the base path."""
        if not self.config.github_base_path.exists():
            self.logger.warning(f"GitHub base path does not exist: {self.config.github_base_path}")
            return
        
        for repo_dir in self.config.github_base_path.iterdir():
            if repo_dir.is_dir() and not repo_dir.name.startswith('.'):
                repo_info = RepositoryInfo(
                    name=repo_dir.name,
                    path=str(repo_dir)
                )
                self.connected_repositories[repo_dir.name] = repo_info
                
                self.logger.debug(f"Discovered repository: {repo_dir.name}")
    
    async def _analyze_all_repositories(self) -> None:
        """Analyze all discovered repositories."""
        self.logger.info("Starting analysis of all repositories")
        
        analysis_tasks = []
        for repo_name in self.connected_repositories:
            task = asyncio.create_task(self._analyze_repository_internal(repo_name))
            analysis_tasks.append(task)
        
        # Run analyses concurrently with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent analyses
        
        async def limited_analysis(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[limited_analysis(task) for task in analysis_tasks],
            return_exceptions=True
        )
        
        successful_analyses = sum(1 for result in results if not isinstance(result, Exception))
        self.logger.info(f"Completed repository analysis: {successful_analyses}/{len(analysis_tasks)} successful")
    
    async def analyze_repository(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze a specific repository."""
        repo_name = repo_path.name
        
        # Check if repository is tracked
        if repo_name not in self.connected_repositories:
            repo_info = RepositoryInfo(
                name=repo_name,
                path=str(repo_path)
            )
            self.connected_repositories[repo_name] = repo_info
        
        return await self._analyze_repository_internal(repo_name)
    
    async def _analyze_repository_internal(self, repo_name: str) -> Dict[str, Any]:
        """Internal repository analysis implementation."""
        repo_info = self.connected_repositories[repo_name]
        repo_path = Path(repo_info.path)
        
        if not repo_path.exists():
            self.logger.error(f"Repository path does not exist: {repo_path}")
            return {}
        
        try:
            # Calculate hash for caching
            analysis_hash = await self._calculate_repo_hash(repo_path)
            
            # Check cache
            if repo_name in self.analysis_cache and self.analysis_cache[repo_name].get("hash") == analysis_hash:
                self.logger.debug(f"Using cached analysis for {repo_name}")
                return self.analysis_cache[repo_name]
            
            self.logger.info(f"Analyzing repository: {repo_name}")
            
            # Perform analysis
            analysis_result = {
                "name": repo_name,
                "path": str(repo_path),
                "hash": analysis_hash,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Analyze file structure
            file_analysis = await self._analyze_files(repo_path)
            analysis_result.update(file_analysis)
            
            # Analyze technologies
            tech_analysis = await self._analyze_technologies(repo_path)
            analysis_result.update(tech_analysis)
            
            # Analyze capabilities
            capability_analysis = await self._analyze_capabilities(repo_path)
            analysis_result.update(capability_analysis)
            
            # Read documentation
            doc_analysis = await self._analyze_documentation(repo_path)
            analysis_result.update(doc_analysis)
            
            # Update repository info
            repo_info.languages = analysis_result.get("languages", [])
            repo_info.technologies = analysis_result.get("technologies", [])
            repo_info.capabilities = analysis_result.get("capabilities", [])
            repo_info.entry_points = analysis_result.get("entry_points", [])
            repo_info.config_files = analysis_result.get("config_files", [])
            repo_info.file_count = analysis_result.get("file_count", 0)
            repo_info.size_bytes = analysis_result.get("size_bytes", 0)
            repo_info.analysis_hash = analysis_hash
            repo_info.last_modified = datetime.utcnow()
            
            # Cache result
            self.analysis_cache[repo_name] = analysis_result
            
            # Update knowledge graph
            await self._update_knowledge_graph(repo_info, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Repository analysis failed for {repo_name}: {e}")
            return {}
    
    async def _calculate_repo_hash(self, repo_path: Path) -> str:
        """Calculate a hash for repository state."""
        hash_inputs = []
        
        try:
            # Include modification times of key files
            key_files = ["README.md", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"]
            for file_name in key_files:
                file_path = repo_path / file_name
                if file_path.exists():
                    stat = file_path.stat()
                    hash_inputs.append(f"{file_name}:{stat.st_mtime}")
            
            # Include directory structure hash
            all_files = list(repo_path.rglob("*"))
            file_list = sorted([str(f.relative_to(repo_path)) for f in all_files if f.is_file()])
            hash_inputs.extend(file_list[:100])  # Limit for performance
            
            combined = "|".join(hash_inputs)
            return hashlib.md5(combined.encode()).hexdigest()
            
        except Exception:
            return hashlib.md5(str(repo_path).encode()).hexdigest()
    
    async def _analyze_files(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze file structure and languages."""
        languages = set()
        entry_points = []
        config_files = []
        file_count = 0
        size_bytes = 0
        
        try:
            for file_path in repo_path.rglob("*"):
                if file_path.is_file() and not any(part.startswith('.') for part in file_path.parts):
                    file_count += 1
                    
                    try:
                        size_bytes += file_path.stat().st_size
                    except:
                        pass
                    
                    # Detect language
                    suffix = file_path.suffix.lower()
                    name = file_path.name.lower()
                    
                    for lang, extensions in self.language_patterns.items():
                        if suffix in extensions or name in extensions:
                            languages.add(lang)
                            break
                    
                    # Identify entry points
                    if name in ["main.py", "app.py", "server.py", "index.js", "main.js", "main.go", "main.rs"]:
                        entry_points.append(str(file_path.relative_to(repo_path)))
                    
                    # Identify config files
                    if name in ["package.json", "pyproject.toml", "cargo.toml", "go.mod", "dockerfile", "docker-compose.yml"]:
                        config_files.append(str(file_path.relative_to(repo_path)))
        
        except Exception as e:
            self.logger.warning(f"File analysis error: {e}")
        
        return {
            "languages": list(languages),
            "entry_points": entry_points,
            "config_files": config_files,
            "file_count": file_count,
            "size_bytes": size_bytes
        }
    
    async def _analyze_technologies(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze technologies used in the repository."""
        technologies = set()
        
        try:
            # Check file presence for technology detection
            all_files = [f.name.lower() for f in repo_path.rglob("*") if f.is_file()]
            all_dirs = [d.name.lower() for d in repo_path.rglob("*") if d.is_dir()]
            all_paths = [str(p).lower() for p in repo_path.rglob("*")]
            
            for tech, indicators in self.technology_indicators.items():
                if any(
                    indicator in all_files or 
                    indicator in all_dirs or 
                    any(indicator in path for path in all_paths)
                    for indicator in indicators
                ):
                    technologies.add(tech)
        
        except Exception as e:
            self.logger.warning(f"Technology analysis error: {e}")
        
        return {"technologies": list(technologies)}
    
    async def _analyze_capabilities(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze capabilities of the repository."""
        capabilities = set()
        
        try:
            # Read README for capability keywords
            readme_content = ""
            for readme_name in ["README.md", "README.txt", "README"]:
                readme_path = repo_path / readme_name
                if readme_path.exists():
                    try:
                        readme_content = readme_path.read_text(encoding='utf-8').lower()
                        break
                    except:
                        continue
            
            # Capability keywords
            capability_keywords = {
                "api": ["api", "endpoint", "rest", "graphql"],
                "web": ["web", "frontend", "ui", "react", "vue", "angular"],
                "agent": ["agent", "autonomous", "ai agent", "assistant"],
                "llm": ["llm", "language model", "gpt", "claude", "gemini"],
                "ai": ["artificial intelligence", "machine learning", "neural", "deep learning"],
                "database": ["database", "db", "sql", "nosql", "mongodb", "postgres"],
                "cli": ["cli", "command line", "terminal", "console"],
                "backend": ["backend", "server", "service", "microservice"],
                "data": ["data", "analytics", "etl", "pipeline", "processing"],
                "monitoring": ["monitoring", "metrics", "logging", "observability"],
                "orchestration": ["orchestration", "workflow", "automation", "scheduler"]
            }
            
            for capability, keywords in capability_keywords.items():
                if any(keyword in readme_content for keyword in keywords):
                    capabilities.add(capability)
            
            # Check directory structure for capabilities
            dir_names = [d.name.lower() for d in repo_path.iterdir() if d.is_dir()]
            
            if any(name in ["api", "endpoints", "routes"] for name in dir_names):
                capabilities.add("api")
            if any(name in ["frontend", "ui", "web", "client"] for name in dir_names):
                capabilities.add("web")
            if any(name in ["backend", "server", "service"] for name in dir_names):
                capabilities.add("backend")
            if any(name in ["agents", "agent"] for name in dir_names):
                capabilities.add("agent")
        
        except Exception as e:
            self.logger.warning(f"Capability analysis error: {e}")
        
        return {"capabilities": list(capabilities)}
    
    async def _analyze_documentation(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze documentation in the repository."""
        documentation = {}
        
        try:
            # Read README
            for readme_name in ["README.md", "README.txt", "README"]:
                readme_path = repo_path / readme_name
                if readme_path.exists():
                    try:
                        content = readme_path.read_text(encoding='utf-8')
                        documentation["readme"] = {
                            "file": readme_name,
                            "length": len(content),
                            "preview": content[:500] + "..." if len(content) > 500 else content
                        }
                        break
                    except:
                        continue
            
            # Look for other documentation
            doc_files = []
            for pattern in ["*.md", "docs/*", "documentation/*"]:
                doc_files.extend(repo_path.glob(pattern))
            
            documentation["doc_files"] = [str(f.relative_to(repo_path)) for f in doc_files[:10]]
        
        except Exception as e:
            self.logger.warning(f"Documentation analysis error: {e}")
        
        return {"documentation": documentation}
    
    async def _update_knowledge_graph(self, repo_info: RepositoryInfo, analysis: Dict[str, Any]) -> None:
        """Update knowledge graph with repository information."""
        try:
            # Create or update project node
            project_node_id = await self.knowledge_graph.create_project_node({
                "name": repo_info.name,
                "path": repo_info.path,
                "technologies": repo_info.technologies,
                "capabilities": repo_info.capabilities,
                "metadata": {
                    "languages": repo_info.languages,
                    "file_count": repo_info.file_count,
                    "size_bytes": repo_info.size_bytes,
                    "entry_points": repo_info.entry_points,
                    "config_files": repo_info.config_files,
                    "analysis": analysis
                }
            })
            
            self.logger.debug(f"Updated knowledge graph for repository: {repo_info.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to update knowledge graph: {e}")
    
    async def monitor_repositories(self) -> None:
        """Monitor repositories for changes."""
        self.logger.info("Starting repository monitoring")
        
        while True:
            try:
                if self.config.auto_sync:
                    # Check for repository changes
                    changed_repos = await self._check_for_changes()
                    
                    if changed_repos:
                        self.logger.info(f"Changes detected in {len(changed_repos)} repositories")
                        
                        # Re-analyze changed repositories
                        for repo_name in changed_repos:
                            await self._analyze_repository_internal(repo_name)
                
                await asyncio.sleep(self.config.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Repository monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_for_changes(self) -> List[str]:
        """Check for changes in repositories."""
        changed_repos = []
        
        for repo_name, repo_info in self.connected_repositories.items():
            repo_path = Path(repo_info.path)
            
            if not repo_path.exists():
                continue
            
            try:
                current_hash = await self._calculate_repo_hash(repo_path)
                if current_hash != repo_info.analysis_hash:
                    changed_repos.append(repo_name)
            except Exception as e:
                self.logger.warning(f"Error checking changes for {repo_name}: {e}")
        
        return changed_repos
    
    async def get_repository_summary(self) -> Dict[str, Any]:
        """Get summary of all repositories."""
        total_repos = len(self.connected_repositories)
        total_files = sum(repo.file_count for repo in self.connected_repositories.values())
        total_size = sum(repo.size_bytes for repo in self.connected_repositories.values())
        
        # Technology breakdown
        tech_count = {}
        for repo in self.connected_repositories.values():
            for tech in repo.technologies:
                tech_count[tech] = tech_count.get(tech, 0) + 1
        
        # Capability breakdown
        capability_count = {}
        for repo in self.connected_repositories.values():
            for capability in repo.capabilities:
                capability_count[capability] = capability_count.get(capability, 0) + 1
        
        return {
            "total_repositories": total_repos,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "technologies": dict(sorted(tech_count.items(), key=lambda x: x[1], reverse=True)),
            "capabilities": dict(sorted(capability_count.items(), key=lambda x: x[1], reverse=True)),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def shutdown(self) -> None:
        """Shutdown repository manager."""
        self.logger.info("Shutting down Repository Manager")
        
        # Clear caches
        self.analysis_cache.clear()
        self.connected_repositories.clear()
        
        self.logger.info("Repository Manager shutdown complete")