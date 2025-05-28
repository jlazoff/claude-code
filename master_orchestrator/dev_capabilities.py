#!/usr/bin/env python3

"""
Master Orchestrator - Full Development Capabilities
Integrates git, web search, command line control, file system operations, and more
"""

import asyncio
import subprocess
import os
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import shutil
import sys

import structlog

logger = structlog.get_logger()


class GitController:
    """Advanced Git operations and repository management."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.logger = structlog.get_logger("git_controller")
    
    async def clone_repo(self, repo_url: str, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """Clone a repository."""
        try:
            if target_dir:
                clone_path = self.base_path / target_dir
            else:
                repo_name = repo_url.split('/')[-1].replace('.git', '')
                clone_path = self.base_path / repo_name
            
            cmd = ["git", "clone", repo_url, str(clone_path)]
            result = await self._run_git_command(cmd)
            
            if result["success"]:
                self.logger.info(f"Cloned repository: {repo_url} -> {clone_path}")
                return {
                    "success": True,
                    "path": str(clone_path),
                    "url": repo_url,
                    "message": f"Successfully cloned {repo_url}"
                }
            else:
                return {"success": False, "error": result["error"]}
                
        except Exception as e:
            self.logger.error(f"Clone failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_repo_status(self, repo_path: Path) -> Dict[str, Any]:
        """Get comprehensive repository status."""
        try:
            os.chdir(repo_path)
            
            # Get basic status
            status_result = await self._run_git_command(["git", "status", "--porcelain"])
            
            # Get current branch
            branch_result = await self._run_git_command(["git", "branch", "--show-current"])
            
            # Get remote info
            remote_result = await self._run_git_command(["git", "remote", "-v"])
            
            # Get recent commits
            log_result = await self._run_git_command([
                "git", "log", "--oneline", "-10"
            ])
            
            # Get stash list
            stash_result = await self._run_git_command(["git", "stash", "list"])
            
            return {
                "success": True,
                "path": str(repo_path),
                "branch": branch_result.get("output", "").strip(),
                "status": status_result.get("output", "").strip(),
                "remotes": remote_result.get("output", "").strip(),
                "recent_commits": log_result.get("output", "").strip().split('\n')[:10],
                "stashes": stash_result.get("output", "").strip().split('\n') if stash_result.get("output") else [],
                "is_dirty": bool(status_result.get("output", "").strip()),
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def commit_changes(self, repo_path: Path, message: str, add_all: bool = True) -> Dict[str, Any]:
        """Commit changes to repository."""
        try:
            os.chdir(repo_path)
            
            if add_all:
                add_result = await self._run_git_command(["git", "add", "."])
                if not add_result["success"]:
                    return add_result
            
            commit_result = await self._run_git_command(["git", "commit", "-m", message])
            
            if commit_result["success"]:
                self.logger.info(f"Committed changes: {message}")
                return {
                    "success": True,
                    "message": message,
                    "output": commit_result["output"]
                }
            else:
                return commit_result
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def push_changes(self, repo_path: Path, remote: str = "origin", branch: str = None) -> Dict[str, Any]:
        """Push changes to remote repository."""
        try:
            os.chdir(repo_path)
            
            if not branch:
                branch_result = await self._run_git_command(["git", "branch", "--show-current"])
                branch = branch_result.get("output", "").strip() or "main"
            
            push_result = await self._run_git_command(["git", "push", remote, branch])
            
            if push_result["success"]:
                self.logger.info(f"Pushed to {remote}/{branch}")
            
            return push_result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def create_branch(self, repo_path: Path, branch_name: str, checkout: bool = True) -> Dict[str, Any]:
        """Create a new branch."""
        try:
            os.chdir(repo_path)
            
            if checkout:
                result = await self._run_git_command(["git", "checkout", "-b", branch_name])
            else:
                result = await self._run_git_command(["git", "branch", branch_name])
            
            if result["success"]:
                self.logger.info(f"Created branch: {branch_name}")
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _run_git_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a git command and return result."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else "",
                "returncode": process.returncode
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class WebSearchController:
    """Web search and content retrieval capabilities."""
    
    def __init__(self):
        self.logger = structlog.get_logger("web_search")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    async def search_github(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search GitHub repositories."""
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": limit
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            repositories = []
            
            for repo in data.get("items", []):
                repositories.append({
                    "name": repo["name"],
                    "full_name": repo["full_name"],
                    "description": repo.get("description", ""),
                    "url": repo["html_url"],
                    "clone_url": repo["clone_url"],
                    "stars": repo["stargazers_count"],
                    "language": repo.get("language", ""),
                    "updated_at": repo["updated_at"]
                })
            
            return {
                "success": True,
                "query": query,
                "total_count": data.get("total_count", 0),
                "repositories": repositories
            }
            
        except Exception as e:
            self.logger.error(f"GitHub search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_web(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Search the web using DuckDuckGo."""
        try:
            # Using DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            
            # Add abstract if available
            if data.get("Abstract"):
                results.append({
                    "title": data.get("AbstractText", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", ""),
                    "source": "DuckDuckGo Abstract"
                })
            
            # Add related topics
            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                        "source": "DuckDuckGo Related"
                    })
            
            return {
                "success": True,
                "query": query,
                "results": results[:num_results]
            }
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def fetch_webpage(self, url: str) -> Dict[str, Any]:
        """Fetch and parse webpage content."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Basic content extraction
            content = response.text
            
            # Extract title
            title = ""
            if "<title>" in content:
                start = content.find("<title>") + 7
                end = content.find("</title>", start)
                if end > start:
                    title = content[start:end].strip()
            
            return {
                "success": True,
                "url": url,
                "title": title,
                "content": content,
                "length": len(content),
                "status_code": response.status_code
            }
            
        except Exception as e:
            self.logger.error(f"Webpage fetch failed: {e}")
            return {"success": False, "error": str(e)}


class CommandLineController:
    """Advanced command line operations and system control."""
    
    def __init__(self):
        self.logger = structlog.get_logger("command_line")
        self.command_history = []
        
    async def execute_command(self, command: str, cwd: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
        """Execute a command line operation."""
        try:
            self.logger.info(f"Executing command: {command}")
            
            # Security check - prevent dangerous commands
            dangerous_patterns = ['rm -rf /', 'format', 'del /f', 'sudo rm', 'rm -rf *']
            if any(pattern in command.lower() for pattern in dangerous_patterns):
                return {
                    "success": False,
                    "error": "Command blocked for security reasons",
                    "command": command
                }
            
            # Split command
            cmd_parts = command.split()
            
            # Set working directory
            work_dir = Path(cwd) if cwd else Path.cwd()
            
            process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "command": command
                }
            
            # Store in history
            self.command_history.append({
                "command": command,
                "timestamp": datetime.utcnow().isoformat(),
                "returncode": process.returncode,
                "success": process.returncode == 0
            })
            
            # Keep only last 100 commands
            if len(self.command_history) > 100:
                self.command_history = self.command_history[-100:]
            
            result = {
                "success": process.returncode == 0,
                "command": command,
                "returncode": process.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "cwd": str(work_dir)
            }
            
            if result["success"]:
                self.logger.info(f"Command succeeded: {command}")
            else:
                self.logger.error(f"Command failed: {command} (exit code: {process.returncode})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e), "command": command}
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            info = {}
            
            # Operating system
            os_info = await self.execute_command("uname -a")
            if os_info["success"]:
                info["os"] = os_info["stdout"].strip()
            
            # Disk usage
            disk_info = await self.execute_command("df -h")
            if disk_info["success"]:
                info["disk_usage"] = disk_info["stdout"]
            
            # Memory info (macOS)
            if sys.platform == "darwin":
                mem_info = await self.execute_command("vm_stat")
                if mem_info["success"]:
                    info["memory"] = mem_info["stdout"]
            
            # CPU info (macOS)
            cpu_info = await self.execute_command("sysctl -n machdep.cpu.brand_string")
            if cpu_info["success"]:
                info["cpu"] = cpu_info["stdout"].strip()
            
            # Network interfaces
            network_info = await self.execute_command("ifconfig")
            if network_info["success"]:
                info["network"] = network_info["stdout"]
            
            # Running processes
            processes = await self.execute_command("ps aux")
            if processes["success"]:
                info["processes"] = processes["stdout"]
            
            return {
                "success": True,
                "system_info": info,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def install_package(self, package_name: str, package_manager: str = "auto") -> Dict[str, Any]:
        """Install a package using appropriate package manager."""
        try:
            # Detect package manager if auto
            if package_manager == "auto":
                if sys.platform == "darwin":
                    # Check for Homebrew
                    brew_check = await self.execute_command("which brew")
                    if brew_check["success"]:
                        package_manager = "brew"
                    else:
                        package_manager = "pip"
                else:
                    package_manager = "pip"
            
            # Install based on package manager
            if package_manager == "brew":
                result = await self.execute_command(f"brew install {package_name}")
            elif package_manager == "pip":
                result = await self.execute_command(f"pip install {package_name}")
            elif package_manager == "npm":
                result = await self.execute_command(f"npm install -g {package_name}")
            elif package_manager == "apt":
                result = await self.execute_command(f"sudo apt-get install -y {package_name}")
            else:
                return {"success": False, "error": f"Unknown package manager: {package_manager}"}
            
            if result["success"]:
                self.logger.info(f"Package installed: {package_name} via {package_manager}")
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class FileSystemController:
    """Advanced file system operations."""
    
    def __init__(self):
        self.logger = structlog.get_logger("filesystem")
    
    async def search_files(self, pattern: str, directory: str = None, include_content: bool = False) -> Dict[str, Any]:
        """Search for files by pattern."""
        try:
            search_dir = Path(directory) if directory else Path.cwd()
            
            if not search_dir.exists():
                return {"success": False, "error": f"Directory does not exist: {search_dir}"}
            
            matches = []
            
            # Search using glob pattern
            for file_path in search_dir.rglob(pattern):
                if file_path.is_file():
                    file_info = {
                        "path": str(file_path),
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    }
                    
                    if include_content and file_path.suffix in ['.txt', '.py', '.js', '.json', '.md', '.yaml', '.yml']:
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            file_info["content"] = content[:1000]  # First 1000 chars
                            file_info["content_length"] = len(content)
                        except:
                            file_info["content"] = "Could not read content"
                    
                    matches.append(file_info)
            
            return {
                "success": True,
                "pattern": pattern,
                "directory": str(search_dir),
                "matches": matches,
                "count": len(matches)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def create_file(self, file_path: str, content: str = "") -> Dict[str, Any]:
        """Create a new file with optional content."""
        try:
            path = Path(file_path)
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            path.write_text(content, encoding='utf-8')
            
            self.logger.info(f"Created file: {file_path}")
            
            return {
                "success": True,
                "path": file_path,
                "size": len(content),
                "message": f"Created file: {file_path}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def backup_directory(self, source_dir: str, backup_name: str = None) -> Dict[str, Any]:
        """Create a backup of a directory."""
        try:
            source_path = Path(source_dir)
            
            if not source_path.exists():
                return {"success": False, "error": f"Source directory does not exist: {source_dir}"}
            
            if not backup_name:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{source_path.name}_backup_{timestamp}"
            
            backup_path = source_path.parent / backup_name
            
            # Copy directory
            shutil.copytree(source_path, backup_path)
            
            self.logger.info(f"Created backup: {source_dir} -> {backup_path}")
            
            return {
                "success": True,
                "source": source_dir,
                "backup": str(backup_path),
                "message": f"Backup created: {backup_path}"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class MasterDevController:
    """Master controller integrating all development capabilities."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.git = GitController(base_path)
        self.web_search = WebSearchController()
        self.command_line = CommandLineController()
        self.filesystem = FileSystemController()
        self.logger = structlog.get_logger("master_dev")
        
        # Capabilities registry
        self.capabilities = {
            "git": {
                "clone_repo": self.git.clone_repo,
                "get_repo_status": self.git.get_repo_status,
                "commit_changes": self.git.commit_changes,
                "push_changes": self.git.push_changes,
                "create_branch": self.git.create_branch
            },
            "web_search": {
                "search_github": self.web_search.search_github,
                "search_web": self.web_search.search_web,
                "fetch_webpage": self.web_search.fetch_webpage
            },
            "command_line": {
                "execute_command": self.command_line.execute_command,
                "get_system_info": self.command_line.get_system_info,
                "install_package": self.command_line.install_package
            },
            "filesystem": {
                "search_files": self.filesystem.search_files,
                "create_file": self.filesystem.create_file,
                "backup_directory": self.filesystem.backup_directory
            }
        }
    
    async def execute_capability(self, category: str, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute a development capability."""
        try:
            if category not in self.capabilities:
                return {"success": False, "error": f"Unknown capability category: {category}"}
            
            if operation not in self.capabilities[category]:
                return {"success": False, "error": f"Unknown operation: {operation} in {category}"}
            
            # Execute the capability
            func = self.capabilities[category][operation]
            result = await func(**kwargs)
            
            self.logger.info(f"Executed {category}.{operation}", success=result.get("success", False))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Capability execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_available_capabilities(self) -> Dict[str, Any]:
        """Get list of all available capabilities."""
        capabilities_info = {}
        
        for category, operations in self.capabilities.items():
            capabilities_info[category] = {
                "operations": list(operations.keys()),
                "description": self._get_category_description(category)
            }
        
        return {
            "success": True,
            "capabilities": capabilities_info,
            "total_operations": sum(len(ops) for ops in self.capabilities.values())
        }
    
    def _get_category_description(self, category: str) -> str:
        """Get description for capability category."""
        descriptions = {
            "git": "Git repository management and version control operations",
            "web_search": "Web search and content retrieval capabilities",
            "command_line": "Command line execution and system operations",
            "filesystem": "File system operations and management"
        }
        return descriptions.get(category, "")
    
    async def demo_capabilities(self) -> Dict[str, Any]:
        """Demonstrate all capabilities with sample operations."""
        demo_results = {}
        
        try:
            # Demo Git capabilities
            demo_results["git"] = await self.git.get_repo_status(self.base_path)
            
            # Demo Web search
            demo_results["web_search"] = await self.web_search.search_github("python automation", limit=3)
            
            # Demo Command line
            demo_results["command_line"] = await self.command_line.execute_command("echo 'Dev capabilities active!'")
            
            # Demo File system
            demo_results["filesystem"] = await self.filesystem.search_files("*.py", str(self.base_path))
            
            return {
                "success": True,
                "demo_results": demo_results,
                "message": "All development capabilities demonstrated successfully"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Integration with Master Orchestrator
async def integrate_dev_capabilities():
    """Integrate development capabilities with Master Orchestrator."""
    base_path = Path("/Users/jlazoff/Documents/GitHub")
    dev_controller = MasterDevController(base_path)
    
    print("🛠️ Master Orchestrator - Full Development Capabilities")
    print("=" * 60)
    
    # Get available capabilities
    capabilities = await dev_controller.get_available_capabilities()
    print(f"✅ Loaded {capabilities['total_operations']} development operations")
    
    # Demo capabilities
    demo_results = await dev_controller.demo_capabilities()
    if demo_results["success"]:
        print("✅ All development capabilities are operational")
    
    print("\n🔧 Available Capabilities:")
    for category, info in capabilities["capabilities"].items():
        print(f"   • {category}: {len(info['operations'])} operations")
        print(f"     {info['description']}")
    
    print("\n🚀 Integration complete - Full dev capabilities active!")
    
    return dev_controller


if __name__ == "__main__":
    asyncio.run(integrate_dev_capabilities())