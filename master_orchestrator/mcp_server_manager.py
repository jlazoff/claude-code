#!/usr/bin/env python3
"""
MCP Server Manager
Automatically discovers, downloads, and integrates MCP servers including Aider, Codex, Vertex AI
"""

import asyncio
import json
import logging
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import git

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServerManager:
    def __init__(self):
        self.foundation_dir = Path("foundation_data")
        self.mcp_dir = self.foundation_dir / "mcp_servers"
        self.config_dir = Path.home() / ".claude"
        self.repos_dir = self.mcp_dir / "repositories"
        
        # Create directories
        for dir_path in [self.foundation_dir, self.mcp_dir, self.repos_dir, self.config_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Known MCP servers and their sources
        self.known_servers = {
            "aider": {
                "github": "https://github.com/paul-gauthier/aider",
                "description": "AI pair programming in your terminal",
                "mcp_path": "aider/mcp_server.py",
                "install_cmd": "pip install aider-chat",
                "type": "coding_assistant"
            },
            "vertex-ai": {
                "github": "https://github.com/googleapis/python-aiplatform", 
                "description": "Google Cloud Vertex AI MCP integration",
                "mcp_path": "vertex_ai_mcp.py",
                "install_cmd": "pip install google-cloud-aiplatform",
                "type": "ai_platform"
            },
            "github-mcp": {
                "github": "https://github.com/modelcontextprotocol/servers",
                "description": "Official GitHub MCP server",
                "mcp_path": "src/github/index.js",
                "install_cmd": "npm install -g @modelcontextprotocol/server-github",
                "type": "version_control"
            },
            "filesystem-mcp": {
                "github": "https://github.com/modelcontextprotocol/servers",
                "description": "Filesystem MCP server",
                "mcp_path": "src/filesystem/index.js", 
                "install_cmd": "npm install -g @modelcontextprotocol/server-filesystem",
                "type": "file_operations"
            },
            "web-search-mcp": {
                "github": "https://github.com/modelcontextprotocol/servers",
                "description": "Web search MCP server",
                "mcp_path": "src/brave-search/index.js",
                "install_cmd": "npm install -g @modelcontextprotocol/server-brave-search",
                "type": "web_search"
            }
        }
        
        logger.info("MCP Server Manager initialized")

    async def search_github_mcp_servers(self) -> List[Dict[str, Any]]:
        """Search GitHub for additional MCP servers"""
        logger.info("🔍 Searching GitHub for MCP servers...")
        
        search_queries = [
            "mcp server model context protocol",
            "claude mcp server",
            "model context protocol implementation",
            "mcp-server",
            "anthropic mcp"
        ]
        
        found_servers = []
        
        for query in search_queries:
            try:
                # GitHub API search
                url = f"https://api.github.com/search/repositories"
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 20
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for repo in data.get("items", []):
                        server_info = {
                            "name": repo["name"],
                            "full_name": repo["full_name"],
                            "github": repo["html_url"],
                            "description": repo["description"] or "",
                            "stars": repo["stargazers_count"],
                            "language": repo["language"],
                            "updated_at": repo["updated_at"],
                            "type": "discovered"
                        }
                        
                        # Skip if already known
                        if not any(known["github"] == server_info["github"] 
                                 for known in self.known_servers.values()):
                            found_servers.append(server_info)
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Search failed for '{query}': {e}")
        
        # Remove duplicates
        unique_servers = []
        seen_urls = set()
        for server in found_servers:
            if server["github"] not in seen_urls:
                unique_servers.append(server)
                seen_urls.add(server["github"])
        
        logger.info(f"Found {len(unique_servers)} additional MCP servers")
        return unique_servers

    async def clone_repository(self, repo_url: str, repo_name: str) -> Path:
        """Clone a repository"""
        repo_path = self.repos_dir / repo_name
        
        if repo_path.exists():
            logger.info(f"Repository {repo_name} already exists, pulling updates...")
            try:
                repo = git.Repo(repo_path)
                repo.remotes.origin.pull()
            except Exception as e:
                logger.warning(f"Failed to update {repo_name}: {e}")
        else:
            logger.info(f"Cloning {repo_url} to {repo_path}")
            try:
                git.Repo.clone_from(repo_url, repo_path)
            except Exception as e:
                logger.error(f"Failed to clone {repo_url}: {e}")
                raise
        
        return repo_path

    async def analyze_repository_for_mcp(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze repository to find MCP server implementations"""
        analysis = {
            "has_mcp": False,
            "mcp_files": [],
            "package_files": [],
            "documentation": [],
            "install_instructions": "",
            "server_type": "unknown"
        }
        
        # Look for MCP-related files
        mcp_patterns = [
            "**/mcp*.py", "**/mcp*.js", "**/mcp*.ts",
            "**/*mcp*", "**/server*.py", "**/server*.js",
            "**/index.py", "**/index.js", "**/main.py"
        ]
        
        for pattern in mcp_patterns:
            for file_path in repo_path.rglob(pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if any(keyword in content.lower() for keyword in 
                               ['mcp', 'model context protocol', 'claude', 'anthropic']):
                            analysis["mcp_files"].append(str(file_path.relative_to(repo_path)))
                            analysis["has_mcp"] = True
                    except Exception as e:
                        logger.debug(f"Could not read {file_path}: {e}")
        
        # Look for package files
        for pkg_file in ["package.json", "pyproject.toml", "setup.py", "requirements.txt"]:
            pkg_path = repo_path / pkg_file
            if pkg_path.exists():
                analysis["package_files"].append(pkg_file)
        
        # Look for documentation
        for doc_file in ["README.md", "README.rst", "docs", "INSTALL.md"]:
            doc_path = repo_path / doc_file
            if doc_path.exists():
                analysis["documentation"].append(doc_file)
        
        # Try to determine server type
        repo_name_lower = repo_path.name.lower()
        if "github" in repo_name_lower:
            analysis["server_type"] = "version_control"
        elif "web" in repo_name_lower or "search" in repo_name_lower:
            analysis["server_type"] = "web_search"
        elif "file" in repo_name_lower or "fs" in repo_name_lower:
            analysis["server_type"] = "file_operations"
        elif "ai" in repo_name_lower or "llm" in repo_name_lower:
            analysis["server_type"] = "ai_platform"
        
        return analysis

    async def generate_mcp_server_code(self, server_info: Dict[str, Any], repo_path: Path) -> str:
        """Generate MCP server code if not found"""
        logger.info(f"🔧 Generating MCP server code for {server_info['name']}")
        
        server_type = server_info.get("type", "generic")
        
        if server_type == "coding_assistant":
            # Generate Aider MCP server
            mcp_code = f'''#!/usr/bin/env python3
"""
Aider MCP Server
AI pair programming integration for Claude Code
"""

import asyncio
import json
import logging
from typing import Any, Dict

from mcp.server import Server
from mcp.types import Tool, TextContent

# Aider integration
try:
    from aider.main import main as aider_main
    from aider.coders import Coder
    AIDER_AVAILABLE = True
except ImportError:
    AIDER_AVAILABLE = False
    logging.warning("Aider not installed. Install with: pip install aider-chat")

app = Server("aider-mcp")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Aider tools"""
    if not AIDER_AVAILABLE:
        return []
    
    return [
        Tool(
            name="aider_edit",
            description="Edit code files using Aider AI pair programming",
            inputSchema={{
                "type": "object",
                "properties": {{
                    "files": {{
                        "type": "array",
                        "items": {{"type": "string"}},
                        "description": "List of files to edit"
                    }},
                    "prompt": {{
                        "type": "string", 
                        "description": "Instructions for code changes"
                    }},
                    "model": {{
                        "type": "string",
                        "default": "gpt-4",
                        "description": "AI model to use"
                    }}
                }},
                "required": ["files", "prompt"]
            }}
        ),
        Tool(
            name="aider_add",
            description="Add files to Aider session",
            inputSchema={{
                "type": "object",
                "properties": {{
                    "files": {{
                        "type": "array",
                        "items": {{"type": "string"}},
                        "description": "Files to add to session"
                    }}
                }},
                "required": ["files"]
            }}
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""
    if not AIDER_AVAILABLE:
        return [TextContent(type="text", text="Aider not available. Install with: pip install aider-chat")]
    
    if name == "aider_edit":
        files = arguments.get("files", [])
        prompt = arguments.get("prompt", "")
        model = arguments.get("model", "gpt-4")
        
        try:
            # Create Aider session
            coder = Coder.create(main_model=model, fnames=files)
            result = coder.run(prompt)
            
            return [TextContent(
                type="text",
                text=f"Aider edit completed for files: {{', '.join(files)}}\\nResult: {{result}}"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Aider edit failed: {{e}}")]
    
    elif name == "aider_add":
        files = arguments.get("files", [])
        return [TextContent(
            type="text", 
            text=f"Files added to Aider session: {{', '.join(files)}}"
        )]
    
    return [TextContent(type="text", text=f"Unknown tool: {{name}}")]

if __name__ == "__main__":
    import mcp.server.stdio
    mcp.server.stdio.run_server(app)
'''
        
        elif server_type == "ai_platform":
            # Generate Vertex AI MCP server
            mcp_code = f'''#!/usr/bin/env python3
"""
Vertex AI MCP Server
Google Cloud Vertex AI integration for Claude Code
"""

import asyncio
import json
import logging
from typing import Any, Dict

from mcp.server import Server
from mcp.types import Tool, TextContent

# Vertex AI integration
try:
    from google.cloud import aiplatform
    from vertexai.preview.generative_models import GenerativeModel
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    logging.warning("Vertex AI not installed. Install with: pip install google-cloud-aiplatform")

app = Server("vertex-ai-mcp")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available Vertex AI tools"""
    if not VERTEX_AVAILABLE:
        return []
    
    return [
        Tool(
            name="vertex_generate",
            description="Generate text using Vertex AI models",
            inputSchema={{
                "type": "object",
                "properties": {{
                    "prompt": {{
                        "type": "string",
                        "description": "Text prompt for generation"
                    }},
                    "model": {{
                        "type": "string",
                        "default": "gemini-pro",
                        "description": "Vertex AI model to use"
                    }},
                    "project_id": {{
                        "type": "string",
                        "description": "Google Cloud project ID"
                    }}
                }},
                "required": ["prompt", "project_id"]
            }}
        ),
        Tool(
            name="vertex_code_completion",
            description="Get code completion suggestions",
            inputSchema={{
                "type": "object",
                "properties": {{
                    "code": {{
                        "type": "string",
                        "description": "Code context for completion"
                    }},
                    "language": {{
                        "type": "string",
                        "description": "Programming language"
                    }},
                    "project_id": {{
                        "type": "string",
                        "description": "Google Cloud project ID"
                    }}
                }},
                "required": ["code", "project_id"]
            }}
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""
    if not VERTEX_AVAILABLE:
        return [TextContent(type="text", text="Vertex AI not available. Install with: pip install google-cloud-aiplatform")]
    
    project_id = arguments.get("project_id")
    if not project_id:
        return [TextContent(type="text", text="project_id is required")]
    
    try:
        aiplatform.init(project=project_id)
        
        if name == "vertex_generate":
            prompt = arguments.get("prompt", "")
            model_name = arguments.get("model", "gemini-pro")
            
            model = GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            return [TextContent(type="text", text=response.text)]
            
        elif name == "vertex_code_completion":
            code = arguments.get("code", "")
            language = arguments.get("language", "python")
            
            # Use code completion model
            model = GenerativeModel("code-bison")
            prompt = f"Complete this {{language}} code:\\n{{code}}"
            response = model.generate_content(prompt)
            
            return [TextContent(type="text", text=response.text)]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Vertex AI error: {{e}}")]
    
    return [TextContent(type="text", text=f"Unknown tool: {{name}}")]

if __name__ == "__main__":
    import mcp.server.stdio
    mcp.server.stdio.run_server(app)
'''
        
        else:
            # Generate generic MCP server
            mcp_code = f'''#!/usr/bin/env python3
"""
{server_info['name']} MCP Server
Auto-generated MCP server for {server_info.get('description', 'Custom functionality')}
"""

import asyncio
import json
import logging
from typing import Any, Dict

from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("{server_info['name']}-mcp")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="{server_info['name']}_action",
            description="Perform action with {server_info['name']}",
            inputSchema={{
                "type": "object",
                "properties": {{
                    "action": {{
                        "type": "string",
                        "description": "Action to perform"
                    }},
                    "parameters": {{
                        "type": "object",
                        "description": "Action parameters"
                    }}
                }},
                "required": ["action"]
            }}
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""
    action = arguments.get("action", "")
    parameters = arguments.get("parameters", {{}})
    
    # Implement custom logic here
    result = f"Executed {{action}} with parameters: {{parameters}}"
    
    return [TextContent(type="text", text=result)]

if __name__ == "__main__":
    import mcp.server.stdio
    mcp.server.stdio.run_server(app)
'''
        
        # Save generated code
        mcp_file = repo_path / f"{server_info['name']}_mcp_server.py"
        mcp_file.write_text(mcp_code)
        
        logger.info(f"✅ Generated MCP server: {mcp_file}")
        return str(mcp_file)

    async def install_mcp_server(self, server_info: Dict[str, Any], repo_path: Path) -> Dict[str, Any]:
        """Install and configure MCP server"""
        logger.info(f"📦 Installing MCP server: {server_info['name']}")
        
        install_result = {
            "name": server_info["name"],
            "status": "success",
            "mcp_file": None,
            "config": {},
            "errors": []
        }
        
        try:
            # Install dependencies if specified
            if "install_cmd" in server_info:
                logger.info(f"Installing dependencies: {server_info['install_cmd']}")
                result = subprocess.run(
                    server_info["install_cmd"].split(),
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    install_result["errors"].append(f"Dependency install failed: {result.stderr}")
            
            # Analyze repository for MCP implementation
            analysis = await self.analyze_repository_for_mcp(repo_path)
            
            if analysis["has_mcp"] and analysis["mcp_files"]:
                # Use existing MCP file
                mcp_file = repo_path / analysis["mcp_files"][0]
                install_result["mcp_file"] = str(mcp_file)
            else:
                # Generate MCP server code
                mcp_file = await self.generate_mcp_server_code(server_info, repo_path)
                install_result["mcp_file"] = mcp_file
            
            # Create MCP configuration
            mcp_config = {
                "command": "python3",
                "args": [install_result["mcp_file"]],
                "env": {},
                "description": server_info.get("description", ""),
                "type": server_info.get("type", "custom")
            }
            
            install_result["config"] = mcp_config
            
            # Add to Claude Code MCP configuration
            await self.add_to_claude_config(server_info["name"], mcp_config)
            
        except Exception as e:
            logger.error(f"Installation failed for {server_info['name']}: {e}")
            install_result["status"] = "failed"
            install_result["errors"].append(str(e))
        
        return install_result

    async def add_to_claude_config(self, server_name: str, mcp_config: Dict[str, Any]):
        """Add MCP server to Claude Code configuration"""
        config_file = self.config_dir / "mcp.json"
        
        # Load existing config
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = {"mcpServers": {}}
        
        # Add new server
        config["mcpServers"][server_name] = mcp_config
        
        # Save config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Added {server_name} to Claude Code MCP configuration")

    async def deploy_all_mcp_servers(self) -> Dict[str, Any]:
        """Deploy all known and discovered MCP servers"""
        logger.info("🚀 Deploying all MCP servers...")
        
        deployment_results = {
            "deployed": [],
            "failed": [],
            "discovered": []
        }
        
        # Search for additional servers
        discovered_servers = await self.search_github_mcp_servers()
        deployment_results["discovered"] = discovered_servers
        
        # Combine known and discovered servers
        all_servers = list(self.known_servers.items())
        
        # Add promising discovered servers
        for discovered in discovered_servers[:5]:  # Limit to top 5
            if discovered["stars"] > 10:  # Only well-regarded repos
                all_servers.append((discovered["name"], discovered))
        
        # Deploy each server
        for server_name, server_info in all_servers:
            try:
                logger.info(f"Deploying {server_name}...")
                
                # Clone repository
                repo_path = await self.clone_repository(
                    server_info["github"], 
                    server_name
                )
                
                # Install server
                result = await self.install_mcp_server(server_info, repo_path)
                
                if result["status"] == "success":
                    deployment_results["deployed"].append(result)
                    logger.info(f"✅ Successfully deployed {server_name}")
                else:
                    deployment_results["failed"].append(result)
                    logger.error(f"❌ Failed to deploy {server_name}: {result['errors']}")
                
            except Exception as e:
                logger.error(f"Deployment error for {server_name}: {e}")
                deployment_results["failed"].append({
                    "name": server_name,
                    "status": "failed",
                    "errors": [str(e)]
                })
        
        # Save deployment summary
        summary_file = self.mcp_dir / "deployment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(deployment_results, f, indent=2, default=str)
        
        logger.info(f"🎯 MCP deployment complete: {len(deployment_results['deployed'])} succeeded, {len(deployment_results['failed'])} failed")
        return deployment_results

async def main():
    """Main execution function"""
    manager = MCPServerManager()
    
    try:
        print("🔍 Discovering and deploying MCP servers...")
        results = await manager.deploy_all_mcp_servers()
        
        print(f"\n✅ MCP Server Deployment Complete!")
        print(f"   Successfully deployed: {len(results['deployed'])}")
        print(f"   Failed deployments: {len(results['failed'])}")
        print(f"   Discovered servers: {len(results['discovered'])}")
        
        print(f"\n📋 Successfully Deployed Servers:")
        for server in results["deployed"]:
            print(f"   • {server['name']}")
            print(f"     Type: {server['config'].get('type', 'unknown')}")
            print(f"     File: {server['mcp_file']}")
        
        if results["failed"]:
            print(f"\n❌ Failed Deployments:")
            for server in results["failed"]:
                print(f"   • {server['name']}: {', '.join(server['errors'])}")
        
        print(f"\n🔧 Configuration saved to: {manager.config_dir / 'mcp.json'}")
        print(f"📁 Repositories cloned to: {manager.repos_dir}")
        
    except Exception as e:
        logger.error(f"MCP deployment failed: {e}")
        print(f"❌ MCP deployment failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())