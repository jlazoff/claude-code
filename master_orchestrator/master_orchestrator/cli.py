"""Command Line Interface for Master Orchestrator."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import typer
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .core import MasterOrchestrator
from .config import OrchestratorConfig
from .agents import AgentTask

app = typer.Typer(
    name="master-orchestrator",
    help="Agentic Multi-Project Orchestration System",
    rich_markup_mode="rich"
)

console = Console()


def get_config(config_file: Optional[Path] = None) -> OrchestratorConfig:
    """Get configuration from file or environment."""
    if config_file and config_file.exists():
        return OrchestratorConfig.from_file(config_file)
    else:
        return OrchestratorConfig.from_env()


@app.command()
def start(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run as daemon"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging")
):
    """Start the Master Orchestrator system."""
    
    # Setup logging
    if debug:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer()
            ]
        )
    
    config = get_config(config_file)
    
    async def _start():
        orchestrator = MasterOrchestrator(config)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            init_task = progress.add_task("Initializing Master Orchestrator...", total=None)
            
            try:
                await orchestrator.initialize()
                progress.update(init_task, description="✅ Initialization complete")
                
                start_task = progress.add_task("Starting system components...", total=None)
                await orchestrator.start()
                progress.update(start_task, description="✅ System started successfully")
                
                # Display system status
                status = await orchestrator.get_status()
                
                panel = Panel.fit(
                    f"""[green]Master Orchestrator Started Successfully![/green]
                    
Status: {status.status}
Active Agents: {status.active_agents}
Connected Repositories: {status.repositories_connected}
Hardware Nodes: {status.hardware_nodes}

Access the web UI at: http://{config.api_host}:{config.api_port}
Press Ctrl+C to stop the system
""",
                    title="🚀 Master Orchestrator",
                    border_style="green"
                )
                
                console.print(panel)
                
                if not daemon:
                    # Run interactively
                    try:
                        while True:
                            await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        console.print("\n[yellow]Stopping Master Orchestrator...[/yellow]")
                        await orchestrator.stop()
                        console.print("[green]✅ System stopped successfully[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Failed to start Master Orchestrator: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(_start())


@app.command()
def status(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path")
):
    """Show system status."""
    
    config = get_config(config_file)
    
    async def _status():
        orchestrator = MasterOrchestrator(config)
        
        try:
            await orchestrator.initialize()
            status = await orchestrator.get_status()
            
            # Create status table
            table = Table(title="Master Orchestrator Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="white")
            
            table.add_row("System", status.status, f"Uptime: {status.uptime}")
            table.add_row("Agents", str(status.active_agents), "Active agents")
            table.add_row("Repositories", str(status.repositories_connected), "Connected repositories")
            table.add_row("Hardware", str(status.hardware_nodes), "Connected nodes")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to get status: {e}[/red]")
        finally:
            await orchestrator.stop()
    
    asyncio.run(_status())


@app.command()
def analyze_repo(
    path: Path = typer.Argument(..., help="Path to repository to analyze"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path")
):
    """Analyze a repository."""
    
    if not path.exists():
        console.print(f"[red]❌ Repository path does not exist: {path}[/red]")
        sys.exit(1)
    
    config = get_config(config_file)
    
    async def _analyze():
        orchestrator = MasterOrchestrator(config)
        
        try:
            await orchestrator.initialize()
            await orchestrator.start()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Analyzing repository: {path.name}...", total=None)
                
                result = await orchestrator.execute_command("analyze_repository", {
                    "path": str(path)
                })
                
                progress.update(task, description="✅ Analysis complete")
                
                if result.get("success"):
                    analysis = result["analysis"]
                    
                    panel = Panel.fit(
                        f"""[green]Repository Analysis Complete![/green]
                        
Repository: {path.name}
Path: {path}

Analysis Results:
{analysis}
""",
                        title="📊 Repository Analysis",
                        border_style="blue"
                    )
                    
                    console.print(panel)
                else:
                    console.print(f"[red]❌ Analysis failed: {result.get('error', 'Unknown error')}[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Failed to analyze repository: {e}[/red]")
        finally:
            await orchestrator.stop()
    
    asyncio.run(_analyze())


@app.command()
def list_agents(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path")
):
    """List all agents in the system."""
    
    config = get_config(config_file)
    
    async def _list_agents():
        orchestrator = MasterOrchestrator(config)
        
        try:
            await orchestrator.initialize()
            
            if orchestrator.agent_framework:
                agents = orchestrator.agent_framework.agents
                
                if not agents:
                    console.print("[yellow]No agents found[/yellow]")
                    return
                
                table = Table(title="System Agents")
                table.add_column("ID", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Capabilities", style="white")
                
                for agent_id, agent in agents.items():
                    capabilities = ", ".join([cap.name for cap in agent.capabilities])
                    table.add_row(
                        agent_id[:8] + "...",
                        agent.name,
                        agent.status,
                        capabilities
                    )
                
                console.print(table)
            else:
                console.print("[red]❌ Agent framework not initialized[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Failed to list agents: {e}[/red]")
        finally:
            await orchestrator.stop()
    
    asyncio.run(_list_agents())


@app.command()
def create_agent(
    agent_type: str = typer.Argument(..., help="Type of agent to create"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Agent name"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path")
):
    """Create a new agent."""
    
    config = get_config(config_file)
    
    async def _create_agent():
        orchestrator = MasterOrchestrator(config)
        
        try:
            await orchestrator.initialize()
            await orchestrator.start()
            
            agent_config = {}
            if name:
                agent_config["name"] = name
            
            result = await orchestrator.execute_command("create_agent", {
                "type": agent_type,
                "config": agent_config
            })
            
            if result.get("success"):
                agent_id = result["agent_id"]
                console.print(f"[green]✅ Agent created successfully![/green]")
                console.print(f"Agent ID: {agent_id}")
                console.print(f"Type: {agent_type}")
                console.print(f"Name: {name or 'auto-generated'}")
            else:
                console.print(f"[red]❌ Failed to create agent: {result.get('error', 'Unknown error')}[/red]")
                
        except Exception as e:
            console.print(f"[red]❌ Failed to create agent: {e}[/red]")
        finally:
            await orchestrator.stop()
    
    asyncio.run(_create_agent())


@app.command()
def init_config(
    output: Path = typer.Option(Path("config.yaml"), "--output", "-o", help="Output configuration file")
):
    """Initialize a configuration file."""
    
    config = OrchestratorConfig.from_env()
    
    try:
        config.to_file(output)
        console.print(f"[green]✅ Configuration file created: {output}[/green]")
        console.print(f"Edit the file to customize your setup")
    except Exception as e:
        console.print(f"[red]❌ Failed to create configuration file: {e}[/red]")


@app.command()
def version():
    """Show version information."""
    
    from . import __version__
    
    panel = Panel.fit(
        f"""[bold green]Master Orchestrator[/bold green]
        
Version: {__version__}
Description: Agentic Multi-Project Orchestration System

Components:
• DSPY-based Agent Framework
• ArangoDB Knowledge Graph
• Multi-LLM Provider Support  
• Kubernetes/Docker Infrastructure
• 24/7 Autonomous Operation

Hardware Support:
• Mac Studios, Mac Minis
• NAS Systems (Synology, Asustor)
• Cloud Integration

[dim]Built for enterprise-scale AI orchestration[/dim]
""",
        title="🚀 Master Orchestrator",
        border_style="blue"
    )
    
    console.print(panel)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()