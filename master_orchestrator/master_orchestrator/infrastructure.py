"""Infrastructure Management for Master Orchestrator."""

import asyncio
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json

import structlog
from pydantic import BaseModel, Field

from .config import InfrastructureConfig

logger = structlog.get_logger()


class HardwareNode(BaseModel):
    """Hardware node model."""
    
    id: str = Field(description="Node identifier")
    name: str = Field(description="Node name")
    type: str = Field(description="Hardware type")
    ip_address: str = Field(description="IP address")
    status: str = Field(default="unknown")  # online, offline, error
    cpu_usage: float = Field(default=0.0)
    memory_usage: float = Field(default=0.0)
    disk_usage: float = Field(default=0.0)
    specifications: Dict[str, Any] = Field(default_factory=dict)
    last_checked: Optional[str] = Field(default=None)


class ContainerInfo(BaseModel):
    """Container information model."""
    
    id: str = Field(description="Container ID")
    name: str = Field(description="Container name")
    image: str = Field(description="Container image")
    status: str = Field(description="Container status")
    ports: List[str] = Field(default_factory=list)
    created: str = Field(description="Creation timestamp")


class InfrastructureManager:
    """
    Infrastructure Manager for Master Orchestrator.
    
    Manages Kubernetes clusters, Docker containers, hardware nodes,
    and provides infrastructure automation via Ansible and Terraform.
    """
    
    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self.logger = structlog.get_logger("infrastructure")
        
        # State tracking
        self.connected_nodes: Dict[str, HardwareNode] = {}
        self.running_containers: Dict[str, ContainerInfo] = {}
        self.k8s_available = False
        self.docker_available = False
        
        # Resource limits
        self.resource_limits = {
            "cpu_threshold": 0.8,
            "memory_threshold": 0.8,
            "disk_threshold": 0.9
        }
    
    async def initialize(self) -> None:
        """Initialize infrastructure management."""
        self.logger.info("Initializing Infrastructure Manager")
        
        try:
            # Check Docker availability
            await self._check_docker()
            
            # Check Kubernetes availability
            await self._check_kubernetes()
            
            # Discover hardware nodes
            await self._discover_hardware_nodes()
            
            self.logger.info(
                "Infrastructure Manager initialized",
                docker_available=self.docker_available,
                k8s_available=self.k8s_available,
                nodes_discovered=len(self.connected_nodes)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Infrastructure Manager: {e}")
            raise
    
    async def _check_docker(self) -> None:
        """Check Docker availability."""
        try:
            result = await self._run_command(["docker", "version", "--format", "json"])
            if result.returncode == 0:
                self.docker_available = True
                self.logger.info("Docker is available")
            else:
                self.logger.warning("Docker is not available")
        except Exception as e:
            self.logger.warning(f"Docker check failed: {e}")
    
    async def _check_kubernetes(self) -> None:
        """Check Kubernetes availability."""
        try:
            result = await self._run_command(["kubectl", "version", "--client", "--output=json"])
            if result.returncode == 0:
                self.k8s_available = True
                self.logger.info("Kubernetes is available")
            else:
                self.logger.warning("Kubernetes is not available")
        except Exception as e:
            self.logger.warning(f"Kubernetes check failed: {e}")
    
    async def _discover_hardware_nodes(self) -> None:
        """Discover available hardware nodes."""
        # Add localhost as primary node
        localhost_node = HardwareNode(
            id="localhost",
            name="Local Machine",
            type="mac_studio",  # Assuming based on user's setup
            ip_address="127.0.0.1",
            status="online"
        )
        self.connected_nodes["localhost"] = localhost_node
        
        # Discover configured Mac Studios
        for i, mac_studio in enumerate(self.config.mac_studios):
            node = HardwareNode(
                id=f"mac_studio_{i}",
                name=f"Mac Studio {i+1}",
                type="mac_studio",
                ip_address=mac_studio,
                specifications={
                    "ram": "512GB",
                    "cpu": "Apple Silicon",
                    "gpu": "Integrated"
                }
            )
            await self._ping_node(node)
            self.connected_nodes[node.id] = node
        
        # Discover configured Mac Minis
        for i, mac_mini in enumerate(self.config.mac_minis):
            node = HardwareNode(
                id=f"mac_mini_{i}",
                name=f"Mac Mini {i+1}",
                type="mac_mini",
                ip_address=mac_mini,
                specifications={
                    "ram": "64GB",
                    "cpu": "Apple M4 Max",
                    "gpu": "Integrated"
                }
            )
            await self._ping_node(node)
            self.connected_nodes[node.id] = node
        
        # Discover NAS systems
        for i, nas_system in enumerate(self.config.nas_systems):
            node = HardwareNode(
                id=f"nas_{i}",
                name=f"NAS System {i+1}",
                type="nas",
                ip_address=nas_system,
                specifications={
                    "storage": "1PB+" if i == 0 else "High-Speed SSD"
                }
            )
            await self._ping_node(node)
            self.connected_nodes[node.id] = node
    
    async def _ping_node(self, node: HardwareNode) -> None:
        """Ping a node to check availability."""
        try:
            result = await self._run_command(["ping", "-c", "1", "-W", "2000", node.ip_address])
            if result.returncode == 0:
                node.status = "online"
            else:
                node.status = "offline"
        except Exception:
            node.status = "error"
    
    async def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a system command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
    
    async def deploy_container(
        self,
        image: str,
        name: str,
        ports: Optional[Dict[int, int]] = None,
        environment: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None
    ) -> bool:
        """Deploy a Docker container."""
        if not self.docker_available:
            self.logger.error("Docker not available for container deployment")
            return False
        
        command = ["docker", "run", "-d", "--name", name]
        
        # Add port mappings
        if ports:
            for host_port, container_port in ports.items():
                command.extend(["-p", f"{host_port}:{container_port}"])
        
        # Add environment variables
        if environment:
            for key, value in environment.items():
                command.extend(["-e", f"{key}={value}"])
        
        # Add volume mounts
        if volumes:
            for host_path, container_path in volumes.items():
                command.extend(["-v", f"{host_path}:{container_path}"])
        
        command.append(image)
        
        try:
            result = await self._run_command(command)
            if result.returncode == 0:
                container_id = result.stdout.decode().strip()
                self.logger.info(f"Container deployed successfully", name=name, id=container_id)
                
                # Add to running containers
                container_info = ContainerInfo(
                    id=container_id,
                    name=name,
                    image=image,
                    status="running",
                    ports=[f"{hp}:{cp}" for hp, cp in (ports or {}).items()],
                    created="now"
                )
                self.running_containers[container_id] = container_info
                
                return True
            else:
                error = result.stderr.decode()
                self.logger.error(f"Container deployment failed: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Container deployment error: {e}")
            return False
    
    async def stop_container(self, container_id: str) -> bool:
        """Stop a Docker container."""
        if not self.docker_available:
            return False
        
        try:
            result = await self._run_command(["docker", "stop", container_id])
            if result.returncode == 0:
                self.logger.info(f"Container stopped", id=container_id)
                
                # Remove from running containers
                if container_id in self.running_containers:
                    del self.running_containers[container_id]
                
                return True
            else:
                error = result.stderr.decode()
                self.logger.error(f"Failed to stop container: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Container stop error: {e}")
            return False
    
    async def get_container_logs(self, container_id: str, lines: int = 100) -> str:
        """Get container logs."""
        if not self.docker_available:
            return ""
        
        try:
            result = await self._run_command([
                "docker", "logs", "--tail", str(lines), container_id
            ])
            if result.returncode == 0:
                return result.stdout.decode()
            else:
                return f"Error getting logs: {result.stderr.decode()}"
                
        except Exception as e:
            return f"Error getting logs: {e}"
    
    async def deploy_arangodb(self) -> bool:
        """Deploy ArangoDB container."""
        return await self.deploy_container(
            image="arangodb/arangodb:latest",
            name="master-orchestrator-arangodb",
            ports={8529: 8529},
            environment={
                "ARANGO_ROOT_PASSWORD": "orchestrator123",
                "ARANGO_NO_AUTH": "0"
            },
            volumes={
                str(Path.home() / "arangodb_data"): "/var/lib/arangodb3"
            }
        )
    
    async def deploy_monitoring_stack(self) -> bool:
        """Deploy monitoring stack (Prometheus + Grafana)."""
        # Deploy Prometheus
        prometheus_success = await self.deploy_container(
            image="prom/prometheus:latest",
            name="master-orchestrator-prometheus",
            ports={9090: 9090},
            volumes={
                str(Path.cwd() / "monitoring" / "prometheus.yml"): "/etc/prometheus/prometheus.yml"
            }
        )
        
        # Deploy Grafana
        grafana_success = await self.deploy_container(
            image="grafana/grafana:latest",
            name="master-orchestrator-grafana",
            ports={3000: 3000},
            environment={
                "GF_SECURITY_ADMIN_PASSWORD": "orchestrator123"
            }
        )
        
        return prometheus_success and grafana_success
    
    async def get_node_metrics(self, node_id: str) -> Dict[str, float]:
        """Get resource metrics for a node."""
        if node_id not in self.connected_nodes:
            return {}
        
        node = self.connected_nodes[node_id]
        
        # For localhost, get real metrics
        if node_id == "localhost":
            try:
                # Get CPU usage
                cpu_result = await self._run_command([
                    "top", "-l", "1", "-n", "0"
                ])
                # Parse CPU usage from top output
                # This is a simplified version
                
                # Get memory usage
                memory_result = await self._run_command([
                    "vm_stat"
                ])
                # Parse memory usage from vm_stat output
                
                # For now, return mock data
                return {
                    "cpu_usage": 0.25,
                    "memory_usage": 0.45,
                    "disk_usage": 0.60
                }
                
            except Exception:
                return {}
        
        # For remote nodes, would need SSH or monitoring agents
        return {}
    
    async def optimize_resources(self) -> None:
        """Optimize resource allocation across infrastructure."""
        self.logger.debug("Optimizing infrastructure resources")
        
        # Update node metrics
        for node_id in self.connected_nodes:
            metrics = await self.get_node_metrics(node_id)
            if metrics:
                node = self.connected_nodes[node_id]
                node.cpu_usage = metrics.get("cpu_usage", 0.0)
                node.memory_usage = metrics.get("memory_usage", 0.0)
                node.disk_usage = metrics.get("disk_usage", 0.0)
        
        # Check for resource constraints
        overloaded_nodes = []
        for node_id, node in self.connected_nodes.items():
            if (node.cpu_usage > self.resource_limits["cpu_threshold"] or
                node.memory_usage > self.resource_limits["memory_threshold"]):
                overloaded_nodes.append(node_id)
        
        if overloaded_nodes:
            self.logger.warning(f"Overloaded nodes detected: {overloaded_nodes}")
            # Could trigger load balancing or scaling actions
    
    async def run_ansible_playbook(self, playbook_path: Path, inventory: Optional[str] = None) -> bool:
        """Run an Ansible playbook."""
        if not playbook_path.exists():
            self.logger.error(f"Ansible playbook not found: {playbook_path}")
            return False
        
        command = ["ansible-playbook", str(playbook_path)]
        if inventory:
            command.extend(["-i", inventory])
        
        try:
            result = await self._run_command(command)
            if result.returncode == 0:
                self.logger.info(f"Ansible playbook executed successfully: {playbook_path}")
                return True
            else:
                error = result.stderr.decode()
                self.logger.error(f"Ansible playbook failed: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Ansible execution error: {e}")
            return False
    
    async def apply_terraform(self, terraform_dir: Path) -> bool:
        """Apply Terraform configuration."""
        if not terraform_dir.exists():
            self.logger.error(f"Terraform directory not found: {terraform_dir}")
            return False
        
        try:
            # Initialize Terraform
            init_result = await self._run_command([
                "terraform", "init"
            ])
            
            if init_result.returncode != 0:
                self.logger.error("Terraform init failed")
                return False
            
            # Apply configuration
            apply_result = await self._run_command([
                "terraform", "apply", "-auto-approve"
            ])
            
            if apply_result.returncode == 0:
                self.logger.info("Terraform applied successfully")
                return True
            else:
                error = apply_result.stderr.decode()
                self.logger.error(f"Terraform apply failed: {error}")
                return False
                
        except Exception as e:
            self.logger.error(f"Terraform execution error: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown infrastructure manager."""
        self.logger.info("Shutting down Infrastructure Manager")
        
        # Stop managed containers
        for container_id in list(self.running_containers.keys()):
            await self.stop_container(container_id)
        
        self.connected_nodes.clear()
        self.running_containers.clear()
        
        self.logger.info("Infrastructure Manager shutdown complete")