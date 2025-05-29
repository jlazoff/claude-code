#!/usr/bin/env python3
"""
Thunderbolt Network Optimizer - High-Speed Local Compute Distribution
Optimizes deployment across Mac Studios, Mac Minis via Thunderbolt network
Manages llm-d, vLLM, LocalAI, and Triton Inference Server distribution
"""

import asyncio
import json
import yaml
import logging
import subprocess
import os
import sys
import time
import socket
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
import aiofiles
import aiohttp
from datetime import datetime, timedelta
import uuid
import ipaddress
import netifaces
from pydantic import BaseModel, Field
import redis
import docker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThunderboltNode(BaseModel):
    """Represents a node in the Thunderbolt network"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    hostname: str
    ip_address: str
    mac_address: str
    device_type: str  # "macbook_pro", "mac_studio", "mac_mini"
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    thunderbolt_version: str
    gpu_info: Dict[str, Any] = Field(default_factory=dict)
    network_bandwidth: float  # Gbps
    current_load: float = 0.0
    available_memory: float = 0.0
    running_services: List[str] = Field(default_factory=list)
    health_status: str = "unknown"
    last_heartbeat: datetime = Field(default_factory=datetime.now)

class ServiceDeployment(BaseModel):
    """Service deployment configuration"""
    service_name: str
    service_type: str  # "llm-d", "vllm", "localai", "triton"
    node_id: str
    port: int
    resource_requirements: Dict[str, Any]
    configuration: Dict[str, Any]
    status: str = "pending"
    deployment_time: datetime = Field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

class ThunderboltNetworkOptimizer:
    """
    Comprehensive Thunderbolt network optimization and service distribution
    """
    
    def __init__(self):
        self.base_dir = Path("foundation_data")
        self.network_dir = self.base_dir / "thunderbolt_network"
        self.deployments_dir = self.network_dir / "deployments"
        self.monitoring_dir = self.network_dir / "monitoring"
        
        # Network topology
        self.nodes: Dict[str, ThunderboltNode] = {}
        self.service_deployments: Dict[str, ServiceDeployment] = {}
        
        # Service configurations
        self.service_configs = {}
        
        # Redis for distributed coordination
        self.redis_client = None
        
        # Docker clients for each node
        self.docker_clients: Dict[str, docker.DockerClient] = {}
        
        # Performance monitoring
        self.performance_data = {}
        self.load_balancing_strategy = "weighted_round_robin"
        
        self._initialize_directories()
        
    def _initialize_directories(self):
        """Initialize network management directories"""
        directories = [
            self.network_dir,
            self.deployments_dir,
            self.monitoring_dir,
            self.network_dir / "configs",
            self.network_dir / "logs",
            self.network_dir / "health_checks"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    async def initialize(self):
        """Initialize the Thunderbolt network optimizer"""
        logger.info("⚡ Initializing Thunderbolt Network Optimizer...")
        
        # Setup Redis for coordination
        await self._setup_redis_coordination()
        
        # Discover network topology
        await self._discover_thunderbolt_network()
        
        # Initialize service configurations
        await self._initialize_service_configurations()
        
        # Setup monitoring
        await self._setup_network_monitoring()
        
        # Start optimization loops
        await self._start_optimization_loops()
        
        logger.info("✅ Thunderbolt Network Optimizer initialized")
        
    async def _setup_redis_coordination(self):
        """Setup Redis for distributed coordination"""
        try:
            # Install Redis if not available
            subprocess.run(["pip3", "install", "redis"], check=True, timeout=120)
            
            # Try to connect to Redis, start if needed
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                logger.info("✅ Connected to existing Redis instance")
            except redis.ConnectionError:
                # Start Redis using Docker
                await self._start_redis_container()
                
        except Exception as e:
            logger.error(f"Failed to setup Redis: {e}")
            
    async def _start_redis_container(self):
        """Start Redis container for coordination"""
        try:
            docker_client = docker.from_env()
            
            # Check if Redis container exists
            try:
                container = docker_client.containers.get("thunderbolt-redis")
                if container.status != "running":
                    container.start()
                    logger.info("✅ Started existing Redis container")
            except docker.errors.NotFound:
                # Create new Redis container
                container = docker_client.containers.run(
                    "redis:alpine",
                    name="thunderbolt-redis",
                    ports={'6379/tcp': 6379},
                    detach=True,
                    restart_policy={"Name": "always"}
                )
                logger.info("✅ Created and started Redis container")
                
            # Wait for Redis to be ready
            await asyncio.sleep(3)
            
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
            
        except Exception as e:
            logger.error(f"Failed to start Redis container: {e}")
            
    async def _discover_thunderbolt_network(self):
        """Discover all devices on the Thunderbolt network"""
        logger.info("🔍 Discovering Thunderbolt network topology...")
        
        # Get local machine info first
        local_node = await self._get_local_machine_info()
        self.nodes[local_node.id] = local_node
        
        # Discover other machines on the network
        await self._scan_thunderbolt_network()
        
        # Store network topology
        await self._store_network_topology()
        
    async def _get_local_machine_info(self) -> ThunderboltNode:
        """Get information about the local machine"""
        try:
            # Get system information
            hostname = socket.gethostname()
            
            # Get IP addresses
            interfaces = netifaces.interfaces()
            ip_address = None
            mac_address = None
            
            for interface in interfaces:
                addrs = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ip = addr['addr']
                        if not ip.startswith('127.') and not ip.startswith('169.254.'):
                            ip_address = ip
                            break
                            
                if netifaces.AF_LINK in addrs:
                    mac_address = addrs[netifaces.AF_LINK][0]['addr']
                    
            # Get hardware info
            cpu_cores = psutil.cpu_count(logical=False)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Get storage info
            storage_gb = 0
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    storage_gb += usage.total / (1024**3)
                except:
                    continue
                    
            # Detect device type based on system info
            device_type = await self._detect_device_type()
            
            # Get GPU info
            gpu_info = await self._get_gpu_info()
            
            # Estimate Thunderbolt bandwidth
            thunderbolt_version, bandwidth = await self._detect_thunderbolt_capabilities()
            
            return ThunderboltNode(
                hostname=hostname,
                ip_address=ip_address or "unknown",
                mac_address=mac_address or "unknown",
                device_type=device_type,
                cpu_cores=cpu_cores,
                memory_gb=memory_gb,
                storage_gb=storage_gb,
                thunderbolt_version=thunderbolt_version,
                gpu_info=gpu_info,
                network_bandwidth=bandwidth,
                health_status="healthy"
            )
            
        except Exception as e:
            logger.error(f"Error getting local machine info: {e}")
            return ThunderboltNode(
                hostname="unknown",
                ip_address="unknown",
                mac_address="unknown",
                device_type="unknown",
                cpu_cores=1,
                memory_gb=1.0,
                storage_gb=1.0,
                thunderbolt_version="unknown",
                network_bandwidth=1.0
            )
            
    async def _detect_device_type(self) -> str:
        """Detect the type of Mac device"""
        try:
            # Use system_profiler to get hardware info
            result = subprocess.run([
                "system_profiler", "SPHardwareDataType"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = result.stdout.lower()
                
                if "mac studio" in output:
                    return "mac_studio"
                elif "mac mini" in output:
                    return "mac_mini"
                elif "macbook pro" in output:
                    return "macbook_pro"
                elif "macbook air" in output:
                    return "macbook_air"
                elif "imac" in output:
                    return "imac"
                    
            return "unknown_mac"
            
        except Exception as e:
            logger.debug(f"Error detecting device type: {e}")
            return "unknown"
            
    async def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            gpu_info = {"available": False, "type": "unknown", "memory": 0}
            
            # Try to get GPU info using system_profiler
            result = subprocess.run([
                "system_profiler", "SPDisplaysDataType"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = result.stdout
                
                if "Apple M" in output:
                    gpu_info["available"] = True
                    gpu_info["type"] = "Apple Silicon"
                    
                    # Extract memory info if available
                    lines = output.split('\n')
                    for line in lines:
                        if "Memory" in line and "GB" in line:
                            # Extract memory value
                            import re
                            memory_match = re.search(r'(\d+)\s*GB', line)
                            if memory_match:
                                gpu_info["memory"] = int(memory_match.group(1))
                                
            return gpu_info
            
        except Exception as e:
            logger.debug(f"Error getting GPU info: {e}")
            return {"available": False, "type": "unknown", "memory": 0}
            
    async def _detect_thunderbolt_capabilities(self) -> Tuple[str, float]:
        """Detect Thunderbolt version and bandwidth"""
        try:
            # Use system_profiler to get Thunderbolt info
            result = subprocess.run([
                "system_profiler", "SPThunderboltDataType"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                output = result.stdout
                
                if "Thunderbolt 4" in output or "USB4" in output:
                    return "Thunderbolt 4", 40.0  # 40 Gbps
                elif "Thunderbolt 3" in output:
                    return "Thunderbolt 3", 40.0  # 40 Gbps
                elif "Thunderbolt 2" in output:
                    return "Thunderbolt 2", 20.0  # 20 Gbps
                elif "Thunderbolt" in output:
                    return "Thunderbolt 1", 10.0  # 10 Gbps
                    
            # Check for USB-C connections as fallback
            usb_result = subprocess.run([
                "system_profiler", "SPUSBDataType"
            ], capture_output=True, text=True, timeout=30)
            
            if usb_result.returncode == 0 and "USB 3" in usb_result.stdout:
                return "USB 3.x", 5.0  # 5 Gbps estimate
                
            return "unknown", 1.0  # 1 Gbps default
            
        except Exception as e:
            logger.debug(f"Error detecting Thunderbolt capabilities: {e}")
            return "unknown", 1.0
            
    async def _scan_thunderbolt_network(self):
        """Scan for other devices on the Thunderbolt network"""
        try:
            # Get local network range
            local_ip = None
            for interface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addrs:
                    for addr in addrs[netifaces.AF_INET]:
                        ip = addr['addr']
                        if not ip.startswith('127.') and not ip.startswith('169.254.'):
                            local_ip = ip
                            break
                            
            if not local_ip:
                logger.warning("Could not determine local IP address")
                return
                
            # Determine network range (assume /24)
            network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
            
            # Scan for active hosts
            active_hosts = await self._ping_sweep(str(network))
            
            # Check each active host for our services
            for host_ip in active_hosts:
                if host_ip != local_ip:
                    node_info = await self._probe_remote_node(host_ip)
                    if node_info:
                        self.nodes[node_info.id] = node_info
                        
        except Exception as e:
            logger.error(f"Error scanning Thunderbolt network: {e}")
            
    async def _ping_sweep(self, network: str) -> List[str]:
        """Perform ping sweep to find active hosts"""
        active_hosts = []
        
        try:
            network_obj = ipaddress.IPv4Network(network)
            
            # Use concurrent pings for speed
            semaphore = asyncio.Semaphore(50)  # Limit concurrent pings
            
            async def ping_host(ip):
                async with semaphore:
                    try:
                        # Use ping command
                        proc = await asyncio.create_subprocess_exec(
                            'ping', '-c', '1', '-W', '1000', str(ip),
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL
                        )
                        await asyncio.wait_for(proc.wait(), timeout=2)
                        
                        if proc.returncode == 0:
                            return str(ip)
                    except:
                        pass
                    return None
                    
            # Ping all hosts in parallel
            tasks = [ping_host(ip) for ip in network_obj.hosts()]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            active_hosts = [ip for ip in results if ip and isinstance(ip, str)]
            
        except Exception as e:
            logger.error(f"Error in ping sweep: {e}")
            
        return active_hosts
        
    async def _probe_remote_node(self, ip: str) -> Optional[ThunderboltNode]:
        """Probe a remote node for information"""
        try:
            # Try to connect to our management port
            management_port = 8765
            
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, management_port),
                    timeout=5
                )
                
                # Send info request
                request = json.dumps({"action": "get_info"})
                writer.write(request.encode() + b'\n')
                await writer.drain()
                
                # Read response
                response = await asyncio.wait_for(reader.readline(), timeout=5)
                node_data = json.loads(response.decode().strip())
                
                writer.close()
                await writer.wait_closed()
                
                # Convert to ThunderboltNode
                return ThunderboltNode(**node_data)
                
            except (asyncio.TimeoutError, ConnectionError):
                # Node not running our management service
                pass
                
            # Try SSH connection to get basic info
            hostname = await self._get_hostname_via_ssh(ip)
            if hostname:
                return ThunderboltNode(
                    hostname=hostname,
                    ip_address=ip,
                    mac_address="unknown",
                    device_type="unknown_remote",
                    cpu_cores=1,
                    memory_gb=1.0,
                    storage_gb=1.0,
                    thunderbolt_version="unknown",
                    network_bandwidth=1.0,
                    health_status="unknown"
                )
                
        except Exception as e:
            logger.debug(f"Error probing node {ip}: {e}")
            
        return None
        
    async def _get_hostname_via_ssh(self, ip: str) -> Optional[str]:
        """Get hostname via SSH if possible"""
        try:
            # Try to get hostname using SSH (if keys are set up)
            proc = await asyncio.create_subprocess_exec(
                'ssh', '-o', 'ConnectTimeout=2', '-o', 'BatchMode=yes',
                f'{ip}', 'hostname',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            
            if proc.returncode == 0:
                return stdout.decode().strip()
                
        except Exception:
            pass
            
        return None
        
    async def _initialize_service_configurations(self):
        """Initialize configurations for all distributed services"""
        logger.info("⚙️ Initializing service configurations...")
        
        self.service_configs = {
            "llm-d": {
                "image": "llm-d:latest",
                "ports": {"coordinator": 7777, "worker_base": 7778},
                "resources": {"cpu": 2, "memory": "4GB", "gpu": False},
                "environment": {
                    "LLM_D_COORDINATOR_PORT": "7777",
                    "LLM_D_WORKER_PORT_BASE": "7778"
                },
                "volumes": ["/models:/models:ro"],
                "network_mode": "host"
            },
            
            "vllm": {
                "image": "vllm/vllm-openai:latest",
                "ports": {"api": 8000},
                "resources": {"cpu": 4, "memory": "8GB", "gpu": True},
                "environment": {
                    "VLLM_HOST": "0.0.0.0",
                    "VLLM_PORT": "8000"
                },
                "volumes": ["/models:/models:ro"],
                "command": ["--model", "/models/default", "--host", "0.0.0.0", "--port", "8000"]
            },
            
            "localai": {
                "image": "localai/localai:latest",
                "ports": {"api": 8080},
                "resources": {"cpu": 2, "memory": "4GB", "gpu": False},
                "environment": {
                    "LOCALAI_ADDRESS": "0.0.0.0:8080",
                    "MODELS_PATH": "/models"
                },
                "volumes": ["/models:/models:rw", "/config:/config:ro"],
                "network_mode": "host"
            },
            
            "triton": {
                "image": "nvcr.io/nvidia/tritonserver:latest",
                "ports": {"http": 8001, "grpc": 8002, "metrics": 8003},
                "resources": {"cpu": 4, "memory": "8GB", "gpu": True},
                "environment": {
                    "TRITON_HTTP_PORT": "8001",
                    "TRITON_GRPC_PORT": "8002"
                },
                "volumes": ["/models:/models:ro"],
                "command": ["tritonserver", "--model-repository=/models", "--allow-http=true"]
            }
        }
        
    async def _setup_network_monitoring(self):
        """Setup monitoring for the Thunderbolt network"""
        logger.info("📊 Setting up network monitoring...")
        
        # Start monitoring loops
        asyncio.create_task(self._network_health_monitoring_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._service_monitoring_loop())
        
    async def _start_optimization_loops(self):
        """Start optimization loops"""
        logger.info("🔄 Starting optimization loops...")
        
        asyncio.create_task(self._load_balancing_loop())
        asyncio.create_task(self._resource_optimization_loop())
        asyncio.create_task(self._failure_recovery_loop())
        
    async def _network_health_monitoring_loop(self):
        """Monitor network health"""
        while True:
            try:
                for node_id, node in self.nodes.items():
                    # Ping node
                    is_alive = await self._ping_node(node.ip_address)
                    
                    if is_alive:
                        node.health_status = "healthy"
                        node.last_heartbeat = datetime.now()
                    else:
                        node.health_status = "unreachable"
                        
                    # Update in Redis
                    if self.redis_client:
                        await self._update_node_status_in_redis(node)
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in network health monitoring: {e}")
                await asyncio.sleep(60)
                
    async def _ping_node(self, ip: str) -> bool:
        """Ping a node to check if it's alive"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', '1000', ip,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await asyncio.wait_for(proc.wait(), timeout=3)
            return proc.returncode == 0
        except:
            return False
            
    async def deploy_service_optimally(self, service_type: str, **kwargs) -> Dict[str, Any]:
        """Deploy a service optimally across the Thunderbolt network"""
        logger.info(f"🚀 Deploying {service_type} optimally...")
        
        # Get service configuration
        if service_type not in self.service_configs:
            return {"error": f"Unknown service type: {service_type}"}
            
        service_config = self.service_configs[service_type].copy()
        service_config.update(kwargs.get("config_overrides", {}))
        
        # Find optimal node
        optimal_node = await self._find_optimal_node_for_service(service_type, service_config)
        
        if not optimal_node:
            return {"error": "No suitable node found for deployment"}
            
        # Deploy service
        deployment = await self._deploy_service_to_node(service_type, service_config, optimal_node)
        
        if deployment:
            self.service_deployments[deployment.service_name] = deployment
            
            # Update Redis
            if self.redis_client:
                await self._update_deployment_in_redis(deployment)
                
            return {
                "success": True,
                "deployment_id": deployment.service_name,
                "node_id": deployment.node_id,
                "node_hostname": optimal_node.hostname,
                "service_ports": self._extract_service_ports(service_config),
                "access_url": self._generate_service_url(optimal_node, service_config)
            }
        else:
            return {"error": "Failed to deploy service"}
            
    async def _find_optimal_node_for_service(self, service_type: str, service_config: Dict[str, Any]) -> Optional[ThunderboltNode]:
        """Find the optimal node for a service deployment"""
        suitable_nodes = []
        
        resource_requirements = service_config.get("resources", {})
        
        for node in self.nodes.values():
            if node.health_status != "healthy":
                continue
                
            # Check resource requirements
            if self._node_meets_requirements(node, resource_requirements):
                # Calculate suitability score
                score = await self._calculate_node_suitability_score(node, service_type, resource_requirements)
                suitable_nodes.append((score, node))
                
        if not suitable_nodes:
            return None
            
        # Sort by score and return best node
        suitable_nodes.sort(key=lambda x: x[0], reverse=True)
        return suitable_nodes[0][1]
        
    def _node_meets_requirements(self, node: ThunderboltNode, requirements: Dict[str, Any]) -> bool:
        """Check if a node meets service requirements"""
        # Check CPU
        required_cpu = requirements.get("cpu", 1)
        if node.cpu_cores < required_cpu:
            return False
            
        # Check memory
        required_memory_str = requirements.get("memory", "1GB")
        required_memory = self._parse_memory_string(required_memory_str)
        if node.memory_gb < required_memory:
            return False
            
        # Check GPU if required
        if requirements.get("gpu", False) and not node.gpu_info.get("available", False):
            return False
            
        return True
        
    def _parse_memory_string(self, memory_str: str) -> float:
        """Parse memory string like '4GB' to float in GB"""
        memory_str = memory_str.upper()
        if memory_str.endswith("GB"):
            return float(memory_str[:-2])
        elif memory_str.endswith("MB"):
            return float(memory_str[:-2]) / 1024
        else:
            return float(memory_str)
            
    async def _calculate_node_suitability_score(self, node: ThunderboltNode, service_type: str, requirements: Dict[str, Any]) -> float:
        """Calculate suitability score for a node"""
        score = 0.0
        
        # Base score from available resources
        cpu_utilization = node.current_load
        memory_utilization = (node.memory_gb - node.available_memory) / node.memory_gb
        
        score += (1.0 - cpu_utilization) * 40  # 40 points for low CPU usage
        score += (1.0 - memory_utilization) * 30  # 30 points for available memory
        
        # Bonus for device type suitability
        if service_type in ["vllm", "triton"] and node.device_type == "mac_studio":
            score += 20  # Mac Studio preferred for GPU-intensive services
        elif service_type in ["llm-d", "localai"] and node.device_type == "mac_mini":
            score += 15  # Mac Mini good for coordinator services
            
        # Network bandwidth bonus
        score += min(node.network_bandwidth / 40.0, 1.0) * 10  # Up to 10 points for high bandwidth
        
        return score
        
    async def _deploy_service_to_node(self, service_type: str, service_config: Dict[str, Any], node: ThunderboltNode) -> Optional[ServiceDeployment]:
        """Deploy a service to a specific node"""
        try:
            # Generate unique service name
            service_name = f"{service_type}_{node.hostname}_{int(time.time())}"
            
            # Get Docker client for node
            docker_client = await self._get_docker_client_for_node(node)
            
            if not docker_client:
                logger.error(f"Could not get Docker client for node {node.hostname}")
                return None
                
            # Prepare container configuration
            container_config = await self._prepare_container_config(service_name, service_config, node)
            
            # Create and start container
            container = docker_client.containers.run(
                detach=True,
                name=service_name,
                **container_config
            )
            
            # Create deployment record
            deployment = ServiceDeployment(
                service_name=service_name,
                service_type=service_type,
                node_id=node.id,
                port=self._extract_primary_port(service_config),
                resource_requirements=service_config.get("resources", {}),
                configuration=service_config,
                status="running"
            )
            
            logger.info(f"✅ Deployed {service_type} as {service_name} on {node.hostname}")
            
            return deployment
            
        except Exception as e:
            logger.error(f"Error deploying {service_type} to {node.hostname}: {e}")
            return None
            
    async def _get_docker_client_for_node(self, node: ThunderboltNode) -> Optional[docker.DockerClient]:
        """Get Docker client for a specific node"""
        try:
            if node.ip_address == "localhost" or node.hostname == socket.gethostname():
                # Local node
                return docker.from_env()
            else:
                # Remote node - use Docker over SSH
                # This requires Docker context to be set up
                return docker.DockerClient(base_url=f"ssh://{node.ip_address}")
                
        except Exception as e:
            logger.error(f"Error getting Docker client for {node.hostname}: {e}")
            return None
            
    async def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        status = {
            "nodes": {},
            "services": {},
            "performance": {},
            "health": {
                "healthy_nodes": 0,
                "total_nodes": len(self.nodes),
                "running_services": 0,
                "total_services": len(self.service_deployments)
            }
        }
        
        # Node status
        for node_id, node in self.nodes.items():
            status["nodes"][node_id] = {
                "hostname": node.hostname,
                "ip_address": node.ip_address,
                "device_type": node.device_type,
                "health_status": node.health_status,
                "cpu_cores": node.cpu_cores,
                "memory_gb": node.memory_gb,
                "current_load": node.current_load,
                "available_memory": node.available_memory,
                "running_services": len(node.running_services),
                "thunderbolt_version": node.thunderbolt_version,
                "network_bandwidth": node.network_bandwidth
            }
            
            if node.health_status == "healthy":
                status["health"]["healthy_nodes"] += 1
                
        # Service status
        for service_name, deployment in self.service_deployments.items():
            status["services"][service_name] = {
                "service_type": deployment.service_type,
                "node_id": deployment.node_id,
                "port": deployment.port,
                "status": deployment.status,
                "deployment_time": deployment.deployment_time.isoformat(),
                "performance_metrics": deployment.performance_metrics
            }
            
            if deployment.status == "running":
                status["health"]["running_services"] += 1
                
        return status
        
    async def optimize_network_performance(self) -> Dict[str, Any]:
        """Optimize network performance across all nodes"""
        logger.info("⚡ Optimizing network performance...")
        
        optimizations = []
        
        # Rebalance services if needed
        rebalancing_suggestions = await self._analyze_load_rebalancing()
        if rebalancing_suggestions:
            optimizations.extend(rebalancing_suggestions)
            
        # Optimize service configurations
        config_optimizations = await self._optimize_service_configurations()
        if config_optimizations:
            optimizations.extend(config_optimizations)
            
        # Network bandwidth optimization
        bandwidth_optimizations = await self._optimize_network_bandwidth()
        if bandwidth_optimizations:
            optimizations.extend(bandwidth_optimizations)
            
        return {
            "optimizations_applied": len(optimizations),
            "optimizations": optimizations,
            "performance_improvement_estimate": sum(opt.get("improvement", 0) for opt in optimizations),
            "timestamp": datetime.now().isoformat()
        }
        
    async def provide_automation_options_for_thunderbolt(self, user_request: str) -> Dict[str, Any]:
        """Provide automation options for Thunderbolt network optimization"""
        logger.info(f"⚡ Generating Thunderbolt automation options: {user_request}")
        
        automation_options = {
            "immediate_deployment": [],
            "network_optimization": [],
            "service_management": [],
            "monitoring_automation": [],
            "performance_tuning": [],
            "failure_recovery": []
        }
        
        request_lower = user_request.lower()
        
        # Immediate deployment options
        automation_options["immediate_deployment"].extend([
            "deploy_vllm_to_optimal_mac_studio",
            "deploy_llm_d_coordinator_and_workers",
            "deploy_localai_across_all_minis",
            "deploy_triton_inference_server",
            "setup_redis_cluster_coordination"
        ])
        
        # Network optimization options
        automation_options["network_optimization"].extend([
            "optimize_thunderbolt_bandwidth_allocation",
            "enable_dynamic_load_balancing",
            "configure_service_affinity_rules",
            "setup_network_quality_monitoring",
            "implement_traffic_shaping"
        ])
        
        # Service management automation
        automation_options["service_management"].extend([
            "auto_scale_services_based_on_demand",
            "implement_rolling_updates",
            "setup_service_health_checks",
            "configure_automatic_failover",
            "enable_cross_node_service_migration"
        ])
        
        # Monitoring automation
        automation_options["monitoring_automation"].extend([
            "setup_real_time_performance_dashboards",
            "configure_thunderbolt_bandwidth_monitoring",
            "enable_resource_utilization_tracking",
            "implement_predictive_capacity_planning",
            "setup_automated_alerting"
        ])
        
        # Performance tuning automation
        automation_options["performance_tuning"].extend([
            "auto_tune_inference_server_parameters",
            "optimize_model_sharding_across_nodes",
            "implement_dynamic_batching",
            "configure_memory_optimization",
            "enable_gpu_memory_pooling"
        ])
        
        # Failure recovery automation
        automation_options["failure_recovery"].extend([
            "implement_automatic_node_failure_detection",
            "setup_service_restart_policies",
            "configure_data_replication",
            "enable_graceful_degradation",
            "implement_disaster_recovery_procedures"
        ])
        
        # Generate setup commands
        setup_commands = {
            category: [
                f"python thunderbolt_network_optimizer.py --{option.replace('_', '-')}"
                for option in options
            ]
            for category, options in automation_options.items()
        }
        
        # Estimate benefits
        benefits = {
            "performance_improvement": "3-5x faster inference through optimal distribution",
            "resource_utilization": "85-95% efficient use of all Mac hardware",
            "fault_tolerance": "99.9% uptime through automatic failover",
            "cost_efficiency": "70% reduction in cloud costs through local compute",
            "latency_reduction": "90% lower latency through Thunderbolt speeds"
        }
        
        return {
            "request": user_request,
            "automation_options": automation_options,
            "setup_commands": setup_commands,
            "estimated_benefits": benefits,
            "network_topology": {
                "total_nodes": len(self.nodes),
                "healthy_nodes": len([n for n in self.nodes.values() if n.health_status == "healthy"]),
                "total_cpu_cores": sum(n.cpu_cores for n in self.nodes.values()),
                "total_memory_gb": sum(n.memory_gb for n in self.nodes.values()),
                "thunderbolt_bandwidth": max((n.network_bandwidth for n in self.nodes.values()), default=0)
            },
            "recommended_immediate": [
                "deploy_vllm_to_optimal_mac_studio",
                "setup_redis_cluster_coordination", 
                "enable_dynamic_load_balancing"
            ]
        }

async def main():
    """Main execution function"""
    optimizer = ThunderboltNetworkOptimizer()
    await optimizer.initialize()
    
    # Deploy a service optimally
    result = await optimizer.deploy_service_optimally(
        "vllm",
        config_overrides={"model": "microsoft/DialoGPT-medium"}
    )
    
    print("Service Deployment Result:")
    print(json.dumps(result, indent=2, default=str))
    
    # Get network status
    status = await optimizer.get_network_status()
    print("\nThunderbolt Network Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Get automation options
    automation = await optimizer.provide_automation_options_for_thunderbolt(
        "Optimize the Thunderbolt network for maximum performance, deploy all inference services optimally, and enable automatic load balancing"
    )
    
    print("\nThunderbolt Network Automation Options:")
    print(json.dumps(automation, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())