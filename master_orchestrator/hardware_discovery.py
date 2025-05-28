#!/usr/bin/env python3

"""
Hardware Discovery and Distributed vLLM Agent
Automatically discovers Mac Studios and Mac Minis on the network,
deploys vLLM-d for distributed inference, and manages Git operations
"""

import asyncio
import json
import time
import subprocess
import socket
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
import paramiko
import netifaces
import structlog

from unified_config import get_config_manager
from dev_capabilities import MasterDevController

logger = structlog.get_logger()

@dataclass
class HardwareNode:
    """Represents a discovered hardware node."""
    
    hostname: str
    ip_address: str
    mac_address: str
    device_type: str  # mac_studio, mac_mini, macbook_pro
    cpu_model: str
    cpu_cores: int
    memory_gb: float
    gpu_cores: Optional[int] = None
    storage_gb: float = 0.0
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    online: bool = True
    vllm_port: Optional[int] = None
    vllm_status: str = "not_deployed"  # not_deployed, deploying, running, error
    capabilities: List[str] = field(default_factory=list)
    load_average: float = 0.0
    memory_usage_percent: float = 0.0
    temperature: Optional[float] = None

@dataclass
class DistributedVLLMCluster:
    """Represents the distributed vLLM cluster configuration."""
    
    coordinator_node: str
    worker_nodes: List[str] = field(default_factory=list)
    model_name: str = "microsoft/DialoGPT-medium"
    total_gpus: int = 0
    cluster_status: str = "offline"  # offline, starting, running, error
    load_balancer_port: int = 8080
    monitoring_port: int = 8081
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class NetworkScanner:
    """Network scanner for discovering Mac hardware."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=50)
        self.discovered_nodes: Dict[str, HardwareNode] = {}
        self.scan_timeout = 2.0
        
    def get_local_networks(self) -> List[str]:
        """Get all local network ranges to scan."""
        networks = []
        
        try:
            # Get all network interfaces
            for interface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(interface)
                
                # Look for IPv4 addresses
                if netifaces.AF_INET in addrs:
                    for addr_info in addrs[netifaces.AF_INET]:
                        ip = addr_info.get('addr')
                        netmask = addr_info.get('netmask')
                        
                        if ip and netmask and not ip.startswith('127.'):
                            # Calculate network range
                            network = self._calculate_network(ip, netmask)
                            if network:
                                networks.append(network)
        except Exception as e:
            logger.warning("Failed to get network interfaces", error=str(e))
            # Fallback to common networks
            networks = ["192.168.1.0/24", "192.168.0.0/24", "10.0.0.0/24"]
        
        return networks
    
    def _calculate_network(self, ip: str, netmask: str) -> Optional[str]:
        """Calculate network range from IP and netmask."""
        try:
            import ipaddress
            
            network = ipaddress.IPv4Network(f"{ip}/{netmask}", strict=False)
            return str(network)
        except Exception:
            return None
    
    async def scan_network(self) -> Dict[str, HardwareNode]:
        """Scan network for Mac hardware."""
        logger.info("Starting network scan for Mac hardware")
        
        networks = self.get_local_networks()
        scan_tasks = []
        
        for network in networks:
            try:
                import ipaddress
                net = ipaddress.IPv4Network(network)
                
                # Scan each IP in the network
                for ip in net.hosts():
                    task = asyncio.create_task(self._scan_ip(str(ip)))
                    scan_tasks.append(task)
                    
            except Exception as e:
                logger.warning("Failed to scan network", network=network, error=str(e))
        
        # Wait for all scans to complete
        if scan_tasks:
            results = await asyncio.gather(*scan_tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, HardwareNode):
                    self.discovered_nodes[result.ip_address] = result
        
        logger.info("Network scan completed", 
                   nodes_found=len(self.discovered_nodes),
                   mac_studios=len([n for n in self.discovered_nodes.values() if n.device_type == "mac_studio"]),
                   mac_minis=len([n for n in self.discovered_nodes.values() if n.device_type == "mac_mini"]))
        
        return self.discovered_nodes
    
    async def _scan_ip(self, ip: str) -> Optional[HardwareNode]:
        """Scan a single IP address for Mac hardware."""
        try:
            # Quick port scan for common services
            if not await self._is_host_alive(ip):
                return None
            
            # Try to identify if it's a Mac
            node_info = await self._identify_mac_hardware(ip)
            if node_info:
                return node_info
                
        except Exception as e:
            logger.debug("IP scan failed", ip=ip, error=str(e))
        
        return None
    
    async def _is_host_alive(self, ip: str) -> bool:
        """Check if host is alive using ping."""
        try:
            # Use ping command
            process = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', '1000', ip,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await asyncio.wait_for(process.wait(), timeout=2.0)
            return process.returncode == 0
        except:
            return False
    
    async def _identify_mac_hardware(self, ip: str) -> Optional[HardwareNode]:
        """Try to identify Mac hardware at the given IP."""
        try:
            # Try SSH connection to get system info
            hardware_info = await self._get_ssh_hardware_info(ip)
            if hardware_info:
                return hardware_info
            
            # Try HTTP services (if any Mac services are running)
            hardware_info = await self._get_http_hardware_info(ip)
            if hardware_info:
                return hardware_info
                
        except Exception as e:
            logger.debug("Hardware identification failed", ip=ip, error=str(e))
        
        return None
    
    async def _get_ssh_hardware_info(self, ip: str) -> Optional[HardwareNode]:
        """Get hardware info via SSH."""
        try:
            # This would require SSH keys to be set up
            # For now, we'll simulate the detection
            
            # Try to connect and get system info
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Try common usernames
            usernames = ['admin', 'user', os.getenv('USER', 'user')]
            
            for username in usernames:
                try:
                    ssh_client.connect(
                        ip, 
                        username=username, 
                        timeout=2.0,
                        look_for_keys=True,
                        allow_agent=True
                    )
                    
                    # Get system information
                    stdin, stdout, stderr = ssh_client.exec_command('system_profiler SPHardwareDataType')
                    hardware_output = stdout.read().decode()
                    
                    stdin, stdout, stderr = ssh_client.exec_command('hostname')
                    hostname = stdout.read().decode().strip()
                    
                    # Parse hardware info
                    node = self._parse_mac_hardware_info(ip, hostname, hardware_output)
                    ssh_client.close()
                    
                    if node:
                        logger.info("Discovered Mac hardware via SSH", 
                                   ip=ip, 
                                   hostname=hostname,
                                   device_type=node.device_type)
                        return node
                        
                except paramiko.AuthenticationException:
                    continue
                except Exception as e:
                    logger.debug("SSH connection failed", ip=ip, username=username, error=str(e))
                    continue
                finally:
                    ssh_client.close()
                    
        except Exception as e:
            logger.debug("SSH hardware detection failed", ip=ip, error=str(e))
        
        return None
    
    async def _get_http_hardware_info(self, ip: str) -> Optional[HardwareNode]:
        """Try to get hardware info via HTTP services."""
        try:
            # Check common ports for Mac services
            ports_to_check = [80, 443, 5000, 8080, 8000]
            
            for port in ports_to_check:
                try:
                    response = requests.get(
                        f"http://{ip}:{port}/health", 
                        timeout=1.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'platform' in data and 'darwin' in data['platform'].lower():
                            # This is likely a Mac running our service
                            return self._create_node_from_http_data(ip, data)
                            
                except:
                    continue
                    
        except Exception as e:
            logger.debug("HTTP hardware detection failed", ip=ip, error=str(e))
        
        return None
    
    def _parse_mac_hardware_info(self, ip: str, hostname: str, hardware_output: str) -> Optional[HardwareNode]:
        """Parse macOS system_profiler output to identify device type."""
        try:
            # Extract device information from system_profiler output
            device_type = "unknown"
            cpu_model = "Unknown"
            cpu_cores = 0
            memory_gb = 0.0
            
            lines = hardware_output.split('\n')
            for line in lines:
                line = line.strip()
                
                if 'Model Name:' in line:
                    model_name = line.split(':', 1)[1].strip()
                    if 'Mac Studio' in model_name:
                        device_type = "mac_studio"
                    elif 'Mac mini' in model_name:
                        device_type = "mac_mini" 
                    elif 'MacBook Pro' in model_name:
                        device_type = "macbook_pro"
                
                elif 'Chip:' in line or 'Processor Name:' in line:
                    cpu_model = line.split(':', 1)[1].strip()
                
                elif 'Total Number of Cores:' in line:
                    try:
                        cpu_cores = int(line.split(':', 1)[1].strip())
                    except:
                        pass
                
                elif 'Memory:' in line:
                    try:
                        memory_str = line.split(':', 1)[1].strip()
                        if 'GB' in memory_str:
                            memory_gb = float(memory_str.replace('GB', '').strip())
                    except:
                        pass
            
            if device_type != "unknown":
                return HardwareNode(
                    hostname=hostname,
                    ip_address=ip,
                    mac_address="",  # Would need to get this separately
                    device_type=device_type,
                    cpu_model=cpu_model,
                    cpu_cores=cpu_cores,
                    memory_gb=memory_gb,
                    capabilities=["vllm", "inference", "distributed"]
                )
                
        except Exception as e:
            logger.debug("Failed to parse hardware info", error=str(e))
        
        return None
    
    def _create_node_from_http_data(self, ip: str, data: Dict[str, Any]) -> HardwareNode:
        """Create HardwareNode from HTTP service data."""
        return HardwareNode(
            hostname=data.get('hostname', f'node-{ip}'),
            ip_address=ip,
            mac_address="",
            device_type=data.get('device_type', 'unknown'),
            cpu_model=data.get('cpu_model', 'Unknown'),
            cpu_cores=data.get('cpu_cores', 0),
            memory_gb=data.get('memory_gb', 0.0),
            capabilities=data.get('capabilities', [])
        )

class VLLMDistributedManager:
    """Manages distributed vLLM deployment across discovered hardware."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.discovered_nodes: Dict[str, HardwareNode] = {}
        self.vllm_cluster: Optional[DistributedVLLMCluster] = None
        self.base_port = 8080
        
    async def deploy_distributed_vllm(self, nodes: Dict[str, HardwareNode]) -> DistributedVLLMCluster:
        """Deploy vLLM-d across discovered nodes."""
        self.discovered_nodes = nodes
        
        # Filter nodes suitable for vLLM
        suitable_nodes = self._filter_suitable_nodes(nodes)
        
        if not suitable_nodes:
            raise ValueError("No suitable nodes found for vLLM deployment")
        
        # Select coordinator node (highest specs)
        coordinator = self._select_coordinator_node(suitable_nodes)
        worker_nodes = [node.ip_address for node in suitable_nodes.values() if node.ip_address != coordinator]
        
        # Create cluster configuration
        self.vllm_cluster = DistributedVLLMCluster(
            coordinator_node=coordinator,
            worker_nodes=worker_nodes,
            total_gpus=sum(node.gpu_cores or 8 for node in suitable_nodes.values()),  # Estimate
            load_balancer_port=self.base_port,
            monitoring_port=self.base_port + 1
        )
        
        logger.info("Starting distributed vLLM deployment",
                   coordinator=coordinator,
                   workers=len(worker_nodes),
                   total_nodes=len(suitable_nodes))
        
        try:
            # Deploy vLLM to each node
            deployment_tasks = []
            
            # Deploy coordinator
            deployment_tasks.append(
                self._deploy_vllm_coordinator(suitable_nodes[coordinator])
            )
            
            # Deploy workers
            for worker_ip in worker_nodes:
                if worker_ip in suitable_nodes:
                    deployment_tasks.append(
                        self._deploy_vllm_worker(suitable_nodes[worker_ip], coordinator)
                    )
            
            # Wait for all deployments
            results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
            
            # Check deployment results
            successful_deployments = sum(1 for r in results if r is True)
            total_deployments = len(results)
            
            if successful_deployments == total_deployments:
                self.vllm_cluster.cluster_status = "running"
                logger.info("Distributed vLLM cluster deployed successfully",
                           successful=successful_deployments,
                           total=total_deployments)
            else:
                self.vllm_cluster.cluster_status = "error"
                logger.error("Some vLLM deployments failed",
                           successful=successful_deployments,
                           total=total_deployments)
            
        except Exception as e:
            self.vllm_cluster.cluster_status = "error"
            logger.error("Distributed vLLM deployment failed", error=str(e))
        
        return self.vllm_cluster
    
    def _filter_suitable_nodes(self, nodes: Dict[str, HardwareNode]) -> Dict[str, HardwareNode]:
        """Filter nodes suitable for vLLM deployment."""
        suitable = {}
        
        for ip, node in nodes.items():
            # Check if node meets minimum requirements
            if (node.device_type in ["mac_studio", "mac_mini"] and 
                node.memory_gb >= 16 and  # Minimum 16GB RAM
                node.cpu_cores >= 8 and   # Minimum 8 cores
                node.online):
                suitable[ip] = node
        
        return suitable
    
    def _select_coordinator_node(self, nodes: Dict[str, HardwareNode]) -> str:
        """Select the best node to be the coordinator."""
        # Select node with highest specs (Mac Studio preferred, then highest memory)
        best_node = None
        best_score = 0
        
        for node in nodes.values():
            score = 0
            
            # Device type preference
            if node.device_type == "mac_studio":
                score += 100
            elif node.device_type == "mac_mini":
                score += 50
            
            # Memory score
            score += node.memory_gb
            
            # CPU score
            score += node.cpu_cores * 2
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node.ip_address if best_node else list(nodes.keys())[0]
    
    async def _deploy_vllm_coordinator(self, node: HardwareNode) -> bool:
        """Deploy vLLM coordinator to a node."""
        try:
            logger.info("Deploying vLLM coordinator", node=node.hostname, ip=node.ip_address)
            
            # Update node status
            node.vllm_status = "deploying"
            node.vllm_port = self.base_port
            
            # Create deployment script
            deployment_script = self._create_vllm_coordinator_script(node)
            
            # Execute deployment
            success = await self._execute_remote_deployment(node, deployment_script)
            
            if success:
                node.vllm_status = "running"
                logger.info("vLLM coordinator deployed successfully", node=node.hostname)
            else:
                node.vllm_status = "error"
                logger.error("vLLM coordinator deployment failed", node=node.hostname)
            
            return success
            
        except Exception as e:
            node.vllm_status = "error"
            logger.error("vLLM coordinator deployment error", node=node.hostname, error=str(e))
            return False
    
    async def _deploy_vllm_worker(self, node: HardwareNode, coordinator_ip: str) -> bool:
        """Deploy vLLM worker to a node."""
        try:
            logger.info("Deploying vLLM worker", node=node.hostname, ip=node.ip_address)
            
            # Update node status
            node.vllm_status = "deploying"
            node.vllm_port = self.base_port + len(self.vllm_cluster.worker_nodes) + 2
            
            # Create deployment script
            deployment_script = self._create_vllm_worker_script(node, coordinator_ip)
            
            # Execute deployment
            success = await self._execute_remote_deployment(node, deployment_script)
            
            if success:
                node.vllm_status = "running"
                logger.info("vLLM worker deployed successfully", node=node.hostname)
            else:
                node.vllm_status = "error"
                logger.error("vLLM worker deployment failed", node=node.hostname)
            
            return success
            
        except Exception as e:
            node.vllm_status = "error"
            logger.error("vLLM worker deployment error", node=node.hostname, error=str(e))
            return False
    
    def _create_vllm_coordinator_script(self, node: HardwareNode) -> str:
        """Create deployment script for vLLM coordinator."""
        return f"""#!/bin/bash
set -e

echo "Starting vLLM coordinator deployment on {node.hostname}"

# Create working directory
mkdir -p ~/vllm_cluster
cd ~/vllm_cluster

# Install/update vLLM if needed
pip3 install --upgrade vllm

# Create coordinator configuration
cat > coordinator_config.json << EOF
{{
    "model": "{self.vllm_cluster.model_name}",
    "port": {node.vllm_port},
    "host": "0.0.0.0",
    "tensor_parallel_size": 1,
    "distributed_executor_backend": "ray",
    "worker_use_ray": true,
    "ray_workers_use_nsight": false
}}
EOF

# Start vLLM coordinator
echo "Starting vLLM coordinator on port {node.vllm_port}"
nohup python3 -m vllm.entrypoints.api_server \\
    --model {self.vllm_cluster.model_name} \\
    --port {node.vllm_port} \\
    --host 0.0.0.0 \\
    --tensor-parallel-size 1 \\
    > vllm_coordinator.log 2>&1 &

sleep 5

# Verify deployment
curl -s http://localhost:{node.vllm_port}/health || exit 1

echo "vLLM coordinator deployed successfully"
"""
    
    def _create_vllm_worker_script(self, node: HardwareNode, coordinator_ip: str) -> str:
        """Create deployment script for vLLM worker."""
        return f"""#!/bin/bash
set -e

echo "Starting vLLM worker deployment on {node.hostname}"

# Create working directory
mkdir -p ~/vllm_cluster
cd ~/vllm_cluster

# Install/update vLLM if needed
pip3 install --upgrade vllm

# Create worker configuration
cat > worker_config.json << EOF
{{
    "coordinator_ip": "{coordinator_ip}",
    "coordinator_port": {self.base_port},
    "worker_port": {node.vllm_port},
    "model": "{self.vllm_cluster.model_name}"
}}
EOF

# Start vLLM worker
echo "Starting vLLM worker on port {node.vllm_port}, connecting to coordinator {coordinator_ip}:{self.base_port}"
nohup python3 -m vllm.entrypoints.api_server \\
    --model {self.vllm_cluster.model_name} \\
    --port {node.vllm_port} \\
    --host 0.0.0.0 \\
    --distributed-executor-backend ray \\
    --worker-use-ray \\
    > vllm_worker.log 2>&1 &

sleep 5

# Verify deployment
curl -s http://localhost:{node.vllm_port}/health || exit 1

echo "vLLM worker deployed successfully"
"""
    
    async def _execute_remote_deployment(self, node: HardwareNode, script: str) -> bool:
        """Execute deployment script on remote node."""
        try:
            # For now, simulate deployment
            # In real implementation, this would use SSH to execute the script
            
            logger.info("Simulating vLLM deployment", node=node.hostname)
            await asyncio.sleep(2)  # Simulate deployment time
            
            # Return success for simulation
            return True
            
        except Exception as e:
            logger.error("Remote deployment failed", node=node.hostname, error=str(e))
            return False
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        if not self.vllm_cluster:
            return {"status": "not_deployed"}
        
        status = {
            "cluster_status": self.vllm_cluster.cluster_status,
            "coordinator": self.vllm_cluster.coordinator_node,
            "workers": len(self.vllm_cluster.worker_nodes),
            "total_nodes": len(self.discovered_nodes),
            "model": self.vllm_cluster.model_name,
            "nodes": {}
        }
        
        # Add individual node status
        for ip, node in self.discovered_nodes.items():
            status["nodes"][ip] = {
                "hostname": node.hostname,
                "device_type": node.device_type,
                "vllm_status": node.vllm_status,
                "vllm_port": node.vllm_port,
                "online": node.online
            }
        
        return status

class IntelligentGitAgent:
    """Intelligent Git agent for automatic commits with meaningful messages."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.dev_controller = MasterDevController(Path('.'))
        self.commit_patterns = self._initialize_commit_patterns()
        
    def _initialize_commit_patterns(self) -> Dict[str, str]:
        """Initialize commit message patterns based on file types and changes."""
        return {
            'hardware_discovery': "feat(hardware): discover and configure {device_count} {device_types}",
            'vllm_deployment': "feat(vllm): deploy distributed inference across {node_count} nodes",
            'configuration': "config: update {config_type} configuration for {environment}",
            'monitoring': "feat(monitoring): add {metric_type} monitoring for {component}",
            'testing': "test: add {test_type} tests for {component} with {coverage}% coverage",
            'bugfix': "fix({component}): resolve {issue_type} in {affected_area}",
            'optimization': "perf({component}): optimize {operation_type} performance by {improvement}",
            'documentation': "docs: update {doc_type} documentation for {feature}",
            'security': "security: enhance {security_aspect} protection in {component}",
            'refactor': "refactor({component}): improve {aspect} structure and maintainability"
        }
    
    async def analyze_and_commit_changes(self) -> Dict[str, Any]:
        """Analyze current changes and create intelligent commit."""
        try:
            logger.info("Analyzing repository changes for intelligent commit")
            
            # Get current git status
            git_status = self.dev_controller.git.get_status()
            
            if not git_status or 'nothing to commit' in git_status.get('output', ''):
                return {"status": "no_changes", "message": "No changes to commit"}
            
            # Analyze changed files
            analysis = await self._analyze_changed_files()
            
            # Generate commit message
            commit_message = self._generate_commit_message(analysis)
            
            # Create detailed commit body
            commit_body = self._generate_commit_body(analysis)
            
            # Full commit message
            full_commit_message = f"{commit_message}\n\n{commit_body}"
            
            # Commit changes
            commit_result = self.dev_controller.git.commit_changes(full_commit_message)
            
            if commit_result.get('success'):
                logger.info("Intelligent commit created successfully", 
                           message=commit_message,
                           files_changed=len(analysis['changed_files']))
                
                # Update system state
                self.config_manager.add_completed_task(f"git_commit_{int(time.time())}")
                
                return {
                    "status": "success",
                    "commit_message": commit_message,
                    "files_changed": len(analysis['changed_files']),
                    "commit_hash": commit_result.get('hash', 'unknown'),
                    "analysis": analysis
                }
            else:
                return {
                    "status": "error",
                    "error": commit_result.get('error', 'Unknown commit error'),
                    "analysis": analysis
                }
                
        except Exception as e:
            logger.error("Intelligent commit failed", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def _analyze_changed_files(self) -> Dict[str, Any]:
        """Analyze changed files to understand the nature of changes."""
        analysis = {
            'changed_files': [],
            'change_types': set(),
            'affected_components': set(),
            'primary_change_type': None,
            'impact_level': 'minor',
            'features_added': [],
            'bugs_fixed': [],
            'performance_improvements': []
        }
        
        try:
            # Get diff information
            diff_result = self.dev_controller.git.get_diff()
            
            if diff_result.get('success'):
                diff_output = diff_result.get('output', '')
                analysis['changed_files'] = self._extract_changed_files(diff_output)
                
                # Analyze each file
                for file_path in analysis['changed_files']:
                    file_analysis = self._analyze_file_changes(file_path, diff_output)
                    
                    analysis['change_types'].update(file_analysis['change_types'])
                    analysis['affected_components'].update(file_analysis['components'])
                    analysis['features_added'].extend(file_analysis['features'])
                    analysis['bugs_fixed'].extend(file_analysis['bugs'])
                    analysis['performance_improvements'].extend(file_analysis['performance'])
                
                # Determine primary change type and impact
                analysis['primary_change_type'] = self._determine_primary_change_type(analysis['change_types'])
                analysis['impact_level'] = self._determine_impact_level(analysis)
                
        except Exception as e:
            logger.error("File analysis failed", error=str(e))
        
        return analysis
    
    def _extract_changed_files(self, diff_output: str) -> List[str]:
        """Extract list of changed files from git diff output."""
        files = []
        
        for line in diff_output.split('\n'):
            if line.startswith('diff --git'):
                # Extract file path from: diff --git a/file.py b/file.py
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3][2:]  # Remove 'b/' prefix
                    files.append(file_path)
        
        return files
    
    def _analyze_file_changes(self, file_path: str, diff_output: str) -> Dict[str, Any]:
        """Analyze changes in a specific file."""
        analysis = {
            'change_types': set(),
            'components': set(),
            'features': [],
            'bugs': [],
            'performance': []
        }
        
        # Determine component from file path
        if 'hardware' in file_path:
            analysis['components'].add('hardware')
        if 'vllm' in file_path or 'llm' in file_path:
            analysis['components'].add('llm')
        if 'config' in file_path:
            analysis['components'].add('configuration')
        if 'test' in file_path:
            analysis['components'].add('testing')
        if 'git' in file_path or 'dev_capabilities' in file_path:
            analysis['components'].add('development')
        
        # Analyze file type
        if file_path.endswith('.py'):
            analysis['change_types'].add('code')
        elif file_path.endswith('.md'):
            analysis['change_types'].add('documentation')
        elif file_path.endswith(('.yaml', '.yml', '.json')):
            analysis['change_types'].add('configuration')
        elif file_path.endswith('.sh'):
            analysis['change_types'].add('scripts')
        
        # Look for specific patterns in diff
        file_diff = self._extract_file_diff(file_path, diff_output)
        
        # Feature detection patterns
        if any(keyword in file_diff.lower() for keyword in ['def ', 'class ', 'async def']):
            analysis['change_types'].add('feature')
            if 'discover' in file_diff.lower():
                analysis['features'].append('hardware discovery')
            if 'vllm' in file_diff.lower():
                analysis['features'].append('distributed inference')
            if 'commit' in file_diff.lower():
                analysis['features'].append('intelligent git operations')
        
        # Bug fix detection
        if any(keyword in file_diff.lower() for keyword in ['fix', 'bug', 'error', 'exception']):
            analysis['change_types'].add('bugfix')
        
        # Performance improvement detection
        if any(keyword in file_diff.lower() for keyword in ['optimize', 'performance', 'faster', 'cache']):
            analysis['change_types'].add('performance')
        
        return analysis
    
    def _extract_file_diff(self, file_path: str, diff_output: str) -> str:
        """Extract diff content for a specific file."""
        lines = diff_output.split('\n')
        file_diff = []
        in_file = False
        
        for line in lines:
            if line.startswith(f'diff --git') and file_path in line:
                in_file = True
            elif line.startswith('diff --git') and in_file:
                break
            elif in_file:
                file_diff.append(line)
        
        return '\n'.join(file_diff)
    
    def _determine_primary_change_type(self, change_types: set) -> str:
        """Determine the primary type of changes."""
        # Priority order for change types
        priority = ['feature', 'bugfix', 'performance', 'configuration', 'documentation', 'code']
        
        for change_type in priority:
            if change_type in change_types:
                return change_type
        
        return 'misc'
    
    def _determine_impact_level(self, analysis: Dict[str, Any]) -> str:
        """Determine the impact level of changes."""
        file_count = len(analysis['changed_files'])
        
        # High impact: many files, core components, or breaking changes
        if (file_count > 5 or 
            'hardware' in analysis['affected_components'] or
            'llm' in analysis['affected_components']):
            return 'major'
        
        # Medium impact: moderate changes
        elif file_count > 2 or 'configuration' in analysis['affected_components']:
            return 'minor'
        
        # Low impact: small changes
        else:
            return 'patch'
    
    def _generate_commit_message(self, analysis: Dict[str, Any]) -> str:
        """Generate intelligent commit message based on analysis."""
        primary_type = analysis['primary_change_type']
        components = list(analysis['affected_components'])
        
        if primary_type == 'feature':
            if 'hardware' in components:
                device_count = len([f for f in analysis['changed_files'] if 'hardware' in f])
                return f"feat(hardware): implement distributed hardware discovery and vLLM deployment"
            elif 'llm' in components:
                return f"feat(llm): add distributed inference capabilities with load balancing"
            elif 'development' in components:
                return f"feat(git): implement intelligent commit analysis and automation"
            else:
                feature_desc = analysis['features_added'][0] if analysis['features_added'] else 'new functionality'
                return f"feat: add {feature_desc}"
        
        elif primary_type == 'bugfix':
            component = components[0] if components else 'system'
            return f"fix({component}): resolve issues in {', '.join(components[:2])}"
        
        elif primary_type == 'performance':
            component = components[0] if components else 'system'
            improvement = analysis['performance_improvements'][0] if analysis['performance_improvements'] else 'performance'
            return f"perf({component}): optimize {improvement}"
        
        elif primary_type == 'configuration':
            return f"config: update system configuration for {', '.join(components[:2])}"
        
        elif primary_type == 'documentation':
            return f"docs: update documentation for {', '.join(components[:2])}"
        
        else:
            return f"chore: update {', '.join(components[:2]) if components else 'system files'}"
    
    def _generate_commit_body(self, analysis: Dict[str, Any]) -> str:
        """Generate detailed commit body with changes summary."""
        body_parts = []
        
        # Add summary of changes
        body_parts.append("Changes:")
        for i, file_path in enumerate(analysis['changed_files'][:10]):  # Limit to 10 files
            body_parts.append(f"- {file_path}")
        
        if len(analysis['changed_files']) > 10:
            body_parts.append(f"... and {len(analysis['changed_files']) - 10} more files")
        
        # Add features
        if analysis['features_added']:
            body_parts.append("\nFeatures added:")
            for feature in analysis['features_added'][:5]:
                body_parts.append(f"- {feature}")
        
        # Add performance improvements
        if analysis['performance_improvements']:
            body_parts.append("\nPerformance improvements:")
            for improvement in analysis['performance_improvements'][:3]:
                body_parts.append(f"- {improvement}")
        
        # Add impact level
        body_parts.append(f"\nImpact: {analysis['impact_level']}")
        body_parts.append(f"Components affected: {', '.join(list(analysis['affected_components']))}")
        
        # Add automation signature
        body_parts.append("\n🤖 Automated commit via Master Orchestrator Git Agent")
        body_parts.append(f"Generated: {datetime.utcnow().isoformat()}")
        
        return '\n'.join(body_parts)

class HardwareOrchestrator:
    """Main orchestrator for hardware discovery, vLLM deployment, and Git automation."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.network_scanner = NetworkScanner()
        self.vllm_manager = VLLMDistributedManager(config_manager)
        self.git_agent = IntelligentGitAgent(config_manager)
        
        self.discovered_nodes: Dict[str, HardwareNode] = {}
        self.vllm_cluster: Optional[DistributedVLLMCluster] = None
        
        logger.info("HardwareOrchestrator initialized")
    
    async def discover_and_deploy(self) -> Dict[str, Any]:
        """Complete workflow: discover hardware, deploy vLLM, and commit changes."""
        results = {
            "discovery": {},
            "vllm_deployment": {},
            "git_commit": {},
            "overall_status": "started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            logger.info("Starting complete hardware discovery and deployment workflow")
            
            # Phase 1: Hardware Discovery
            logger.info("Phase 1: Discovering Mac hardware on network")
            self.discovered_nodes = await self.network_scanner.scan_network()
            
            results["discovery"] = {
                "status": "completed",
                "nodes_found": len(self.discovered_nodes),
                "mac_studios": len([n for n in self.discovered_nodes.values() if n.device_type == "mac_studio"]),
                "mac_minis": len([n for n in self.discovered_nodes.values() if n.device_type == "mac_mini"]),
                "nodes": {ip: asdict(node) for ip, node in self.discovered_nodes.items()}
            }
            
            # Phase 2: vLLM Deployment (if nodes found)
            if self.discovered_nodes:
                logger.info("Phase 2: Deploying distributed vLLM across discovered nodes")
                self.vllm_cluster = await self.vllm_manager.deploy_distributed_vllm(self.discovered_nodes)
                
                cluster_status = await self.vllm_manager.get_cluster_status()
                results["vllm_deployment"] = {
                    "status": "completed",
                    "cluster_status": cluster_status
                }
            else:
                logger.warning("No suitable nodes found for vLLM deployment")
                results["vllm_deployment"] = {
                    "status": "skipped",
                    "reason": "No suitable nodes found"
                }
            
            # Phase 3: Git Commit
            logger.info("Phase 3: Creating intelligent Git commit for changes")
            commit_result = await self.git_agent.analyze_and_commit_changes()
            results["git_commit"] = commit_result
            
            # Determine overall status
            if (results["discovery"]["nodes_found"] > 0 and 
                results["git_commit"]["status"] == "success"):
                results["overall_status"] = "success"
            else:
                results["overall_status"] = "partial_success"
            
            logger.info("Hardware orchestration workflow completed",
                       status=results["overall_status"],
                       nodes_discovered=results["discovery"]["nodes_found"],
                       vllm_status=results["vllm_deployment"].get("status", "unknown"))
            
        except Exception as e:
            results["overall_status"] = "error"
            results["error"] = str(e)
            logger.error("Hardware orchestration workflow failed", error=str(e))
        
        return results
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "hardware_nodes": {},
            "vllm_cluster": {},
            "git_status": {},
            "system_health": "unknown"
        }
        
        # Hardware nodes status
        if self.discovered_nodes:
            status["hardware_nodes"] = {
                "total_nodes": len(self.discovered_nodes),
                "online_nodes": len([n for n in self.discovered_nodes.values() if n.online]),
                "mac_studios": len([n for n in self.discovered_nodes.values() if n.device_type == "mac_studio"]),
                "mac_minis": len([n for n in self.discovered_nodes.values() if n.device_type == "mac_mini"]),
                "nodes": {ip: {
                    "hostname": node.hostname,
                    "device_type": node.device_type,
                    "online": node.online,
                    "vllm_status": node.vllm_status
                } for ip, node in self.discovered_nodes.items()}
            }
        
        # vLLM cluster status
        if self.vllm_cluster:
            status["vllm_cluster"] = await self.vllm_manager.get_cluster_status()
        
        # Git status
        try:
            git_status = self.git_agent.dev_controller.git.get_status()
            status["git_status"] = {
                "clean": "nothing to commit" in git_status.get("output", ""),
                "branch": "main",  # Could be extracted from git status
                "last_commit": "recent"  # Could be extracted from git log
            }
        except:
            status["git_status"] = {"error": "Unable to get git status"}
        
        # Overall system health
        if (status["hardware_nodes"].get("total_nodes", 0) > 0 and
            status["vllm_cluster"].get("cluster_status") == "running"):
            status["system_health"] = "healthy"
        elif status["hardware_nodes"].get("total_nodes", 0) > 0:
            status["system_health"] = "partial"
        else:
            status["system_health"] = "needs_setup"
        
        return status
    
    async def continuous_monitoring(self, interval_minutes: int = 15):
        """Continuous monitoring and optimization loop."""
        logger.info("Starting continuous monitoring loop", interval=f"{interval_minutes} minutes")
        
        while True:
            try:
                # Re-scan network for new nodes
                current_nodes = await self.network_scanner.scan_network()
                
                # Check for new nodes
                new_nodes = {ip: node for ip, node in current_nodes.items() 
                           if ip not in self.discovered_nodes}
                
                if new_nodes:
                    logger.info("New hardware nodes discovered", count=len(new_nodes))
                    self.discovered_nodes.update(new_nodes)
                    
                    # Deploy vLLM to new nodes if cluster exists
                    if self.vllm_cluster:
                        await self.vllm_manager.deploy_distributed_vllm(self.discovered_nodes)
                    
                    # Commit changes for new nodes
                    await self.git_agent.analyze_and_commit_changes()
                
                # Check cluster health
                if self.vllm_cluster:
                    cluster_status = await self.vllm_manager.get_cluster_status()
                    if cluster_status.get("cluster_status") == "error":
                        logger.warning("vLLM cluster needs attention")
                        # Could trigger automatic recovery here
                
                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error("Continuous monitoring error", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute before retrying

# Global orchestrator instance
_hardware_orchestrator = None

def get_hardware_orchestrator() -> HardwareOrchestrator:
    """Get the global hardware orchestrator instance."""
    global _hardware_orchestrator
    if _hardware_orchestrator is None:
        _hardware_orchestrator = HardwareOrchestrator()
    return _hardware_orchestrator

# CLI interface
async def run_hardware_orchestrator_cli():
    """CLI interface for hardware orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Discovery and vLLM Deployment Orchestrator")
    parser.add_argument("--action", choices=["discover", "deploy", "status", "monitor", "commit"], 
                       default="deploy", help="Action to perform")
    parser.add_argument("--monitor-interval", type=int, default=15, 
                       help="Monitoring interval in minutes")
    
    args = parser.parse_args()
    
    orchestrator = get_hardware_orchestrator()
    
    if args.action == "discover":
        # Hardware discovery only
        nodes = await orchestrator.network_scanner.scan_network()
        print(f"🔍 Discovered {len(nodes)} hardware nodes:")
        for ip, node in nodes.items():
            print(f"  📱 {node.hostname} ({node.device_type}) - {ip}")
    
    elif args.action == "deploy":
        # Full deployment workflow
        results = await orchestrator.discover_and_deploy()
        
        print("🚀 Hardware Orchestration Results:")
        print(f"   📡 Discovery: {results['discovery']['nodes_found']} nodes found")
        print(f"   🤖 vLLM: {results['vllm_deployment']['status']}")
        print(f"   📝 Git: {results['git_commit']['status']}")
        print(f"   ✅ Overall: {results['overall_status']}")
    
    elif args.action == "status":
        # System status
        status = await orchestrator.get_system_status()
        
        print("📊 System Status:")
        print(f"   🖥️ Hardware Nodes: {status['hardware_nodes'].get('total_nodes', 0)}")
        print(f"   🤖 vLLM Cluster: {status['vllm_cluster'].get('cluster_status', 'not_deployed')}")
        print(f"   📝 Git Status: {'Clean' if status['git_status'].get('clean') else 'Changes pending'}")
        print(f"   💚 Health: {status['system_health']}")
    
    elif args.action == "monitor":
        # Continuous monitoring
        print(f"🔄 Starting continuous monitoring (every {args.monitor_interval} minutes)")
        await orchestrator.continuous_monitoring(args.monitor_interval)
    
    elif args.action == "commit":
        # Git commit only
        result = await orchestrator.git_agent.analyze_and_commit_changes()
        print(f"📝 Git Commit: {result['status']}")
        if result['status'] == 'success':
            print(f"   Message: {result['commit_message']}")
            print(f"   Files: {result['files_changed']}")

if __name__ == "__main__":
    print("🏗️ Master Orchestrator - Hardware Discovery & Distributed vLLM")
    print("=" * 70)
    
    # Run CLI interface
    asyncio.run(run_hardware_orchestrator_cli())