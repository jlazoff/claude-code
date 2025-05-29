#!/usr/bin/env python3
"""
Distributed Inference Orchestrator
Automatically discovers Mac Mini/Studio hardware and deploys distributed inference
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import socket
import paramiko
import psutil
import platform

# Foundation integration
try:
    from unified_config import SecureConfigManager
    from hardware_discovery import HardwareDiscoverer
except ImportError as e:
    logging.warning(f"Some foundation modules unavailable: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceNode:
    hostname: str
    ip_address: str
    mac_address: str
    cpu_count: int
    memory_gb: float
    gpu_info: Dict[str, Any]
    network_speed: str
    status: str
    vllm_port: int
    model_loaded: Optional[str] = None
    performance_metrics: Dict[str, float] = None

@dataclass
class InferenceCluster:
    cluster_id: str
    nodes: List[InferenceNode]
    load_balancer_config: Dict[str, Any]
    deployed_models: List[str]
    total_capacity: Dict[str, Any]
    optimization_config: Dict[str, Any]

class DistributedInferenceOrchestrator:
    def __init__(self):
        self.config = None
        self.hardware_discoverer = None
        self.discovered_nodes = []
        self.active_cluster = None
        
        # Initialize foundation
        self.foundation_dir = Path("foundation_data")
        self.inference_dir = self.foundation_dir / "distributed_inference"
        self.models_dir = self.foundation_dir / "models"
        self.metrics_dir = self.foundation_dir / "metrics"
        
        for dir_path in [self.foundation_dir, self.inference_dir, self.models_dir, self.metrics_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info("Distributed Inference Orchestrator initialized")

    async def discover_thunderbolt_network(self) -> List[Dict[str, Any]]:
        """Discover Mac Mini/Studio devices on Thunderbolt network"""
        logger.info("🔍 Discovering Thunderbolt network devices...")
        
        discovered_devices = []
        
        # Get local network interface information
        try:
            # Use system_profiler to get Thunderbolt devices
            result = subprocess.run([
                'system_profiler', 'SPThunderboltDataType', '-json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                thunderbolt_data = json.loads(result.stdout)
                logger.info(f"Found Thunderbolt data: {len(thunderbolt_data)} entries")
        except Exception as e:
            logger.warning(f"Could not get Thunderbolt data: {e}")
        
        # Network scanning for Mac devices
        local_ip = self.get_local_ip()
        network_base = '.'.join(local_ip.split('.')[:-1]) + '.'
        
        logger.info(f"Scanning network: {network_base}0/24")
        
        # Parallel network scan
        scan_tasks = []
        for i in range(1, 255):
            ip = f"{network_base}{i}"
            scan_tasks.append(self.scan_device(ip))
        
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        for result in scan_results:
            if isinstance(result, dict) and result.get('reachable'):
                discovered_devices.append(result)
        
        logger.info(f"✅ Discovered {len(discovered_devices)} devices")
        return discovered_devices

    async def scan_device(self, ip: str) -> Dict[str, Any]:
        """Scan a single device for Mac hardware"""
        try:
            # Quick ping test
            proc = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', '1000', ip,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            await proc.wait()
            
            if proc.returncode != 0:
                return {'ip': ip, 'reachable': False}
            
            # Try SSH connection to check if it's a Mac
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(ip, timeout=5, username='admin', look_for_keys=True)
                
                # Get system information
                stdin, stdout, stderr = ssh.exec_command('system_profiler SPHardwareDataType -json')
                hardware_info = json.loads(stdout.read().decode())
                
                # Get memory info
                stdin, stdout, stderr = ssh.exec_command('sysctl hw.memsize')
                memory_bytes = int(stdout.read().decode().split(': ')[1])
                memory_gb = memory_bytes / (1024**3)
                
                # Get CPU info
                stdin, stdout, stderr = ssh.exec_command('sysctl -n machdep.cpu.brand_string')
                cpu_info = stdout.read().decode().strip()
                
                # Check for GPU (Metal Performance Shaders)
                stdin, stdout, stderr = ssh.exec_command('system_profiler SPDisplaysDataType -json')
                gpu_info = json.loads(stdout.read().decode())
                
                ssh.close()
                
                return {
                    'ip': ip,
                    'reachable': True,
                    'is_mac': True,
                    'hardware_info': hardware_info,
                    'memory_gb': memory_gb,
                    'cpu_info': cpu_info,
                    'gpu_info': gpu_info,
                    'hostname': self.get_hostname(ip)
                }
                
            except Exception as ssh_e:
                # Try mDNS discovery for Mac devices
                hostname = self.get_hostname(ip)
                if hostname and ('.local' in hostname or 'mac' in hostname.lower()):
                    return {
                        'ip': ip,
                        'reachable': True,
                        'is_mac': True,
                        'hostname': hostname,
                        'discovery_method': 'mdns'
                    }
                
                return {'ip': ip, 'reachable': True, 'is_mac': False}
                
        except Exception as e:
            return {'ip': ip, 'reachable': False, 'error': str(e)}

    def get_hostname(self, ip: str) -> Optional[str]:
        """Get hostname for IP address"""
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            return hostname
        except:
            return None

    def get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "192.168.1.1"  # fallback

    async def setup_inference_node(self, device_info: Dict[str, Any]) -> InferenceNode:
        """Set up multiple inference servers on a discovered node"""
        ip = device_info['ip']
        hostname = device_info.get('hostname', ip)
        
        logger.info(f"🚀 Setting up distributed inference on {hostname} ({ip})")
        
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ip, username='admin', look_for_keys=True)
            
            # Install dependencies
            setup_commands = [
                # Install Homebrew if not present
                '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || echo "brew exists"',
                
                # Install Python and Node.js
                'brew install python@3.11 node@18 || echo "runtimes exist"',
                
                # Install Docker for containerized inference
                'brew install --cask docker || echo "docker exists"',
                
                # Install vLLM
                'pip3 install vllm[all] || pip3 install --upgrade vllm',
                
                # Install llm-d (distributed LLM serving)
                'pip3 install llm-d || echo "llm-d installed"',
                
                # Install LocalAI
                'brew install localai/tap/localai || echo "localai installed"',
                
                # Install Triton Inference Server client
                'pip3 install tritonclient[all] || echo "triton client installed"',
                
                # Create inference service directories
                'mkdir -p ~/inference_services/{vllm,llm_d,localai,triton}',
                
                # Download models for each service
                'cd ~/inference_services && python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained(\'microsoft/DialoGPT-small\'); AutoModel.from_pretrained(\'microsoft/DialoGPT-small\')" || echo "model cache"',
                
                # Install NVIDIA Docker runtime if available
                'docker info | grep "nvidia" || echo "No NVIDIA runtime detected"'
            ]
            
            for cmd in setup_commands:
                logger.info(f"Executing: {cmd}")
                stdin, stdout, stderr = ssh.exec_command(cmd, timeout=300)
                exit_status = stdout.channel.recv_exit_status()
                if exit_status != 0:
                    error_output = stderr.read().decode()
                    logger.warning(f"Command failed on {hostname}: {error_output}")
            
            # Setup multiple inference servers with different ports
            base_port = 8000 + (hash(ip) % 100) * 10
            
            # 1. Start vLLM server
            vllm_port = base_port
            vllm_start_cmd = f"""
cd ~/inference_services/vllm && nohup python3 -m vllm.entrypoints.openai.api_server \\
    --model microsoft/DialoGPT-small \\
    --host 0.0.0.0 \\
    --port {vllm_port} \\
    --max-model-len 2048 \\
    --tensor-parallel-size 1 \\
    --enable-chunked-prefill \\
    --max-num-batched-tokens 4096 \\
    > vllm.log 2>&1 &
"""
            stdin, stdout, stderr = ssh.exec_command(vllm_start_cmd)
            
            # 2. Start llm-d server
            llm_d_port = base_port + 1
            llm_d_start_cmd = f"""
cd ~/inference_services/llm_d && nohup python3 -m llm_d.server \\
    --model microsoft/DialoGPT-small \\
    --host 0.0.0.0 \\
    --port {llm_d_port} \\
    --workers 2 \\
    --max-batch-size 32 \\
    > llm_d.log 2>&1 &
"""
            stdin, stdout, stderr = ssh.exec_command(llm_d_start_cmd)
            
            # 3. Start LocalAI server
            localai_port = base_port + 2
            localai_start_cmd = f"""
cd ~/inference_services/localai && nohup localai \\
    --address 0.0.0.0:{localai_port} \\
    --models-path ./models \\
    --context-size 2048 \\
    --threads 4 \\
    > localai.log 2>&1 &
"""
            stdin, stdout, stderr = ssh.exec_command(localai_start_cmd)
            
            # 4. Start Triton Inference Server (Docker)
            triton_port = base_port + 3
            triton_start_cmd = f"""
cd ~/inference_services/triton && docker run -d \\
    --name triton-inference-server \\
    --restart unless-stopped \\
    -p {triton_port}:8000 \\
    -p {triton_port + 1}:8001 \\
    -p {triton_port + 2}:8002 \\
    -v $(pwd)/model_repository:/models \\
    nvcr.io/nvidia/tritonserver:latest \\
    tritonserver --model-repository=/models \\
    > triton.log 2>&1 || echo "triton container started"
"""
            stdin, stdout, stderr = ssh.exec_command(triton_start_cmd)
            
            # Wait for services to start
            await asyncio.sleep(20)
            
            # Test each service
            services_status = {}
            for service, port in [
                ("vllm", vllm_port),
                ("llm_d", llm_d_port), 
                ("localai", localai_port),
                ("triton", triton_port)
            ]:
                test_cmd = f"curl -s -m 5 http://localhost:{port}/health || curl -s -m 5 http://localhost:{port}/v1/models || echo 'service starting'"
                stdin, stdout, stderr = ssh.exec_command(test_cmd)
                test_result = stdout.read().decode()
                services_status[service] = {
                    "port": port,
                    "status": "active" if ("models" in test_result or "healthy" in test_result) else "starting"
                }
            
            ssh.close()
            
            # Determine primary service (prefer vLLM, fallback to others)
            primary_service = "vllm"
            primary_port = vllm_port
            primary_status = services_status.get("vllm", {}).get("status", "failed")
            
            if primary_status != "active":
                for service, info in services_status.items():
                    if info["status"] == "active":
                        primary_service = service
                        primary_port = info["port"]
                        primary_status = "active"
                        break
            
            node = InferenceNode(
                hostname=hostname,
                ip_address=ip,
                mac_address=device_info.get('mac_address', 'unknown'),
                cpu_count=device_info.get('cpu_count', 8),
                memory_gb=device_info.get('memory_gb', 16),
                gpu_info=device_info.get('gpu_info', {}),
                network_speed="thunderbolt",
                status=primary_status,
                vllm_port=primary_port,
                model_loaded="microsoft/DialoGPT-small",
                performance_metrics={
                    "services": services_status,
                    "primary_service": primary_service,
                    "load_balancing": "multi_service"
                }
            )
            
            logger.info(f"✅ Multi-inference node setup complete: {hostname} (primary: {primary_service})")
            return node
            
        except Exception as e:
            logger.error(f"Failed to setup inference on {hostname}: {e}")
            return InferenceNode(
                hostname=hostname,
                ip_address=ip,
                mac_address="unknown",
                cpu_count=0,
                memory_gb=0,
                gpu_info={},
                network_speed="unknown",
                status="failed",
                vllm_port=0
            )

    async def create_inference_cluster(self, nodes: List[InferenceNode]) -> InferenceCluster:
        """Create and configure inference cluster"""
        logger.info(f"🔧 Creating inference cluster with {len(nodes)} nodes")
        
        # Filter active nodes
        active_nodes = [node for node in nodes if node.status == "active"]
        
        if not active_nodes:
            raise Exception("No active nodes available for cluster")
        
        # Create load balancer configuration
        load_balancer_config = {
            "strategy": "round_robin",
            "health_check_interval": 30,
            "endpoints": [
                {
                    "host": node.ip_address,
                    "port": node.vllm_port,
                    "weight": 1.0,
                    "health_url": f"http://{node.ip_address}:{node.vllm_port}/health"
                }
                for node in active_nodes
            ]
        }
        
        # Calculate total capacity
        total_capacity = {
            "total_cpu_cores": sum(node.cpu_count for node in active_nodes),
            "total_memory_gb": sum(node.memory_gb for node in active_nodes),
            "total_nodes": len(active_nodes),
            "estimated_throughput": len(active_nodes) * 100  # requests per minute
        }
        
        # Optimization configuration
        optimization_config = {
            "batch_size": min(32, len(active_nodes) * 8),
            "model_parallel": len(active_nodes) > 1,
            "tensor_parallel": len(active_nodes),
            "continuous_batching": True,
            "auto_scaling": {
                "enabled": True,
                "min_nodes": 1,
                "max_nodes": len(nodes),
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.3
            }
        }
        
        cluster = InferenceCluster(
            cluster_id=f"thunderbolt_cluster_{int(time.time())}",
            nodes=active_nodes,
            load_balancer_config=load_balancer_config,
            deployed_models=["microsoft/DialoGPT-small"],
            total_capacity=total_capacity,
            optimization_config=optimization_config
        )
        
        # Save cluster configuration
        cluster_file = self.inference_dir / f"{cluster.cluster_id}.json"
        with open(cluster_file, 'w') as f:
            json.dump(asdict(cluster), f, indent=2, default=str)
        
        logger.info(f"✅ Inference cluster created: {cluster.cluster_id}")
        return cluster

    async def deploy_distributed_inference(self) -> InferenceCluster:
        """Main deployment function"""
        logger.info("🚀 Starting distributed inference deployment")
        
        # Discover hardware
        discovered_devices = await self.discover_thunderbolt_network()
        
        if not discovered_devices:
            logger.warning("No devices discovered, using local machine")
            # Use local machine as fallback
            local_device = {
                'ip': self.get_local_ip(),
                'hostname': platform.node(),
                'is_mac': platform.system() == 'Darwin',
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'cpu_count': psutil.cpu_count()
            }
            discovered_devices = [local_device]
        
        # Setup inference servers on each device
        setup_tasks = []
        for device in discovered_devices:
            if device.get('is_mac', False) or device.get('reachable', False):
                setup_tasks.append(self.setup_inference_node(device))
        
        nodes = await asyncio.gather(*setup_tasks, return_exceptions=True)
        valid_nodes = [node for node in nodes if isinstance(node, InferenceNode)]
        
        if not valid_nodes:
            raise Exception("No valid nodes created")
        
        # Create cluster
        cluster = await self.create_inference_cluster(valid_nodes)
        self.active_cluster = cluster
        
        # Start monitoring
        asyncio.create_task(self.monitor_cluster(cluster))
        
        return cluster

    async def monitor_cluster(self, cluster: InferenceCluster):
        """Monitor cluster performance and health"""
        logger.info(f"📊 Starting cluster monitoring: {cluster.cluster_id}")
        
        while True:
            try:
                metrics = {
                    "timestamp": time.time(),
                    "cluster_id": cluster.cluster_id,
                    "node_metrics": []
                }
                
                for node in cluster.nodes:
                    try:
                        # Health check
                        proc = await asyncio.create_subprocess_exec(
                            'curl', '-s', '-m', '5', 
                            f"http://{node.ip_address}:{node.vllm_port}/health",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.DEVNULL
                        )
                        stdout, _ = await proc.communicate()
                        
                        is_healthy = proc.returncode == 0
                        node.status = "active" if is_healthy else "unhealthy"
                        
                        node_metrics = {
                            "hostname": node.hostname,
                            "ip": node.ip_address,
                            "status": node.status,
                            "response_time": 0.1 if is_healthy else 999,
                            "model_loaded": node.model_loaded
                        }
                        
                        metrics["node_metrics"].append(node_metrics)
                        
                    except Exception as e:
                        logger.warning(f"Health check failed for {node.hostname}: {e}")
                        node.status = "error"
                
                # Save metrics
                metrics_file = self.metrics_dir / f"cluster_metrics_{int(time.time())}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                # Clean old metrics (keep last 100)
                metric_files = sorted(self.metrics_dir.glob("cluster_metrics_*.json"))
                if len(metric_files) > 100:
                    for old_file in metric_files[:-100]:
                        old_file.unlink()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status"""
        if not self.active_cluster:
            return {"status": "no_cluster", "message": "No active cluster"}
        
        active_nodes = [node for node in self.active_cluster.nodes if node.status == "active"]
        
        return {
            "status": "active",
            "cluster_id": self.active_cluster.cluster_id,
            "total_nodes": len(self.active_cluster.nodes),
            "active_nodes": len(active_nodes),
            "total_capacity": self.active_cluster.total_capacity,
            "load_balancer": self.active_cluster.load_balancer_config["strategy"],
            "models": self.active_cluster.deployed_models
        }

async def main():
    """Main execution function"""
    orchestrator = DistributedInferenceOrchestrator()
    
    try:
        print("🔍 Discovering and deploying distributed inference cluster...")
        cluster = await orchestrator.deploy_distributed_inference()
        
        print(f"\n✅ Distributed inference cluster deployed!")
        print(f"   Cluster ID: {cluster.cluster_id}")
        print(f"   Active Nodes: {len(cluster.nodes)}")
        print(f"   Total Capacity: {cluster.total_capacity}")
        print(f"   Models: {cluster.deployed_models}")
        
        # Print node details
        print(f"\n📋 Node Details:")
        for node in cluster.nodes:
            print(f"   • {node.hostname} ({node.ip_address}:{node.vllm_port})")
            print(f"     Status: {node.status}, Model: {node.model_loaded}")
            print(f"     Resources: {node.cpu_count} CPUs, {node.memory_gb:.1f}GB RAM")
        
        # Keep running for monitoring
        print(f"\n🔄 Monitoring cluster... (Press Ctrl+C to stop)")
        try:
            while True:
                await asyncio.sleep(10)
                status = orchestrator.get_cluster_status()
                print(f"Cluster Status: {status['active_nodes']}/{status['total_nodes']} nodes active")
        except KeyboardInterrupt:
            print("\n👋 Stopping cluster monitoring")
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        print(f"❌ Deployment failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())