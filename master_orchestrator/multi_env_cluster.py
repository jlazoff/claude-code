#!/usr/bin/env python3

"""
Multi-Environment Clustering System
Production, Staging, and Development environments with round-robin load balancing,
health checks, auto-scaling, and comprehensive monitoring integration
"""

import asyncio
import json
import time
import docker
import kubernetes
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
import subprocess
import yaml
import structlog

from unified_config import get_config_manager, EnvironmentConfig
from litellm_manager import get_llm_manager

logger = structlog.get_logger()

class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

@dataclass
class ServiceInstance:
    """Represents a single service instance."""
    
    id: str
    name: str
    environment: EnvironmentType
    host: str
    port: int
    health_endpoint: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    request_count: int = 0
    error_count: int = 0
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    
    strategy: str = "round_robin"  # round_robin, weighted, least_connections, health_based
    health_check_interval: int = 30  # seconds
    max_retries: int = 3
    timeout: int = 5
    sticky_sessions: bool = False
    weights: Dict[str, float] = field(default_factory=dict)

@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration."""
    
    enabled: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 80.0  # CPU percentage
    scale_down_threshold: float = 20.0  # CPU percentage
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    metrics_window: int = 300  # seconds

class ServiceRegistry:
    """Service discovery and registry."""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.load_balancer_configs: Dict[str, LoadBalancerConfig] = {}
        self.auto_scaling_configs: Dict[str, AutoScalingConfig] = {}
        self._lock = threading.RLock()
        
        # Round-robin counters
        self._round_robin_counters: Dict[str, int] = {}
    
    def register_service(self, service: ServiceInstance):
        """Register a service instance."""
        with self._lock:
            if service.name not in self.services:
                self.services[service.name] = []
                self._round_robin_counters[service.name] = 0
            
            # Remove existing instance with same ID
            self.services[service.name] = [
                s for s in self.services[service.name] if s.id != service.id
            ]
            
            # Add new instance
            self.services[service.name].append(service)
            
            logger.info("Service registered", 
                       service=service.name, 
                       instance_id=service.id,
                       environment=service.environment.value)
    
    def deregister_service(self, service_name: str, instance_id: str):
        """Deregister a service instance."""
        with self._lock:
            if service_name in self.services:
                self.services[service_name] = [
                    s for s in self.services[service_name] if s.id != instance_id
                ]
                
                logger.info("Service deregistered", 
                           service=service_name, 
                           instance_id=instance_id)
    
    def get_healthy_instances(self, service_name: str, 
                            environment: Optional[EnvironmentType] = None) -> List[ServiceInstance]:
        """Get healthy instances for a service."""
        with self._lock:
            if service_name not in self.services:
                return []
            
            instances = self.services[service_name]
            
            # Filter by environment if specified
            if environment:
                instances = [s for s in instances if s.environment == environment]
            
            # Filter by health status
            healthy_instances = [s for s in instances if s.status == ServiceStatus.HEALTHY]
            
            return healthy_instances
    
    def get_next_instance(self, service_name: str, 
                         environment: Optional[EnvironmentType] = None) -> Optional[ServiceInstance]:
        """Get next instance using load balancing strategy."""
        healthy_instances = self.get_healthy_instances(service_name, environment)
        
        if not healthy_instances:
            return None
        
        config = self.load_balancer_configs.get(service_name, LoadBalancerConfig())
        
        if config.strategy == "round_robin":
            with self._lock:
                counter = self._round_robin_counters.get(service_name, 0)
                instance = healthy_instances[counter % len(healthy_instances)]
                self._round_robin_counters[service_name] = (counter + 1) % len(healthy_instances)
                return instance
        
        elif config.strategy == "least_connections":
            # Choose instance with lowest request count
            return min(healthy_instances, key=lambda s: s.request_count)
        
        elif config.strategy == "health_based":
            # Choose instance with best response time and lowest error rate
            def health_score(instance):
                error_rate = instance.error_count / max(instance.request_count, 1)
                return instance.response_time + (error_rate * 1000)  # Penalize errors
            
            return min(healthy_instances, key=health_score)
        
        elif config.strategy == "weighted":
            # Weighted random selection
            import random
            
            weights = []
            for instance in healthy_instances:
                weight = config.weights.get(instance.id, 1.0)
                # Adjust weight based on health
                if instance.cpu_usage > 80:
                    weight *= 0.5
                weights.append(weight)
            
            if sum(weights) > 0:
                return random.choices(healthy_instances, weights=weights)[0]
            else:
                return healthy_instances[0]
        
        else:
            return healthy_instances[0]

class HealthChecker:
    """Health checking service for all instances."""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.running = False
        self.check_thread = None
    
    def start(self):
        """Start health checking."""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.check_thread.start()
        
        logger.info("Health checker started")
    
    def stop(self):
        """Stop health checking."""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        
        logger.info("Health checker stopped")
    
    def _health_check_loop(self):
        """Main health checking loop."""
        while self.running:
            try:
                self._check_all_services()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error("Health check error", error=str(e))
    
    def _check_all_services(self):
        """Check health of all registered services."""
        futures = []
        
        for service_name, instances in self.service_registry.services.items():
            for instance in instances:
                future = self.executor.submit(self._check_instance, instance)
                futures.append(future)
        
        # Wait for all checks to complete
        for future in futures:
            try:
                future.result(timeout=10)
            except Exception as e:
                logger.warning("Health check failed", error=str(e))
    
    def _check_instance(self, instance: ServiceInstance):
        """Check health of a single instance."""
        try:
            start_time = time.time()
            
            # Make health check request
            response = requests.get(
                f"http://{instance.host}:{instance.port}{instance.health_endpoint}",
                timeout=5
            )
            
            response_time = time.time() - start_time
            
            # Update instance metrics
            instance.last_health_check = datetime.utcnow().isoformat()
            instance.response_time = response_time
            
            if response.status_code == 200:
                instance.status = ServiceStatus.HEALTHY
                
                # Try to extract metrics from response
                try:
                    health_data = response.json()
                    instance.cpu_usage = health_data.get('cpu_usage', 0.0)
                    instance.memory_usage = health_data.get('memory_usage', 0.0)
                    instance.request_count = health_data.get('request_count', 0)
                    instance.error_count = health_data.get('error_count', 0)
                except:
                    pass
            else:
                instance.status = ServiceStatus.UNHEALTHY
                
        except Exception as e:
            instance.status = ServiceStatus.UNHEALTHY
            instance.last_health_check = datetime.utcnow().isoformat()
            logger.debug("Instance health check failed", 
                        instance=instance.id, 
                        error=str(e))

class AutoScaler:
    """Auto-scaling service for managing instance counts."""
    
    def __init__(self, service_registry: ServiceRegistry, cluster_manager):
        self.service_registry = service_registry
        self.cluster_manager = cluster_manager
        self.last_scale_actions: Dict[str, datetime] = {}
        self.running = False
        self.scale_thread = None
    
    def start(self):
        """Start auto-scaling."""
        if self.running:
            return
        
        self.running = True
        self.scale_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scale_thread.start()
        
        logger.info("Auto-scaler started")
    
    def stop(self):
        """Stop auto-scaling."""
        self.running = False
        if self.scale_thread:
            self.scale_thread.join(timeout=5)
        
        logger.info("Auto-scaler stopped")
    
    def _scaling_loop(self):
        """Main scaling loop."""
        while self.running:
            try:
                self._check_scaling_needs()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error("Auto-scaling error", error=str(e))
    
    def _check_scaling_needs(self):
        """Check if any services need scaling."""
        for service_name, instances in self.service_registry.services.items():
            config = self.service_registry.auto_scaling_configs.get(
                service_name, AutoScalingConfig()
            )
            
            if not config.enabled:
                continue
            
            healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
            
            if not healthy_instances:
                continue
            
            # Calculate average metrics
            avg_cpu = sum(i.cpu_usage for i in healthy_instances) / len(healthy_instances)
            current_count = len(healthy_instances)
            
            # Check for scale up
            if (avg_cpu > config.scale_up_threshold and 
                current_count < config.max_instances and
                self._can_scale(service_name, config.scale_up_cooldown)):
                
                self._scale_up(service_name, current_count + 1)
            
            # Check for scale down
            elif (avg_cpu < config.scale_down_threshold and 
                  current_count > config.min_instances and
                  self._can_scale(service_name, config.scale_down_cooldown)):
                
                self._scale_down(service_name, current_count - 1)
    
    def _can_scale(self, service_name: str, cooldown: int) -> bool:
        """Check if service can be scaled (cooldown period)."""
        last_action = self.last_scale_actions.get(service_name)
        if not last_action:
            return True
        
        return (datetime.utcnow() - last_action).seconds >= cooldown
    
    def _scale_up(self, service_name: str, target_count: int):
        """Scale up a service."""
        try:
            self.cluster_manager.scale_service(service_name, target_count)
            self.last_scale_actions[service_name] = datetime.utcnow()
            
            logger.info("Service scaled up", 
                       service=service_name, 
                       target_count=target_count)
            
        except Exception as e:
            logger.error("Scale up failed", 
                        service=service_name, 
                        error=str(e))
    
    def _scale_down(self, service_name: str, target_count: int):
        """Scale down a service."""
        try:
            self.cluster_manager.scale_service(service_name, target_count)
            self.last_scale_actions[service_name] = datetime.utcnow()
            
            logger.info("Service scaled down", 
                       service=service_name, 
                       target_count=target_count)
            
        except Exception as e:
            logger.error("Scale down failed", 
                        service=service_name, 
                        error=str(e))

class ClusterManager:
    """Main cluster management orchestrator."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.service_registry = ServiceRegistry()
        self.health_checker = HealthChecker(self.service_registry)
        self.auto_scaler = AutoScaler(self.service_registry, self)
        
        # Docker and Kubernetes clients
        self.docker_client = None
        self.k8s_client = None
        
        # Service definitions
        self.service_definitions = {}
        
        self._initialize_clients()
        self._load_service_definitions()
        
        logger.info("ClusterManager initialized")
    
    def _initialize_clients(self):
        """Initialize Docker and Kubernetes clients."""
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning("Docker client initialization failed", error=str(e))
        
        try:
            kubernetes.config.load_incluster_config()
            self.k8s_client = kubernetes.client.ApiClient()
            logger.info("Kubernetes client initialized (in-cluster)")
        except:
            try:
                kubernetes.config.load_kube_config()
                self.k8s_client = kubernetes.client.ApiClient()
                logger.info("Kubernetes client initialized (kubeconfig)")
            except Exception as e:
                logger.warning("Kubernetes client initialization failed", error=str(e))
    
    def _load_service_definitions(self):
        """Load service definitions from configuration."""
        
        # Master Orchestrator API
        self.service_definitions["orchestrator-api"] = {
            "image": "master-orchestrator:latest",
            "ports": [8000],
            "health_endpoint": "/health",
            "environment_vars": {
                "ENVIRONMENT": "{{environment}}",
                "CONFIG_PATH": "/app/config"
            },
            "resource_limits": {
                "cpu": "1000m",
                "memory": "2Gi"
            },
            "auto_scaling": AutoScalingConfig(
                min_instances=1,
                max_instances=5,
                scale_up_threshold=70.0
            )
        }
        
        # Live Dashboard
        self.service_definitions["live-dashboard"] = {
            "image": "master-orchestrator-dashboard:latest",
            "ports": [8001],
            "health_endpoint": "/health",
            "environment_vars": {
                "API_ENDPOINT": "{{orchestrator_api_endpoint}}",
                "ENVIRONMENT": "{{environment}}"
            },
            "resource_limits": {
                "cpu": "500m",
                "memory": "1Gi"
            },
            "auto_scaling": AutoScalingConfig(
                min_instances=1,
                max_instances=3
            )
        }
        
        # Agent Workers
        self.service_definitions["agent-worker"] = {
            "image": "master-orchestrator-agent:latest",
            "ports": [8002],
            "health_endpoint": "/health",
            "environment_vars": {
                "ORCHESTRATOR_ENDPOINT": "{{orchestrator_api_endpoint}}",
                "WORKER_TYPE": "general",
                "ENVIRONMENT": "{{environment}}"
            },
            "resource_limits": {
                "cpu": "2000m",
                "memory": "4Gi"
            },
            "auto_scaling": AutoScalingConfig(
                min_instances=2,
                max_instances=20,
                scale_up_threshold=80.0
            )
        }
        
        logger.info("Service definitions loaded", services=list(self.service_definitions.keys()))
    
    def start(self):
        """Start the cluster manager."""
        self.health_checker.start()
        self.auto_scaler.start()
        
        # Deploy services for all environments
        self._deploy_all_environments()
        
        logger.info("Cluster manager started")
    
    def stop(self):
        """Stop the cluster manager."""
        self.health_checker.stop()
        self.auto_scaler.stop()
        
        logger.info("Cluster manager stopped")
    
    def _deploy_all_environments(self):
        """Deploy services to all configured environments."""
        for env_name, env_config in self.config_manager.environments.items():
            if env_config.is_active:
                try:
                    self.deploy_environment(EnvironmentType(env_name))
                except Exception as e:
                    logger.error("Environment deployment failed", 
                                environment=env_name, 
                                error=str(e))
    
    def deploy_environment(self, environment: EnvironmentType):
        """Deploy all services for an environment."""
        env_config = self.config_manager.environments.get(environment.value)
        if not env_config:
            raise ValueError(f"Environment {environment.value} not configured")
        
        logger.info("Deploying environment", environment=environment.value)
        
        # Deploy each service
        for service_name, service_def in self.service_definitions.items():
            try:
                self._deploy_service(service_name, service_def, environment, env_config)
            except Exception as e:
                logger.error("Service deployment failed", 
                            service=service_name, 
                            environment=environment.value,
                            error=str(e))
    
    def _deploy_service(self, service_name: str, service_def: Dict[str, Any], 
                       environment: EnvironmentType, env_config: EnvironmentConfig):
        """Deploy a single service."""
        
        # Determine deployment target
        if self.k8s_client and environment != EnvironmentType.DEVELOPMENT:
            self._deploy_service_k8s(service_name, service_def, environment, env_config)
        else:
            self._deploy_service_docker(service_name, service_def, environment, env_config)
    
    def _deploy_service_docker(self, service_name: str, service_def: Dict[str, Any], 
                              environment: EnvironmentType, env_config: EnvironmentConfig):
        """Deploy service using Docker."""
        if not self.docker_client:
            raise Exception("Docker client not available")
        
        # Prepare environment variables
        env_vars = {}
        for key, value in service_def.get("environment_vars", {}).items():
            # Template substitution
            if value == "{{environment}}":
                env_vars[key] = environment.value
            elif value == "{{orchestrator_api_endpoint}}":
                env_vars[key] = env_config.service_endpoints.get("orchestrator_api", "http://localhost:8000")
            else:
                env_vars[key] = value
        
        # Get auto-scaling config
        auto_scaling = service_def.get("auto_scaling", AutoScalingConfig())
        
        # Deploy minimum number of instances
        for i in range(auto_scaling.min_instances):
            container_name = f"{service_name}-{environment.value}-{i}"
            
            try:
                # Remove existing container if it exists
                try:
                    existing = self.docker_client.containers.get(container_name)
                    existing.stop()
                    existing.remove()
                except docker.errors.NotFound:
                    pass
                
                # Create and start new container
                container = self.docker_client.containers.run(
                    service_def["image"],
                    name=container_name,
                    environment=env_vars,
                    ports={f"{port}/tcp": None for port in service_def["ports"]},
                    detach=True,
                    restart_policy={"Name": "unless-stopped"}
                )
                
                # Get assigned port
                container.reload()
                host_port = None
                for port in service_def["ports"]:
                    port_info = container.attrs['NetworkSettings']['Ports'].get(f'{port}/tcp')
                    if port_info:
                        host_port = int(port_info[0]['HostPort'])
                        break
                
                if host_port:
                    # Register service instance
                    instance = ServiceInstance(
                        id=container.id[:12],
                        name=service_name,
                        environment=environment,
                        host="localhost",
                        port=host_port,
                        health_endpoint=service_def["health_endpoint"],
                        metadata={
                            "container_name": container_name,
                            "image": service_def["image"]
                        }
                    )
                    
                    self.service_registry.register_service(instance)
                    
                    logger.info("Service instance deployed", 
                               service=service_name,
                               instance_id=instance.id,
                               environment=environment.value,
                               port=host_port)
            
            except Exception as e:
                logger.error("Docker deployment failed", 
                            service=service_name,
                            instance=i,
                            error=str(e))
        
        # Register auto-scaling config
        self.service_registry.auto_scaling_configs[service_name] = auto_scaling
    
    def _deploy_service_k8s(self, service_name: str, service_def: Dict[str, Any], 
                           environment: EnvironmentType, env_config: EnvironmentConfig):
        """Deploy service using Kubernetes."""
        # This would implement Kubernetes deployment
        # For now, just log that it would be deployed
        logger.info("Kubernetes deployment placeholder", 
                   service=service_name,
                   environment=environment.value)
    
    def scale_service(self, service_name: str, target_count: int):
        """Scale a service to target instance count."""
        current_instances = self.service_registry.services.get(service_name, [])
        current_count = len([i for i in current_instances if i.status == ServiceStatus.HEALTHY])
        
        if target_count > current_count:
            # Scale up
            for i in range(target_count - current_count):
                self._add_service_instance(service_name)
        elif target_count < current_count:
            # Scale down
            for i in range(current_count - target_count):
                self._remove_service_instance(service_name)
    
    def _add_service_instance(self, service_name: str):
        """Add a new instance of a service."""
        # This would implement adding a new instance
        logger.info("Adding service instance", service=service_name)
    
    def _remove_service_instance(self, service_name: str):
        """Remove an instance of a service."""
        instances = self.service_registry.services.get(service_name, [])
        if instances:
            # Remove the instance with highest CPU usage
            instance_to_remove = max(instances, key=lambda i: i.cpu_usage)
            self.service_registry.deregister_service(service_name, instance_to_remove.id)
            
            # Stop the actual container/pod
            if self.docker_client:
                try:
                    container = self.docker_client.containers.get(instance_to_remove.id)
                    container.stop()
                    container.remove()
                except:
                    pass
            
            logger.info("Removed service instance", 
                       service=service_name,
                       instance_id=instance_to_remove.id)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "environments": {},
            "services": {},
            "total_instances": 0,
            "healthy_instances": 0,
            "load_balancer_stats": {},
            "auto_scaling_status": {}
        }
        
        # Count instances by environment
        for service_name, instances in self.service_registry.services.items():
            status["services"][service_name] = {
                "total_instances": len(instances),
                "healthy_instances": len([i for i in instances if i.status == ServiceStatus.HEALTHY]),
                "environments": {}
            }
            
            for instance in instances:
                env = instance.environment.value
                if env not in status["environments"]:
                    status["environments"][env] = {"instances": 0, "healthy": 0}
                
                if env not in status["services"][service_name]["environments"]:
                    status["services"][service_name]["environments"][env] = {"instances": 0, "healthy": 0}
                
                status["environments"][env]["instances"] += 1
                status["services"][service_name]["environments"][env]["instances"] += 1
                status["total_instances"] += 1
                
                if instance.status == ServiceStatus.HEALTHY:
                    status["environments"][env]["healthy"] += 1
                    status["services"][service_name]["environments"][env]["healthy"] += 1
                    status["healthy_instances"] += 1
        
        return status
    
    def export_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = []
        timestamp = int(time.time() * 1000)
        
        for service_name, instances in self.service_registry.services.items():
            for instance in instances:
                labels = f'service="{service_name}",environment="{instance.environment.value}",instance_id="{instance.id}"'
                
                metrics.append(f'service_health{{labels}} {1 if instance.status == ServiceStatus.HEALTHY else 0} {timestamp}')
                metrics.append(f'service_cpu_usage{{labels}} {instance.cpu_usage} {timestamp}')
                metrics.append(f'service_memory_usage{{labels}} {instance.memory_usage} {timestamp}')
                metrics.append(f'service_request_count{{labels}} {instance.request_count} {timestamp}')
                metrics.append(f'service_error_count{{labels}} {instance.error_count} {timestamp}')
                metrics.append(f'service_response_time{{labels}} {instance.response_time} {timestamp}')
        
        return '\n'.join(metrics)


# Global cluster manager instance
_cluster_manager = None

def get_cluster_manager() -> ClusterManager:
    """Get the global cluster manager instance."""
    global _cluster_manager
    if _cluster_manager is None:
        _cluster_manager = ClusterManager()
    return _cluster_manager


if __name__ == "__main__":
    # Example usage and testing
    
    print("🏗️ Master Orchestrator - Multi-Environment Cluster Manager")
    print("=" * 65)
    
    # Initialize cluster manager
    cluster_manager = get_cluster_manager()
    
    try:
        # Start cluster management
        cluster_manager.start()
        
        print("✅ Cluster manager started")
        print("🚀 Services deploying...")
        
        # Wait a bit for initial deployment
        time.sleep(5)
        
        # Show cluster status
        status = cluster_manager.get_cluster_status()
        print(f"\n📊 Cluster Status:")
        print(f"   Total instances: {status['total_instances']}")
        print(f"   Healthy instances: {status['healthy_instances']}")
        print(f"   Environments: {list(status['environments'].keys())}")
        print(f"   Services: {list(status['services'].keys())}")
        
        # Show Prometheus metrics sample
        print(f"\n📈 Sample Prometheus metrics:")
        metrics = cluster_manager.export_metrics_for_prometheus()
        print(metrics[:500] + "..." if len(metrics) > 500 else metrics)
        
        print(f"\n🎯 Cluster management system is running!")
        print(f"   • Health checks: Every 30 seconds")
        print(f"   • Auto-scaling: Every 60 seconds") 
        print(f"   • Load balancing: Round-robin with health checks")
        print(f"   • Multi-environment: Development, Staging, Production")
        
        # Keep running for demonstration
        print(f"\n⏱️ Running for 30 seconds to demonstrate...")
        time.sleep(30)
        
        # Final status
        final_status = cluster_manager.get_cluster_status()
        print(f"\n📊 Final Status:")
        print(f"   Healthy instances: {final_status['healthy_instances']}")
        
    except KeyboardInterrupt:
        print(f"\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Clean shutdown
        cluster_manager.stop()
        print(f"🔚 Cluster manager stopped")