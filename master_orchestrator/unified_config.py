#!/usr/bin/env python3

"""
Unified Configuration and API Keys Management System
Persistent storage with encryption, environment management, and state preservation
"""

import os
import json
import yaml
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
import logging
from cryptography.fernet import Fernet
import structlog
from contextlib import contextmanager

logger = structlog.get_logger()

@dataclass
class APIKeys:
    """Container for all API keys used across the system."""
    
    # Core LLM Providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    # Local LLM Services
    ollama_endpoint: str = "http://localhost:11434"
    vllm_endpoint: str = "http://localhost:8080"
    localai_endpoint: str = "http://localhost:8080"
    triton_endpoint: str = "http://localhost:8000"
    
    # Vector Databases
    pinecone_api_key: Optional[str] = None
    weaviate_endpoint: str = "http://localhost:8080"
    chroma_endpoint: str = "http://localhost:8000"
    qdrant_endpoint: str = "http://localhost:6333"
    
    # Graph Databases
    arangodb_endpoint: str = "http://localhost:8529"
    arangodb_username: str = "root"
    arangodb_password: Optional[str] = None
    neo4j_endpoint: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: Optional[str] = None
    
    # Cloud Providers
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-west-2"
    gcp_credentials_path: Optional[str] = None
    azure_subscription_id: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None
    
    # Development Tools
    github_token: Optional[str] = None
    gitlab_token: Optional[str] = None
    jira_api_key: Optional[str] = None
    slack_bot_token: Optional[str] = None
    discord_bot_token: Optional[str] = None
    
    # Web Search & External APIs
    serper_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    wolfram_alpha_api_key: Optional[str] = None
    
    # Monitoring & Observability
    prometheus_endpoint: str = "http://localhost:9090"
    grafana_endpoint: str = "http://localhost:3000"
    grafana_api_key: Optional[str] = None
    iceberg_endpoint: str = "http://localhost:8080"
    
    # Infrastructure
    kubernetes_config_path: Optional[str] = None
    docker_registry_url: Optional[str] = None
    docker_registry_username: Optional[str] = None
    docker_registry_password: Optional[str] = None
    
    # Hardware & Network
    nas_endpoint: Optional[str] = None
    nas_username: Optional[str] = None
    nas_password: Optional[str] = None
    cloudflare_api_key: Optional[str] = None
    cloudflare_zone_id: Optional[str] = None


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    
    name: str
    description: str
    api_keys: APIKeys
    database_urls: Dict[str, str] = field(default_factory=dict)
    service_endpoints: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = False


@dataclass
class SystemState:
    """Persistent system state."""
    
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    active_agents: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    repository_analysis: Dict[str, Any] = field(default_factory=dict)
    knowledge_graph_state: Dict[str, Any] = field(default_factory=dict)
    deployment_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics_cache: Dict[str, Any] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)


class SecureConfigManager:
    """Secure configuration and state management with encryption."""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path.home() / ".master_orchestrator"
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        self.config_file = self.config_dir / "config.encrypted"
        self.state_file = self.config_dir / "state.json"
        self.key_file = self.config_dir / ".encryption_key"
        
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
        
        # Environment configurations
        self.environments: Dict[str, EnvironmentConfig] = {}
        self.current_environment = "development"
        
        # System state
        self.system_state = SystemState()
        
        # Load existing configuration
        self._load_configuration()
        self._load_state()
        
        logger.info("SecureConfigManager initialized", 
                   config_dir=str(self.config_dir),
                   environments=list(self.environments.keys()))
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get existing encryption key or create a new one."""
        if self.key_file.exists():
            return self.key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            self.key_file.chmod(0o600)  # Restrict permissions
            return key
    
    def _encrypt_data(self, data: str) -> bytes:
        """Encrypt string data."""
        return self._cipher.encrypt(data.encode('utf-8'))
    
    def _decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt encrypted data."""
        return self._cipher.decrypt(encrypted_data).decode('utf-8')
    
    def _load_configuration(self):
        """Load encrypted configuration."""
        if self.config_file.exists():
            try:
                encrypted_data = self.config_file.read_bytes()
                decrypted_data = self._decrypt_data(encrypted_data)
                config_dict = json.loads(decrypted_data)
                
                for env_name, env_data in config_dict.get('environments', {}).items():
                    api_keys = APIKeys(**env_data.get('api_keys', {}))
                    self.environments[env_name] = EnvironmentConfig(
                        name=env_name,
                        description=env_data.get('description', ''),
                        api_keys=api_keys,
                        database_urls=env_data.get('database_urls', {}),
                        service_endpoints=env_data.get('service_endpoints', {}),
                        resource_limits=env_data.get('resource_limits', {}),
                        monitoring_config=env_data.get('monitoring_config', {}),
                        is_active=env_data.get('is_active', False)
                    )
                
                self.current_environment = config_dict.get('current_environment', 'development')
                
            except Exception as e:
                logger.error("Failed to load configuration", error=str(e))
                self._create_default_environments()
        else:
            self._create_default_environments()
    
    def _create_default_environments(self):
        """Create default environment configurations."""
        
        # Development Environment
        dev_keys = APIKeys()
        self.environments['development'] = EnvironmentConfig(
            name='development',
            description='Local development environment',
            api_keys=dev_keys,
            database_urls={
                'arangodb': 'http://localhost:8529',
                'postgres': 'postgresql://localhost:5432/orchestrator_dev',
                'redis': 'redis://localhost:6379/0'
            },
            service_endpoints={
                'orchestrator_api': 'http://localhost:8000',
                'live_dashboard': 'http://localhost:8001',
                'monitoring': 'http://localhost:3000'
            },
            resource_limits={
                'max_agents': 10,
                'max_concurrent_tasks': 20,
                'memory_limit_gb': 8
            },
            is_active=True
        )
        
        # Staging Environment
        staging_keys = APIKeys()
        self.environments['staging'] = EnvironmentConfig(
            name='staging',
            description='Pre-production staging environment',
            api_keys=staging_keys,
            database_urls={
                'arangodb': 'http://staging-arangodb:8529',
                'postgres': 'postgresql://staging-db:5432/orchestrator_staging',
                'redis': 'redis://staging-redis:6379/0'
            },
            service_endpoints={
                'orchestrator_api': 'http://staging-api:8000',
                'live_dashboard': 'http://staging-dashboard:8001',
                'monitoring': 'http://staging-grafana:3000'
            },
            resource_limits={
                'max_agents': 25,
                'max_concurrent_tasks': 50,
                'memory_limit_gb': 16
            },
            is_active=False
        )
        
        # Production Environment
        prod_keys = APIKeys()
        self.environments['production'] = EnvironmentConfig(
            name='production',
            description='Production environment with full scale',
            api_keys=prod_keys,
            database_urls={
                'arangodb': 'http://prod-arangodb-cluster:8529',
                'postgres': 'postgresql://prod-db-cluster:5432/orchestrator_prod',
                'redis': 'redis://prod-redis-cluster:6379/0'
            },
            service_endpoints={
                'orchestrator_api': 'http://prod-api-lb:8000',
                'live_dashboard': 'http://prod-dashboard-lb:8001',
                'monitoring': 'http://prod-grafana:3000'
            },
            resource_limits={
                'max_agents': 100,
                'max_concurrent_tasks': 200,
                'memory_limit_gb': 64
            },
            monitoring_config={
                'prometheus_scrape_interval': '15s',
                'log_level': 'INFO',
                'metrics_retention': '30d'
            },
            is_active=False
        )
        
        logger.info("Created default environments", 
                   environments=['development', 'staging', 'production'])
    
    def _load_state(self):
        """Load system state from file."""
        if self.state_file.exists():
            try:
                state_data = json.loads(self.state_file.read_text())
                self.system_state = SystemState(**state_data)
            except Exception as e:
                logger.error("Failed to load system state", error=str(e))
                self.system_state = SystemState()
    
    def save_configuration(self):
        """Save encrypted configuration to file."""
        config_dict = {
            'current_environment': self.current_environment,
            'environments': {}
        }
        
        for env_name, env_config in self.environments.items():
            config_dict['environments'][env_name] = {
                'description': env_config.description,
                'api_keys': asdict(env_config.api_keys),
                'database_urls': env_config.database_urls,
                'service_endpoints': env_config.service_endpoints,
                'resource_limits': env_config.resource_limits,
                'monitoring_config': env_config.monitoring_config,
                'is_active': env_config.is_active
            }
        
        config_json = json.dumps(config_dict, indent=2)
        encrypted_data = self._encrypt_data(config_json)
        self.config_file.write_bytes(encrypted_data)
        
        logger.info("Configuration saved securely")
    
    def save_state(self):
        """Save system state to file."""
        self.system_state.last_updated = datetime.utcnow().isoformat()
        state_json = json.dumps(asdict(self.system_state), indent=2)
        self.state_file.write_text(state_json)
        
        logger.info("System state saved")
    
    def get_current_config(self) -> EnvironmentConfig:
        """Get current environment configuration."""
        return self.environments.get(self.current_environment)
    
    def get_api_keys(self) -> APIKeys:
        """Get API keys for current environment."""
        current_config = self.get_current_config()
        return current_config.api_keys if current_config else APIKeys()
    
    def set_api_key(self, service: str, key: str, environment: str = None):
        """Set API key for a service."""
        env = environment or self.current_environment
        if env in self.environments:
            setattr(self.environments[env].api_keys, service, key)
            self.save_configuration()
            logger.info("API key updated", service=service, environment=env)
    
    def switch_environment(self, environment: str):
        """Switch to a different environment."""
        if environment in self.environments:
            self.current_environment = environment
            self.save_configuration()
            logger.info("Switched environment", new_environment=environment)
        else:
            raise ValueError(f"Environment '{environment}' not found")
    
    def add_environment(self, env_config: EnvironmentConfig):
        """Add a new environment configuration."""
        self.environments[env_config.name] = env_config
        self.save_configuration()
        logger.info("Environment added", environment=env_config.name)
    
    def update_system_state(self, **kwargs):
        """Update system state."""
        for key, value in kwargs.items():
            if hasattr(self.system_state, key):
                setattr(self.system_state, key, value)
        
        self.save_state()
        logger.info("System state updated", fields=list(kwargs.keys()))
    
    def add_active_agent(self, agent_id: str):
        """Add an active agent to the system state."""
        if agent_id not in self.system_state.active_agents:
            self.system_state.active_agents.append(agent_id)
            self.save_state()
    
    def remove_active_agent(self, agent_id: str):
        """Remove an active agent from the system state."""
        if agent_id in self.system_state.active_agents:
            self.system_state.active_agents.remove(agent_id)
            self.save_state()
    
    def add_completed_task(self, task_id: str):
        """Add a completed task to the system state."""
        if task_id not in self.system_state.completed_tasks:
            self.system_state.completed_tasks.append(task_id)
            self.save_state()
    
    def update_repository_analysis(self, repo_name: str, analysis: Dict[str, Any]):
        """Update repository analysis in system state."""
        self.system_state.repository_analysis[repo_name] = analysis
        self.save_state()
    
    def log_error(self, error: str, context: Dict[str, Any] = None):
        """Log an error to system state."""
        error_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'error': error,
            'context': context or {},
            'environment': self.current_environment
        }
        self.system_state.error_log.append(error_entry)
        
        # Keep only last 1000 errors
        if len(self.system_state.error_log) > 1000:
            self.system_state.error_log = self.system_state.error_log[-1000:]
        
        self.save_state()
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for current configuration."""
        current_config = self.get_current_config()
        if not current_config:
            return {}
        
        env_vars = {}
        api_keys = current_config.api_keys
        
        # API Keys
        if api_keys.openai_api_key:
            env_vars['OPENAI_API_KEY'] = api_keys.openai_api_key
        if api_keys.anthropic_api_key:
            env_vars['ANTHROPIC_API_KEY'] = api_keys.anthropic_api_key
        if api_keys.github_token:
            env_vars['GITHUB_TOKEN'] = api_keys.github_token
        
        # Database URLs
        for db_name, url in current_config.database_urls.items():
            env_vars[f'{db_name.upper()}_URL'] = url
        
        # Service endpoints
        for service_name, endpoint in current_config.service_endpoints.items():
            env_vars[f'{service_name.upper()}_ENDPOINT'] = endpoint
        
        return env_vars
    
    @contextmanager
    def environment_context(self, environment: str):
        """Context manager for temporarily switching environments."""
        original_env = self.current_environment
        try:
            self.switch_environment(environment)
            yield self.get_current_config()
        finally:
            self.switch_environment(original_env)
    
    def export_config_template(self, output_path: Path):
        """Export a configuration template for easy setup."""
        template = {
            'api_keys': asdict(APIKeys()),
            'environments': {
                'development': {
                    'description': 'Local development environment',
                    'database_urls': {
                        'arangodb': 'http://localhost:8529',
                        'postgres': 'postgresql://localhost:5432/orchestrator_dev'
                    },
                    'service_endpoints': {
                        'orchestrator_api': 'http://localhost:8000'
                    }
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        logger.info("Configuration template exported", path=str(output_path))
    
    def import_config_from_yaml(self, yaml_path: Path):
        """Import configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        for env_name, env_data in config_data.get('environments', {}).items():
            api_keys_data = {**asdict(APIKeys()), **env_data.get('api_keys', {})}
            api_keys = APIKeys(**api_keys_data)
            
            env_config = EnvironmentConfig(
                name=env_name,
                description=env_data.get('description', ''),
                api_keys=api_keys,
                database_urls=env_data.get('database_urls', {}),
                service_endpoints=env_data.get('service_endpoints', {}),
                resource_limits=env_data.get('resource_limits', {}),
                monitoring_config=env_data.get('monitoring_config', {})
            )
            
            self.add_environment(env_config)
        
        logger.info("Configuration imported from YAML", path=str(yaml_path))


# Global configuration manager instance
config_manager = SecureConfigManager()


def get_config_manager() -> SecureConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


def get_api_keys() -> APIKeys:
    """Get API keys for current environment."""
    return config_manager.get_api_keys()


def get_current_config() -> EnvironmentConfig:
    """Get current environment configuration."""
    return config_manager.get_current_config()


def setup_environment_variables():
    """Setup environment variables for current configuration."""
    env_vars = config_manager.get_environment_variables()
    for key, value in env_vars.items():
        os.environ[key] = value
    
    logger.info("Environment variables set", count=len(env_vars))


if __name__ == "__main__":
    # Example usage and testing
    
    print("🔐 Master Orchestrator - Unified Configuration System")
    print("=" * 60)
    
    # Initialize config manager
    cm = get_config_manager()
    
    print(f"📁 Config directory: {cm.config_dir}")
    print(f"🌍 Current environment: {cm.current_environment}")
    print(f"🔧 Available environments: {list(cm.environments.keys())}")
    
    # Show current configuration
    current_config = get_current_config()
    if current_config:
        print(f"📋 Current config: {current_config.description}")
        print(f"🔑 API keys configured: {sum(1 for v in asdict(current_config.api_keys).values() if v)}")
    
    # Export template for easy setup
    template_path = Path("config_template.yaml")
    cm.export_config_template(template_path)
    print(f"📄 Configuration template exported to: {template_path}")
    
    print("\n✨ Configuration system ready!")
    print("📝 Edit the template file and import to set up your API keys")