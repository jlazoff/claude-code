"""Configuration management for Master Orchestrator."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
import yaml


class ArangoDBConfig(BaseModel):
    """ArangoDB configuration."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=8529)
    username: str = Field(default="root")
    password: str = Field(default="")
    database: str = Field(default="master_orchestrator")
    protocol: str = Field(default="http")


class LLMProviderConfig(BaseModel):
    """LLM provider configuration."""
    
    name: str = Field(description="Provider name")
    api_key: str = Field(description="API key")
    base_url: Optional[str] = Field(default=None, description="Custom base URL")
    model: str = Field(description="Default model")
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    enabled: bool = Field(default=True)


class AgentConfig(BaseModel):
    """Agent framework configuration."""
    
    max_concurrent_agents: int = Field(default=10)
    default_timeout: int = Field(default=300)
    retry_attempts: int = Field(default=3)
    llm_providers: List[LLMProviderConfig] = Field(default_factory=list)
    dspy_cache_dir: Path = Field(default=Path("~/.cache/dspy").expanduser())


class InfrastructureConfig(BaseModel):
    """Infrastructure configuration."""
    
    kubernetes_config_path: Optional[Path] = Field(default=None)
    docker_host: str = Field(default="unix:///var/run/docker.sock")
    terraform_dir: Path = Field(default=Path("./infrastructure/terraform"))
    ansible_dir: Path = Field(default=Path("./infrastructure/ansible"))
    
    # Hardware nodes
    mac_studios: List[str] = Field(default_factory=list)
    mac_minis: List[str] = Field(default_factory=list)
    nas_systems: List[str] = Field(default_factory=list)


class RepositoryConfig(BaseModel):
    """Repository management configuration."""
    
    github_base_path: Path = Field(description="Base path to GitHub repositories")
    auto_sync: bool = Field(default=True)
    sync_interval: int = Field(default=3600)  # seconds
    analysis_depth: str = Field(default="medium")  # shallow, medium, deep


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    prometheus_port: int = Field(default=9090)
    grafana_port: int = Field(default=3000)
    log_level: str = Field(default="INFO")
    metrics_enabled: bool = Field(default=True)
    tracing_enabled: bool = Field(default=True)


class OrchestratorConfig(BaseModel):
    """Main orchestrator configuration."""
    
    # Core settings
    name: str = Field(default="master-orchestrator")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    
    # Component configurations
    arangodb_config: ArangoDBConfig = Field(default_factory=ArangoDBConfig)
    agent_config: AgentConfig = Field(default_factory=AgentConfig)
    infrastructure_config: InfrastructureConfig = Field(default_factory=InfrastructureConfig)
    repository_config: RepositoryConfig = Field(default_factory=lambda: RepositoryConfig(
        github_base_path=Path("/Users/jlazoff/Documents/GitHub")
    ))
    monitoring_config: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # API settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    
    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production")
    allowed_hosts: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'OrchestratorConfig':
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    @classmethod
    def from_env(cls) -> 'OrchestratorConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        if os.getenv("ORCHESTRATOR_NAME"):
            config.name = os.getenv("ORCHESTRATOR_NAME")
        
        if os.getenv("ORCHESTRATOR_ENVIRONMENT"):
            config.environment = os.getenv("ORCHESTRATOR_ENVIRONMENT")
        
        # ArangoDB config from env
        if os.getenv("ARANGODB_HOST"):
            config.arangodb_config.host = os.getenv("ARANGODB_HOST")
        if os.getenv("ARANGODB_PORT"):
            config.arangodb_config.port = int(os.getenv("ARANGODB_PORT"))
        if os.getenv("ARANGODB_USERNAME"):
            config.arangodb_config.username = os.getenv("ARANGODB_USERNAME")
        if os.getenv("ARANGODB_PASSWORD"):
            config.arangodb_config.password = os.getenv("ARANGODB_PASSWORD")
        if os.getenv("ARANGODB_DATABASE"):
            config.arangodb_config.database = os.getenv("ARANGODB_DATABASE")
        
        # LLM provider configs from env
        llm_providers = []
        
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            llm_providers.append(LLMProviderConfig(
                name="openai",
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
            ))
        
        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            llm_providers.append(LLMProviderConfig(
                name="anthropic",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20241022"),
            ))
        
        # Google
        if os.getenv("GOOGLE_API_KEY"):
            llm_providers.append(LLMProviderConfig(
                name="google",
                api_key=os.getenv("GOOGLE_API_KEY"),
                model=os.getenv("GOOGLE_MODEL", "gemini-pro"),
            ))
        
        config.agent_config.llm_providers = llm_providers
        
        # API config from env
        if os.getenv("API_HOST"):
            config.api_host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            config.api_port = int(os.getenv("API_PORT"))
        
        # Security
        if os.getenv("SECRET_KEY"):
            config.secret_key = os.getenv("SECRET_KEY")
        
        return config
    
    def to_file(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(
                self.model_dump(exclude={"secret_key"}),  # Don't save secret key
                f,
                default_flow_style=False,
                indent=2
            )
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration summary."""
        return {
            "mac_studios": len(self.infrastructure_config.mac_studios),
            "mac_minis": len(self.infrastructure_config.mac_minis),
            "nas_systems": len(self.infrastructure_config.nas_systems),
            "total_nodes": (
                len(self.infrastructure_config.mac_studios) +
                len(self.infrastructure_config.mac_minis) +
                len(self.infrastructure_config.nas_systems)
            )
        }
    
    def get_llm_providers(self) -> List[str]:
        """Get list of enabled LLM providers."""
        return [
            provider.name 
            for provider in self.agent_config.llm_providers 
            if provider.enabled
        ]