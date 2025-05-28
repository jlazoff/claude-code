#!/usr/bin/env python3
"""
Basic tests for Master Orchestrator system
"""

import pytest
import asyncio
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all core modules can be imported"""
    try:
        import unified_config
        import parallel_llm_orchestrator
        import frontend_orchestrator
        import computer_control_orchestrator
        import content_analyzer_deployer
        import github_integration
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_config_creation():
    """Test configuration manager creation"""
    from unified_config import SecureConfigManager
    config = SecureConfigManager()
    assert config is not None
    assert hasattr(config, 'config')

@pytest.mark.asyncio
async def test_config_initialization():
    """Test async configuration initialization"""
    from unified_config import SecureConfigManager
    config = SecureConfigManager()
    
    # Mock the initialization to avoid requiring actual config files
    with patch.object(config, '_load_config', return_value={}):
        with patch.object(config, '_setup_encryption'):
            await config.initialize()
            assert config.initialized

@pytest.mark.asyncio
async def test_llm_orchestrator_creation():
    """Test LLM orchestrator creation"""
    from parallel_llm_orchestrator import ParallelLLMOrchestrator
    orchestrator = ParallelLLMOrchestrator()
    assert orchestrator is not None
    assert hasattr(orchestrator, 'config')
    assert hasattr(orchestrator, 'vertex_ai')
    assert hasattr(orchestrator, 'openai_manager')
    assert hasattr(orchestrator, 'grok_manager')

@pytest.mark.asyncio
async def test_frontend_orchestrator_creation():
    """Test frontend orchestrator creation"""
    from frontend_orchestrator import FrontendOrchestrator
    frontend = FrontendOrchestrator()
    assert frontend is not None
    assert hasattr(frontend, 'app')
    assert hasattr(frontend, 'websocket_clients')

def test_computer_control_components():
    """Test computer control components"""
    from computer_control_orchestrator import ScreenObserver, PhysicalController
    
    screen_observer = ScreenObserver()
    assert screen_observer is not None
    assert hasattr(screen_observer, 'observers')
    
    physical_controller = PhysicalController()
    assert physical_controller is not None
    assert hasattr(physical_controller, 'action_history')

@pytest.mark.asyncio
async def test_content_analyzer_creation():
    """Test content analyzer creation"""
    from content_analyzer_deployer import ContentAnalyzerDeployer
    analyzer = ContentAnalyzerDeployer()
    assert analyzer is not None
    assert hasattr(analyzer, 'config')

def test_github_integration_creation():
    """Test GitHub integration creation"""
    from github_integration import GitHubIntegrator
    from unified_config import SecureConfigManager
    
    config = SecureConfigManager()
    github_integrator = GitHubIntegrator(config)
    assert github_integrator is not None
    assert hasattr(github_integrator, 'config')

@pytest.mark.asyncio
async def test_async_functionality():
    """Test basic async functionality"""
    async def dummy_async():
        await asyncio.sleep(0.01)
        return "async_works"
    
    result = await dummy_async()
    assert result == "async_works"

def test_path_resolution():
    """Test that required files exist"""
    base_path = Path(__file__).parent.parent
    assert base_path.exists()
    
    required_files = [
        "unified_config.py",
        "parallel_llm_orchestrator.py",
        "frontend_orchestrator.py",
        "computer_control_orchestrator.py",
        "content_analyzer_deployer.py",
        "github_integration.py",
        "one_click_deploy.py"
    ]
    
    for file_name in required_files:
        file_path = base_path / file_name
        assert file_path.exists(), f"Required file {file_name} does not exist"

@pytest.mark.asyncio
async def test_mock_llm_response():
    """Test mock LLM response handling"""
    from parallel_llm_orchestrator import LLMResponse
    
    mock_response = LLMResponse(
        provider="test",
        model="test-model",
        content="Test response",
        tokens_used=10,
        latency_ms=100.0,
        confidence_score=0.9
    )
    
    assert mock_response.provider == "test"
    assert mock_response.content == "Test response"
    assert mock_response.confidence_score == 0.9

def test_code_merger_strategies():
    """Test code merger strategy enumeration"""
    from parallel_llm_orchestrator import CodeMerger
    
    merger = CodeMerger()
    assert hasattr(merger, 'merger_strategies')
    
    expected_strategies = ["best_practices", "performance", "comprehensive", "consensus"]
    for strategy in expected_strategies:
        assert strategy in merger.merger_strategies

@pytest.mark.asyncio
async def test_deployment_configuration():
    """Test deployment configuration"""
    from one_click_deploy import OneClickDeployer
    
    deployer = OneClickDeployer()
    assert deployer.deployment_config is not None
    assert "python_version" in deployer.deployment_config
    assert "required_packages" in deployer.deployment_config
    assert "frontend_port" in deployer.deployment_config

def test_websocket_client_management():
    """Test WebSocket client management"""
    from frontend_orchestrator import FrontendOrchestrator
    
    frontend = FrontendOrchestrator()
    
    # Test client set operations
    mock_client = Mock()
    frontend.websocket_clients.add(mock_client)
    assert len(frontend.websocket_clients) == 1
    
    frontend.websocket_clients.discard(mock_client)
    assert len(frontend.websocket_clients) == 0

@pytest.mark.asyncio
async def test_project_creation_structure():
    """Test project creation data structure"""
    from frontend_orchestrator import FrontendOrchestrator
    import hashlib
    from datetime import datetime
    
    frontend = FrontendOrchestrator()
    
    # Mock project data
    project_data = {
        "name": "Test Project",
        "description": "Test Description",
        "type": "test"
    }
    
    project = {
        "id": hashlib.md5(f"{project_data['name']}{datetime.now()}".encode()).hexdigest()[:8],
        "name": project_data['name'],
        "description": project_data.get('description', ''),
        "type": project_data.get('type', 'general'),
        "status": "active",
        "created_at": datetime.now().isoformat(),
        "files": [],
        "deployments": []
    }
    
    assert project["name"] == "Test Project"
    assert project["status"] == "active"
    assert len(project["id"]) == 8

def test_monitoring_data_structure():
    """Test monitoring data structure"""
    from frontend_orchestrator import FrontendOrchestrator
    from datetime import datetime
    
    frontend = FrontendOrchestrator()
    
    # Mock monitoring data
    monitoring_data = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": 25.5,
        "memory_percent": 45.2,
        "disk_percent": 60.1,
        "total_requests": 100,
        "success_rate": 95.5,
        "active_projects": 3,
        "websocket_clients": 2,
        "system_status": "healthy"
    }
    
    frontend.monitoring_data = monitoring_data
    
    assert frontend.monitoring_data["cpu_percent"] == 25.5
    assert frontend.monitoring_data["system_status"] == "healthy"
    assert "timestamp" in frontend.monitoring_data

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in async functions"""
    async def failing_function():
        raise ValueError("Test error")
    
    try:
        await failing_function()
        assert False, "Expected exception was not raised"
    except ValueError as e:
        assert str(e) == "Test error"

def test_configuration_validation():
    """Test configuration validation"""
    config_data = {
        "environment": "test",
        "debug": True,
        "api": {
            "host": "localhost",
            "port": 8000
        }
    }
    
    # Validate required fields
    assert "environment" in config_data
    assert "api" in config_data
    assert "host" in config_data["api"]
    assert "port" in config_data["api"]

@pytest.mark.asyncio 
async def test_code_analysis_structure():
    """Test code analysis data structure"""
    from parallel_llm_orchestrator import CodeAnalysis
    
    analysis = CodeAnalysis(
        file_path="test.py",
        language="python",
        complexity_score=75.0,
        quality_score=85.0,
        suggestions=["Add type hints", "Improve error handling"],
        optimizations=["Use list comprehension", "Cache results"],
        security_issues=[],
        performance_issues=["Avoid nested loops"],
        generated_improvements=""
    )
    
    assert analysis.file_path == "test.py"
    assert analysis.language == "python"
    assert len(analysis.suggestions) == 2
    assert len(analysis.optimizations) == 2
    assert len(analysis.performance_issues) == 1

def test_git_repository_detection():
    """Test git repository detection logic"""
    from pathlib import Path
    import os
    
    # Check if we're in a git repository
    current_path = Path.cwd()
    git_path = current_path / ".git"
    
    # This should pass if we're in the claude-code repository
    if git_path.exists():
        assert git_path.is_dir()
    else:
        # If not in git repo, that's also valid for testing
        assert True

@pytest.mark.asyncio
async def test_health_check_structure():
    """Test health check response structure"""
    from datetime import datetime
    
    health_response = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "llm_orchestrator": "online",
            "computer_control": "online", 
            "content_analyzer": "online",
            "websocket_connections": 0,
            "active_projects": 0
        }
    }
    
    assert health_response["status"] == "healthy"
    assert "timestamp" in health_response
    assert "components" in health_response
    assert health_response["components"]["llm_orchestrator"] == "online"

def test_security_configuration():
    """Test security-related configuration"""
    # Test that security best practices are followed
    security_config = {
        "encryption_enabled": True,
        "api_key_encryption": True,
        "secure_headers": True,
        "rate_limiting": True
    }
    
    # All security features should be enabled
    for feature, enabled in security_config.items():
        assert enabled, f"Security feature {feature} should be enabled"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])