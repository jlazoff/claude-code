#!/usr/bin/env python3
"""
Deployment and Testing Strategy Generator
Generates deployment configurations and testing strategies for tools in the catalog
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import subprocess
import logging

logger = logging.getLogger(__name__)

@dataclass
class DeploymentStrategy:
    """Deployment strategy for a tool"""
    tool_name: str
    tool_path: str
    deployment_type: str  # docker, kubernetes, systemd, lambda, etc.
    configuration: Dict[str, Any]
    dependencies: List[str]
    environment_variables: Dict[str, str]
    resource_requirements: Dict[str, Any]
    scaling_config: Dict[str, Any]
    health_checks: List[Dict[str, Any]]
    
@dataclass
class TestingStrategy:
    """Testing strategy for a tool"""
    tool_name: str
    tool_path: str
    test_types: List[str]  # unit, integration, e2e, performance, security
    test_commands: Dict[str, str]
    test_dependencies: List[str]
    test_environment: Dict[str, str]
    coverage_threshold: float
    performance_benchmarks: Dict[str, Any]

class DeploymentTestingGenerator:
    """Generates deployment and testing strategies for tools"""
    
    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        self.catalog = self._load_catalog()
        
    def _load_catalog(self) -> Dict:
        """Load tool catalog"""
        with open(self.catalog_path, 'r') as f:
            return json.load(f)
            
    def generate_deployment_strategy(self, tool_path: str, tool_data: Dict) -> DeploymentStrategy:
        """Generate deployment strategy for a tool"""
        metadata = tool_data['metadata']
        
        # Determine deployment type
        deployment_type = self._determine_deployment_type(tool_data)
        
        # Generate configuration based on type
        if deployment_type == "docker":
            config = self._generate_docker_config(tool_data)
        elif deployment_type == "kubernetes":
            config = self._generate_kubernetes_config(tool_data)
        elif deployment_type == "systemd":
            config = self._generate_systemd_config(tool_data)
        elif deployment_type == "lambda":
            config = self._generate_lambda_config(tool_data)
        else:
            config = {}
            
        # Extract environment variables
        env_vars = self._extract_environment_variables(tool_data)
        
        # Determine resource requirements
        resources = self._determine_resource_requirements(tool_data)
        
        # Generate scaling configuration
        scaling = self._generate_scaling_config(tool_data)
        
        # Define health checks
        health_checks = self._define_health_checks(tool_data)
        
        return DeploymentStrategy(
            tool_name=metadata['name'],
            tool_path=tool_path,
            deployment_type=deployment_type,
            configuration=config,
            dependencies=metadata.get('dependencies', []),
            environment_variables=env_vars,
            resource_requirements=resources,
            scaling_config=scaling,
            health_checks=health_checks
        )
        
    def _determine_deployment_type(self, tool_data: Dict) -> str:
        """Determine appropriate deployment type"""
        metadata = tool_data['metadata']
        
        if metadata.get('has_dockerfile'):
            if tool_data.get('category') == 'Infrastructure':
                return "kubernetes"
            else:
                return "docker"
        elif metadata.get('primary_language') == 'Python':
            if any('lambda' in dep for dep in metadata.get('dependencies', [])):
                return "lambda"
            else:
                return "systemd"
        else:
            return "traditional"
            
    def _generate_docker_config(self, tool_data: Dict) -> Dict:
        """Generate Docker deployment configuration"""
        metadata = tool_data['metadata']
        
        config = {
            "compose": {
                "version": "3.8",
                "services": {
                    metadata['name']: {
                        "build": ".",
                        "image": f"{metadata['name']}:latest",
                        "restart": "unless-stopped",
                        "environment": {},
                        "volumes": [],
                        "ports": [],
                        "networks": ["app-network"]
                    }
                },
                "networks": {
                    "app-network": {
                        "driver": "bridge"
                    }
                }
            }
        }
        
        # Add ports if web service
        if tool_data.get('category') == 'Web Services':
            config['compose']['services'][metadata['name']]['ports'] = ["8000:8000"]
            
        return config
        
    def _generate_kubernetes_config(self, tool_data: Dict) -> Dict:
        """Generate Kubernetes deployment configuration"""
        metadata = tool_data['metadata']
        
        config = {
            "deployment": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": metadata['name'],
                    "labels": {
                        "app": metadata['name']
                    }
                },
                "spec": {
                    "replicas": 3,
                    "selector": {
                        "matchLabels": {
                            "app": metadata['name']
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": metadata['name']
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": metadata['name'],
                                "image": f"{metadata['name']}:latest",
                                "ports": [],
                                "resources": {
                                    "requests": {
                                        "memory": "256Mi",
                                        "cpu": "250m"
                                    },
                                    "limits": {
                                        "memory": "512Mi",
                                        "cpu": "500m"
                                    }
                                }
                            }]
                        }
                    }
                }
            },
            "service": {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": metadata['name']
                },
                "spec": {
                    "selector": {
                        "app": metadata['name']
                    },
                    "ports": [],
                    "type": "ClusterIP"
                }
            }
        }
        
        return config
        
    def _generate_systemd_config(self, tool_data: Dict) -> Dict:
        """Generate systemd service configuration"""
        metadata = tool_data['metadata']
        
        config = {
            "unit": {
                "Description": f"{metadata['name']} service",
                "After": "network.target"
            },
            "service": {
                "Type": "simple",
                "User": "app",
                "WorkingDirectory": f"/opt/{metadata['name']}",
                "ExecStart": f"/usr/bin/python3 main.py",
                "Restart": "on-failure",
                "RestartSec": "10s"
            },
            "install": {
                "WantedBy": "multi-user.target"
            }
        }
        
        return config
        
    def _generate_lambda_config(self, tool_data: Dict) -> Dict:
        """Generate AWS Lambda configuration"""
        metadata = tool_data['metadata']
        
        config = {
            "function": {
                "FunctionName": metadata['name'],
                "Runtime": "python3.9",
                "Handler": "lambda_function.lambda_handler",
                "Timeout": 300,
                "MemorySize": 512,
                "Environment": {
                    "Variables": {}
                }
            },
            "serverless": {
                "service": metadata['name'],
                "provider": {
                    "name": "aws",
                    "runtime": "python3.9",
                    "stage": "prod",
                    "region": "us-east-1"
                },
                "functions": {
                    metadata['name']: {
                        "handler": "handler.main",
                        "events": []
                    }
                }
            }
        }
        
        return config
        
    def _extract_environment_variables(self, tool_data: Dict) -> Dict[str, str]:
        """Extract required environment variables"""
        env_vars = {}
        
        # Common environment variables
        if tool_data.get('category') == 'AI/ML':
            env_vars['OPENAI_API_KEY'] = '${OPENAI_API_KEY}'
            env_vars['MODEL_PATH'] = '/models'
            
        if 'database' in str(tool_data.get('integration_points', [])):
            env_vars['DATABASE_URL'] = '${DATABASE_URL}'
            
        if tool_data.get('category') == 'Web Services':
            env_vars['PORT'] = '8000'
            env_vars['HOST'] = '0.0.0.0'
            
        return env_vars
        
    def _determine_resource_requirements(self, tool_data: Dict) -> Dict[str, Any]:
        """Determine resource requirements"""
        metadata = tool_data['metadata']
        size_mb = metadata.get('size_bytes', 0) / (1024 * 1024)
        
        # Base requirements
        resources = {
            "cpu": "0.5",
            "memory": "512Mi",
            "storage": "1Gi"
        }
        
        # Adjust based on category and size
        if tool_data.get('category') == 'AI/ML':
            resources['cpu'] = "2"
            resources['memory'] = "4Gi"
            resources['gpu'] = "optional"
            
        elif tool_data.get('category') == 'Data Processing':
            resources['cpu'] = "1"
            resources['memory'] = "2Gi"
            resources['storage'] = "10Gi"
            
        elif size_mb > 500:
            resources['memory'] = "1Gi"
            
        return resources
        
    def _generate_scaling_config(self, tool_data: Dict) -> Dict[str, Any]:
        """Generate scaling configuration"""
        scaling = {
            "min_replicas": 1,
            "max_replicas": 10,
            "target_cpu_utilization": 70,
            "scale_up_rate": 1,
            "scale_down_rate": 1
        }
        
        # Adjust based on category
        if tool_data.get('category') == 'Web Services':
            scaling['min_replicas'] = 2
            scaling['max_replicas'] = 20
            
        elif tool_data.get('category') == 'Data Processing':
            scaling['min_replicas'] = 1
            scaling['max_replicas'] = 5
            
        return scaling
        
    def _define_health_checks(self, tool_data: Dict) -> List[Dict[str, Any]]:
        """Define health check configurations"""
        health_checks = []
        
        # HTTP health check for web services
        if tool_data.get('category') == 'Web Services':
            health_checks.append({
                "type": "http",
                "path": "/health",
                "port": 8000,
                "interval": "30s",
                "timeout": "10s",
                "success_threshold": 1,
                "failure_threshold": 3
            })
            
        # TCP health check for other services
        else:
            health_checks.append({
                "type": "tcp",
                "port": 8000,
                "interval": "30s",
                "timeout": "10s",
                "success_threshold": 1,
                "failure_threshold": 3
            })
            
        # Process health check
        health_checks.append({
            "type": "exec",
            "command": ["ps", "aux", "|", "grep", tool_data['metadata']['name']],
            "interval": "60s",
            "timeout": "5s"
        })
        
        return health_checks
        
    def generate_testing_strategy(self, tool_path: str, tool_data: Dict) -> TestingStrategy:
        """Generate testing strategy for a tool"""
        metadata = tool_data['metadata']
        
        # Determine test types needed
        test_types = self._determine_test_types(tool_data)
        
        # Generate test commands
        test_commands = self._generate_test_commands(tool_data, test_types)
        
        # Determine test dependencies
        test_deps = self._determine_test_dependencies(tool_data)
        
        # Set up test environment
        test_env = self._setup_test_environment(tool_data)
        
        # Set coverage threshold
        coverage = 80.0 if metadata.get('has_tests') else 60.0
        
        # Define performance benchmarks
        benchmarks = self._define_performance_benchmarks(tool_data)
        
        return TestingStrategy(
            tool_name=metadata['name'],
            tool_path=tool_path,
            test_types=test_types,
            test_commands=test_commands,
            test_dependencies=test_deps,
            test_environment=test_env,
            coverage_threshold=coverage,
            performance_benchmarks=benchmarks
        )
        
    def _determine_test_types(self, tool_data: Dict) -> List[str]:
        """Determine which types of tests are needed"""
        test_types = ['unit']  # Always include unit tests
        
        if tool_data.get('category') == 'Web Services':
            test_types.extend(['integration', 'e2e'])
            
        if tool_data.get('category') == 'AI/ML':
            test_types.append('model_validation')
            
        if 'api' in str(tool_data.get('integration_points', [])):
            test_types.append('api')
            
        if tool_data.get('deployment_type') == 'kubernetes':
            test_types.append('chaos')
            
        test_types.extend(['performance', 'security'])
        
        return list(set(test_types))
        
    def _generate_test_commands(self, tool_data: Dict, test_types: List[str]) -> Dict[str, str]:
        """Generate test commands for each test type"""
        commands = {}
        metadata = tool_data['metadata']
        
        if metadata.get('primary_language') == 'Python':
            commands['unit'] = 'pytest tests/unit -v --cov=src --cov-report=html'
            commands['integration'] = 'pytest tests/integration -v'
            commands['e2e'] = 'pytest tests/e2e -v --browser=chrome'
            commands['performance'] = 'locust -f tests/performance/locustfile.py'
            commands['security'] = 'bandit -r src/ && safety check'
            
        elif metadata.get('primary_language') == 'JavaScript':
            commands['unit'] = 'npm test'
            commands['integration'] = 'npm run test:integration'
            commands['e2e'] = 'npm run test:e2e'
            commands['performance'] = 'npm run test:performance'
            commands['security'] = 'npm audit'
            
        if 'model_validation' in test_types:
            commands['model_validation'] = 'python tests/validate_model.py'
            
        if 'api' in test_types:
            commands['api'] = 'pytest tests/api -v'
            
        if 'chaos' in test_types:
            commands['chaos'] = 'chaos run experiments/network-delay.yaml'
            
        return commands
        
    def _determine_test_dependencies(self, tool_data: Dict) -> List[str]:
        """Determine testing dependencies"""
        deps = []
        metadata = tool_data['metadata']
        
        if metadata.get('primary_language') == 'Python':
            deps.extend(['pytest', 'pytest-cov', 'pytest-asyncio', 'pytest-mock'])
            
            if 'e2e' in self._determine_test_types(tool_data):
                deps.extend(['selenium', 'pytest-selenium'])
                
            if 'performance' in self._determine_test_types(tool_data):
                deps.append('locust')
                
            if 'security' in self._determine_test_types(tool_data):
                deps.extend(['bandit', 'safety'])
                
        elif metadata.get('primary_language') == 'JavaScript':
            deps.extend(['jest', 'mocha', 'chai', 'sinon'])
            
            if 'e2e' in self._determine_test_types(tool_data):
                deps.extend(['cypress', 'puppeteer'])
                
        return deps
        
    def _setup_test_environment(self, tool_data: Dict) -> Dict[str, str]:
        """Set up test environment variables"""
        env = {
            'ENVIRONMENT': 'test',
            'LOG_LEVEL': 'DEBUG',
            'DATABASE_URL': 'sqlite:///test.db'
        }
        
        if tool_data.get('category') == 'AI/ML':
            env['MODEL_PATH'] = './tests/fixtures/test_model'
            env['OPENAI_API_KEY'] = 'test-key'
            
        if tool_data.get('category') == 'Web Services':
            env['API_URL'] = 'http://localhost:8001'
            env['TEST_USER'] = 'test@example.com'
            
        return env
        
    def _define_performance_benchmarks(self, tool_data: Dict) -> Dict[str, Any]:
        """Define performance benchmarks"""
        benchmarks = {
            'response_time_p50': 100,  # ms
            'response_time_p95': 500,  # ms
            'response_time_p99': 1000,  # ms
            'throughput': 100,  # requests/second
            'error_rate': 0.01,  # 1%
            'cpu_usage': 70,  # percentage
            'memory_usage': 80  # percentage
        }
        
        # Adjust based on category
        if tool_data.get('category') == 'AI/ML':
            benchmarks['inference_time'] = 1000  # ms
            benchmarks['model_accuracy'] = 0.95
            
        elif tool_data.get('category') == 'Data Processing':
            benchmarks['processing_rate'] = 1000  # records/second
            
        return benchmarks
        
    def generate_deployment_files(self, deployment: DeploymentStrategy, output_dir: str):
        """Generate deployment configuration files"""
        output_path = Path(output_dir) / deployment.tool_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        if deployment.deployment_type == "docker":
            # Generate docker-compose.yml
            compose_path = output_path / "docker-compose.yml"
            with open(compose_path, 'w') as f:
                yaml.dump(deployment.configuration['compose'], f)
                
            # Generate .env template
            env_path = output_path / ".env.template"
            with open(env_path, 'w') as f:
                for key, value in deployment.environment_variables.items():
                    f.write(f"{key}={value}\n")
                    
        elif deployment.deployment_type == "kubernetes":
            # Generate deployment.yaml
            deployment_path = output_path / "deployment.yaml"
            with open(deployment_path, 'w') as f:
                yaml.dump(deployment.configuration['deployment'], f)
                f.write("---\n")
                yaml.dump(deployment.configuration['service'], f)
                
            # Generate configmap.yaml
            configmap = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": f"{deployment.tool_name}-config"
                },
                "data": deployment.environment_variables
            }
            configmap_path = output_path / "configmap.yaml"
            with open(configmap_path, 'w') as f:
                yaml.dump(configmap, f)
                
        elif deployment.deployment_type == "systemd":
            # Generate systemd service file
            service_path = output_path / f"{deployment.tool_name}.service"
            with open(service_path, 'w') as f:
                for section, values in deployment.configuration.items():
                    f.write(f"[{section.capitalize()}]\n")
                    for key, value in values.items():
                        f.write(f"{key}={value}\n")
                    f.write("\n")
                    
        # Generate deployment script
        deploy_script = self._generate_deployment_script(deployment)
        script_path = output_path / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(deploy_script)
        os.chmod(script_path, 0o755)
        
    def _generate_deployment_script(self, deployment: DeploymentStrategy) -> str:
        """Generate deployment automation script"""
        script = f"""#!/bin/bash
# Deployment script for {deployment.tool_name}
# Generated by Deployment Testing Strategy Generator

set -e

echo "Deploying {deployment.tool_name}..."

# Check prerequisites
"""
        
        if deployment.deployment_type == "docker":
            script += """
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed"
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "Please create .env file from .env.template"
    exit 1
fi

# Build and deploy
docker-compose build
docker-compose up -d

# Wait for service to be ready
echo "Waiting for service to be ready..."
sleep 10

# Health check
if curl -f http://localhost:8000/health; then
    echo "Service is healthy"
else
    echo "Service health check failed"
    docker-compose logs
    exit 1
fi
"""
        
        elif deployment.deployment_type == "kubernetes":
            script += """
if ! command -v kubectl &> /dev/null; then
    echo "kubectl is not installed"
    exit 1
fi

# Check cluster connection
if ! kubectl cluster-info &> /dev/null; then
    echo "Not connected to Kubernetes cluster"
    exit 1
fi

# Apply configurations
kubectl apply -f configmap.yaml
kubectl apply -f deployment.yaml

# Wait for deployment
kubectl rollout status deployment/{deployment.tool_name}

# Check service
kubectl get service {deployment.tool_name}
"""
        
        script += """
echo "Deployment complete!"
"""
        
        return script
        
    def generate_test_files(self, testing: TestingStrategy, output_dir: str):
        """Generate testing configuration files"""
        output_path = Path(output_dir) / testing.tool_name / "tests"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate test structure
        for test_type in testing.test_types:
            (output_path / test_type).mkdir(exist_ok=True)
            
            # Generate sample test file
            test_file = output_path / test_type / f"test_{test_type}.py"
            with open(test_file, 'w') as f:
                f.write(f"""\"\"\"
{test_type.capitalize()} tests for {testing.tool_name}
\"\"\"

import pytest

class Test{test_type.capitalize()}:
    def test_example(self):
        \"\"\"Example {test_type} test\"\"\"
        # TODO: Implement {test_type} tests
        assert True
""")
        
        # Generate test configuration
        if testing.tool_name.endswith('.py'):
            # pytest.ini
            pytest_ini = output_path.parent / "pytest.ini"
            with open(pytest_ini, 'w') as f:
                f.write(f"""[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
""")
        
        # Generate test runner script
        test_script = self._generate_test_script(testing)
        script_path = output_path.parent / "run_tests.sh"
        with open(script_path, 'w') as f:
            f.write(test_script)
        os.chmod(script_path, 0o755)
        
        # Generate CI/CD configuration
        ci_config = self._generate_ci_config(testing)
        ci_path = output_path.parent / ".github" / "workflows" / "test.yml"
        ci_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ci_path, 'w') as f:
            yaml.dump(ci_config, f)
            
    def _generate_test_script(self, testing: TestingStrategy) -> str:
        """Generate test automation script"""
        script = f"""#!/bin/bash
# Test runner script for {testing.tool_name}
# Generated by Deployment Testing Strategy Generator

set -e

echo "Running tests for {testing.tool_name}..."

# Set test environment
"""
        
        for key, value in testing.test_environment.items():
            script += f"export {key}={value}\n"
            
        script += "\n# Install test dependencies\n"
        if testing.tool_name.endswith('.py'):
            script += "pip install " + " ".join(testing.test_dependencies) + "\n\n"
            
        script += "# Run tests\n"
        for test_type, command in testing.test_commands.items():
            script += f"""
echo "Running {test_type} tests..."
{command} || echo "{test_type} tests failed"
"""
        
        script += f"""
# Check coverage
if [ -f htmlcov/index.html ]; then
    coverage=$(grep -oP 'Total coverage: \K[0-9]+' htmlcov/index.html || echo "0")
    if [ "$coverage" -lt "{int(testing.coverage_threshold)}" ]; then
        echo "Coverage $coverage% is below threshold {testing.coverage_threshold}%"
        exit 1
    fi
fi

echo "All tests completed!"
"""
        
        return script
        
    def _generate_ci_config(self, testing: TestingStrategy) -> Dict:
        """Generate CI/CD configuration"""
        config = {
            "name": f"Test {testing.tool_name}",
            "on": {
                "push": {
                    "branches": ["main", "develop"]
                },
                "pull_request": {
                    "branches": ["main"]
                }
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "strategy": {
                        "matrix": {
                            "python-version": ["3.8", "3.9", "3.10"]
                        }
                    },
                    "steps": [
                        {
                            "uses": "actions/checkout@v3"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {
                                "python-version": "${{ matrix.python-version }}"
                            }
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run tests",
                            "run": "./run_tests.sh"
                        },
                        {
                            "name": "Upload coverage",
                            "uses": "codecov/codecov-action@v3"
                        }
                    ]
                }
            }
        }
        
        return config
        
    def generate_all_strategies(self, output_dir: str):
        """Generate deployment and testing strategies for all tools"""
        deployment_dir = os.path.join(output_dir, "deployments")
        testing_dir = os.path.join(output_dir, "testing")
        
        os.makedirs(deployment_dir, exist_ok=True)
        os.makedirs(testing_dir, exist_ok=True)
        
        for tool_path, tool_data in self.catalog['tools'].items():
            logger.info(f"Generating strategies for {tool_data['metadata']['name']}")
            
            # Generate deployment strategy
            deployment = self.generate_deployment_strategy(tool_path, tool_data)
            self.generate_deployment_files(deployment, deployment_dir)
            
            # Generate testing strategy
            testing = self.generate_testing_strategy(tool_path, tool_data)
            self.generate_test_files(testing, testing_dir)
            
        # Generate master deployment orchestration
        self._generate_master_orchestration(deployment_dir)
        
        logger.info(f"Generated strategies for {len(self.catalog['tools'])} tools")
        
    def _generate_master_orchestration(self, deployment_dir: str):
        """Generate master orchestration configuration"""
        master_compose = {
            "version": "3.8",
            "services": {},
            "networks": {
                "master-network": {
                    "driver": "bridge"
                }
            }
        }
        
        # Add all services
        for tool_path, tool_data in self.catalog['tools'].items():
            name = tool_data['metadata']['name']
            if tool_data['metadata'].get('has_dockerfile'):
                master_compose['services'][name] = {
                    "build": tool_path,
                    "networks": ["master-network"],
                    "restart": "unless-stopped"
                }
                
        # Write master compose file
        master_path = os.path.join(deployment_dir, "docker-compose.master.yml")
        with open(master_path, 'w') as f:
            yaml.dump(master_compose, f)
            
        # Generate master deployment script
        master_script = """#!/bin/bash
# Master deployment script for all services

set -e

echo "Deploying all services..."

# Build all images
docker-compose -f docker-compose.master.yml build

# Start services in dependency order
docker-compose -f docker-compose.master.yml up -d

# Wait for all services to be ready
sleep 30

# Health check all services
docker-compose -f docker-compose.master.yml ps

echo "All services deployed!"
"""
        
        script_path = os.path.join(deployment_dir, "deploy_all.sh")
        with open(script_path, 'w') as f:
            f.write(master_script)
        os.chmod(script_path, 0o755)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deployment and Testing Strategy Generator')
    parser.add_argument('--catalog', default='tool_catalog.json',
                       help='Path to tool catalog JSON file')
    parser.add_argument('--output', default='deployment_strategies',
                       help='Output directory for generated files')
    
    args = parser.parse_args()
    
    # Create generator
    generator = DeploymentTestingGenerator(args.catalog)
    
    # Generate all strategies
    generator.generate_all_strategies(args.output)
    
    print(f"Generated deployment and testing strategies in {args.output}")
    

if __name__ == '__main__':
    main()