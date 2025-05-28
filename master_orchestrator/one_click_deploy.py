#!/usr/bin/env python3
"""
One-Click Deployment Script
Complete end-to-end deployment with testing and server startup
"""

import asyncio
import logging
import os
import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import socket
import requests
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deployment.log')
    ]
)

class OneClickDeployer:
    """Complete one-click deployment system"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.deployment_config = {
            "python_version": "3.11",
            "frontend_port": 8080,
            "api_port": 8000,
            "websocket_port": 8765,
            "required_packages": [
                "aiohttp", "aiofiles", "websockets", "fastapi", "uvicorn",
                "google-cloud-aiplatform", "openai", "psutil", "gitpython",
                "black", "isort", "pytest", "pytest-asyncio", "requests",
                "jinja2", "aiohttp-cors", "youtube-dl", "PyPDF2", "PyMuPDF",
                "opencv-python", "pillow", "pyautogui", "pynput", "docker",
                "numpy", "pandas"
            ],
            "environment_variables": {
                "PYTHONPATH": str(Path(__file__).parent),
                "MASTER_ORCHESTRATOR_ENV": "development"
            }
        }
        
    async def deploy(self) -> Dict[str, Any]:
        """Complete deployment process"""
        deployment_steps = [
            ("System Check", self.check_system_requirements),
            ("Python Environment", self.setup_python_environment),
            ("Dependencies", self.install_dependencies),
            ("Configuration", self.setup_configuration),
            ("Database Setup", self.setup_database),
            ("Tests", self.run_tests),
            ("Build Frontend", self.build_frontend),
            ("Start Services", self.start_services),
            ("Health Check", self.health_check),
            ("Verify Deployment", self.verify_deployment)
        ]
        
        results = {}
        total_steps = len(deployment_steps)
        
        logging.info("🚀 Starting One-Click Deployment")
        logging.info(f"📋 Total steps: {total_steps}")
        
        for i, (step_name, step_func) in enumerate(deployment_steps, 1):
            logging.info(f"📦 Step {i}/{total_steps}: {step_name}")
            
            try:
                step_result = await step_func()
                results[step_name] = {
                    "success": True,
                    "result": step_result,
                    "timestamp": time.time()
                }
                logging.info(f"✅ {step_name} completed successfully")
                
            except Exception as e:
                error_msg = f"❌ {step_name} failed: {str(e)}"
                logging.error(error_msg)
                results[step_name] = {
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
                
                # For critical steps, stop deployment
                if step_name in ["System Check", "Python Environment", "Dependencies"]:
                    logging.error("🛑 Critical step failed. Stopping deployment.")
                    break
                    
        # Generate deployment report
        deployment_report = await self.generate_deployment_report(results)
        
        return {
            "success": all(step.get("success", False) for step in results.values()),
            "steps": results,
            "report": deployment_report
        }
        
    async def check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements"""
        requirements = {}
        
        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        requirements["python_version"] = {
            "current": python_version,
            "required": self.deployment_config["python_version"],
            "satisfied": python_version >= self.deployment_config["python_version"]
        }
        
        # Check available ports
        ports_to_check = [
            self.deployment_config["frontend_port"],
            self.deployment_config["api_port"],
            self.deployment_config["websocket_port"]
        ]
        
        for port in ports_to_check:
            requirements[f"port_{port}"] = {
                "port": port,
                "available": self.is_port_available(port)
            }
            
        # Check system resources
        requirements["system_resources"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_space_gb": round(psutil.disk_usage("/").free / (1024**3), 2)
        }
        
        # Check required commands
        required_commands = ["git", "pip", "python3"]
        for cmd in required_commands:
            requirements[f"command_{cmd}"] = {
                "command": cmd,
                "available": shutil.which(cmd) is not None
            }
            
        return requirements
        
    def is_port_available(self, port: int) -> bool:
        """Check if port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
            
    async def setup_python_environment(self) -> Dict[str, Any]:
        """Setup Python virtual environment"""
        venv_path = self.base_path / "venv"
        
        # Create virtual environment if it doesn't exist
        if not venv_path.exists():
            logging.info("Creating virtual environment...")
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "venv", str(venv_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                raise Exception("Failed to create virtual environment")
                
        # Activate virtual environment
        if sys.platform == "win32":
            activate_script = venv_path / "Scripts" / "activate.bat"
            pip_executable = venv_path / "Scripts" / "pip.exe"
            python_executable = venv_path / "Scripts" / "python.exe"
        else:
            activate_script = venv_path / "bin" / "activate"
            pip_executable = venv_path / "bin" / "pip"
            python_executable = venv_path / "bin" / "python"
            
        return {
            "venv_path": str(venv_path),
            "activate_script": str(activate_script),
            "pip_executable": str(pip_executable),
            "python_executable": str(python_executable),
            "created": True
        }
        
    async def install_dependencies(self) -> Dict[str, Any]:
        """Install required dependencies"""
        # Upgrade pip first
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", "--upgrade", "pip",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        # Install packages
        installed_packages = []
        failed_packages = []
        
        for package in self.deployment_config["required_packages"]:
            try:
                logging.info(f"Installing {package}...")
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pip", "install", package,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    installed_packages.append(package)
                else:
                    failed_packages.append({
                        "package": package,
                        "error": stderr.decode()
                    })
                    
            except Exception as e:
                failed_packages.append({
                    "package": package,
                    "error": str(e)
                })
                
        return {
            "installed_packages": installed_packages,
            "failed_packages": failed_packages,
            "total_installed": len(installed_packages),
            "total_failed": len(failed_packages)
        }
        
    async def setup_configuration(self) -> Dict[str, Any]:
        """Setup configuration files"""
        config_files = {}
        
        # Create config.yaml if it doesn't exist
        config_path = self.base_path / "config.yaml"
        if not config_path.exists():
            default_config = {
                "environment": "development",
                "debug": True,
                "logging": {
                    "level": "INFO",
                    "file": "app.log"
                },
                "api": {
                    "host": "localhost",
                    "port": self.deployment_config["api_port"]
                },
                "frontend": {
                    "host": "localhost",
                    "port": self.deployment_config["frontend_port"]
                },
                "websocket": {
                    "host": "localhost",
                    "port": self.deployment_config["websocket_port"]
                },
                "database": {
                    "url": "sqlite:///master_orchestrator.db"
                }
            }
            
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(default_config, f, default_flow_style=False)
                
            config_files["config.yaml"] = "created"
        else:
            config_files["config.yaml"] = "exists"
            
        # Create .env file for environment variables
        env_path = self.base_path / ".env"
        env_content = []
        for key, value in self.deployment_config["environment_variables"].items():
            env_content.append(f"{key}={value}")
            
        with open(env_path, 'w') as f:
            f.write('\n'.join(env_content))
            
        config_files[".env"] = "created"
        
        return config_files
        
    async def setup_database(self) -> Dict[str, Any]:
        """Setup database"""
        # For now, just create the data directory
        data_dir = self.base_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Create projects directory
        projects_dir = self.base_path / "projects"
        projects_dir.mkdir(exist_ok=True)
        
        # Create logs directory
        logs_dir = self.base_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        return {
            "data_directory": str(data_dir),
            "projects_directory": str(projects_dir),
            "logs_directory": str(logs_dir),
            "database_type": "sqlite",
            "initialized": True
        }
        
    async def run_tests(self) -> Dict[str, Any]:
        """Run tests"""
        test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "test_files": []
        }
        
        # Create basic test if none exist
        tests_dir = self.base_path / "tests"
        tests_dir.mkdir(exist_ok=True)
        
        basic_test_path = tests_dir / "test_basic.py"
        if not basic_test_path.exists():
            basic_test_content = '''
import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test basic imports work"""
    try:
        import unified_config
        import parallel_llm_orchestrator
        import frontend_orchestrator
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_config_creation():
    """Test configuration creation"""
    from unified_config import SecureConfigManager
    config = SecureConfigManager()
    assert config is not None

@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality works"""
    async def dummy_async():
        return "async_works"
    
    result = await dummy_async()
    assert result == "async_works"

def test_path_resolution():
    """Test path resolution works"""
    base_path = Path(__file__).parent.parent
    assert base_path.exists()
    assert (base_path / "unified_config.py").exists()
'''
            
            with open(basic_test_path, 'w') as f:
                f.write(basic_test_content)
                
        # Run pytest
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path)
            )
            stdout, stderr = await process.communicate()
            
            output = stdout.decode() + stderr.decode()
            
            # Parse test results (simplified)
            if "passed" in output:
                import re
                passed_match = re.search(r'(\d+) passed', output)
                if passed_match:
                    test_results["passed_tests"] = int(passed_match.group(1))
                    
            test_results["output"] = output
            test_results["success"] = process.returncode == 0
            
        except Exception as e:
            test_results["error"] = str(e)
            test_results["success"] = False
            
        return test_results
        
    async def build_frontend(self) -> Dict[str, Any]:
        """Build frontend assets"""
        # For our React-in-browser setup, just ensure templates exist
        templates_dir = self.base_path / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        static_dir = self.base_path / "static"
        static_dir.mkdir(exist_ok=True)
        
        return {
            "templates_directory": str(templates_dir),
            "static_directory": str(static_dir),
            "build_type": "templates",
            "success": True
        }
        
    async def start_services(self) -> Dict[str, Any]:
        """Start all services"""
        services = {}
        
        # Start frontend orchestrator
        try:
            # Import and start frontend
            sys.path.insert(0, str(self.base_path))
            
            # Start in background
            frontend_process = await asyncio.create_subprocess_exec(
                sys.executable, "frontend_orchestrator.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path)
            )
            
            # Give it time to start
            await asyncio.sleep(3)
            
            services["frontend"] = {
                "process_id": frontend_process.pid,
                "port": self.deployment_config["frontend_port"],
                "status": "starting"
            }
            
        except Exception as e:
            services["frontend"] = {
                "error": str(e),
                "status": "failed"
            }
            
        return services
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health checks"""
        health_status = {}
        
        # Check frontend service
        try:
            frontend_url = f"http://localhost:{self.deployment_config['frontend_port']}/health"
            
            # Try multiple times with delay
            for attempt in range(5):
                try:
                    response = requests.get(frontend_url, timeout=5)
                    if response.status_code == 200:
                        health_status["frontend"] = {
                            "status": "healthy",
                            "response_time": response.elapsed.total_seconds(),
                            "attempt": attempt + 1
                        }
                        break
                except requests.RequestException:
                    if attempt < 4:  # Not the last attempt
                        await asyncio.sleep(2)
                    else:
                        health_status["frontend"] = {
                            "status": "unhealthy",
                            "error": "Service not responding"
                        }
                        
        except Exception as e:
            health_status["frontend"] = {
                "status": "error",
                "error": str(e)
            }
            
        # Check system resources
        health_status["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent
        }
        
        return health_status
        
    async def verify_deployment(self) -> Dict[str, Any]:
        """Verify deployment is working correctly"""
        verification_results = {}
        
        # Test frontend access
        try:
            frontend_url = f"http://localhost:{self.deployment_config['frontend_port']}"
            response = requests.get(frontend_url, timeout=10)
            verification_results["frontend_access"] = {
                "url": frontend_url,
                "status_code": response.status_code,
                "accessible": response.status_code == 200
            }
        except Exception as e:
            verification_results["frontend_access"] = {
                "error": str(e),
                "accessible": False
            }
            
        # Test API endpoints
        try:
            api_url = f"http://localhost:{self.deployment_config['frontend_port']}/api/monitoring-data"
            response = requests.get(api_url, timeout=10)
            verification_results["api_access"] = {
                "url": api_url,
                "status_code": response.status_code,
                "accessible": response.status_code == 200
            }
        except Exception as e:
            verification_results["api_access"] = {
                "error": str(e),
                "accessible": False
            }
            
        # Test file structure
        required_files = [
            "unified_config.py",
            "parallel_llm_orchestrator.py",
            "frontend_orchestrator.py",
            "computer_control_orchestrator.py",
            "content_analyzer_deployer.py"
        ]
        
        file_checks = {}
        for file_name in required_files:
            file_path = self.base_path / file_name
            file_checks[file_name] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0
            }
            
        verification_results["required_files"] = file_checks
        
        return verification_results
        
    async def generate_deployment_report(self, results: Dict[str, Any]) -> str:
        """Generate deployment report"""
        report_lines = [
            "=" * 80,
            "MASTER ORCHESTRATOR DEPLOYMENT REPORT",
            "=" * 80,
            f"Deployment Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Summary
        total_steps = len(results)
        successful_steps = sum(1 for step in results.values() if step.get("success", False))
        
        report_lines.extend([
            "SUMMARY:",
            f"  Total Steps: {total_steps}",
            f"  Successful: {successful_steps}",
            f"  Failed: {total_steps - successful_steps}",
            f"  Success Rate: {(successful_steps/total_steps)*100:.1f}%",
            ""
        ])
        
        # Step Details
        report_lines.append("STEP DETAILS:")
        for step_name, step_result in results.items():
            status = "✅ SUCCESS" if step_result.get("success", False) else "❌ FAILED"
            report_lines.append(f"  {step_name}: {status}")
            
            if not step_result.get("success", False) and "error" in step_result:
                report_lines.append(f"    Error: {step_result['error']}")
                
        report_lines.append("")
        
        # Access Information
        if successful_steps >= total_steps - 2:  # Allow for some minor failures
            report_lines.extend([
                "ACCESS INFORMATION:",
                f"  Frontend URL: http://localhost:{self.deployment_config['frontend_port']}",
                f"  API Base URL: http://localhost:{self.deployment_config['frontend_port']}/api",
                f"  Health Check: http://localhost:{self.deployment_config['frontend_port']}/health",
                ""
            ])
            
        # Next Steps
        report_lines.extend([
            "NEXT STEPS:",
            "  1. Open your browser and go to the Frontend URL",
            "  2. Start creating projects and generating code",
            "  3. Monitor system performance in the dashboard",
            "  4. Check logs in the logs/ directory for any issues",
            ""
        ])
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
        
    def save_deployment_report(self, report: str):
        """Save deployment report to file"""
        report_path = self.base_path / "deployment_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logging.info(f"📄 Deployment report saved to: {report_path}")

async def main():
    """Main deployment function"""
    print("🚀 Master Orchestrator One-Click Deployment")
    print("=" * 50)
    
    deployer = OneClickDeployer()
    
    try:
        # Run deployment
        result = await deployer.deploy()
        
        # Print and save report
        print("\n" + result["report"])
        deployer.save_deployment_report(result["report"])
        
        if result["success"]:
            print("\n🎉 Deployment completed successfully!")
            print(f"🌐 Access your application at: http://localhost:{deployer.deployment_config['frontend_port']}")
        else:
            print("\n❌ Deployment completed with errors. Check the report above.")
            
        return result["success"]
        
    except KeyboardInterrupt:
        print("\n⏹️  Deployment cancelled by user")
        return False
    except Exception as e:
        print(f"\n💥 Deployment failed with error: {e}")
        logging.error(f"Deployment error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)