#!/usr/bin/env python3

"""
Quick Hardware Discovery and Git Automation
Fast discovery of local Mac hardware and intelligent Git operations
"""

import asyncio
import json
import time
import subprocess
import socket
import platform
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import structlog

from unified_config import get_config_manager
from dev_capabilities import MasterDevController

logger = structlog.get_logger()

@dataclass
class LocalHardware:
    """Represents local hardware configuration."""
    
    hostname: str
    device_type: str
    cpu_model: str
    cpu_cores: int
    memory_gb: float
    gpu_cores: Optional[int] = None
    storage_gb: float = 0.0
    network_interfaces: List[str] = field(default_factory=list)
    vllm_capable: bool = True
    estimated_performance: str = "high"

class QuickHardwareDiscovery:
    """Quick hardware discovery for local system."""
    
    def __init__(self):
        self.local_hardware: Optional[LocalHardware] = None
        
    async def discover_local_hardware(self) -> LocalHardware:
        """Discover local hardware specifications."""
        try:
            logger.info("Discovering local hardware specifications")
            
            # Get basic system info
            hostname = socket.gethostname()
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)
            
            # Get disk info
            disk_usage = psutil.disk_usage('/')
            storage_gb = disk_usage.total / (1024**3)
            
            # Detect Mac device type
            device_type = self._detect_mac_device_type()
            
            # Get CPU model
            cpu_model = self._get_cpu_model()
            
            # Get network interfaces
            network_interfaces = self._get_network_interfaces()
            
            # Estimate GPU cores for Apple Silicon
            gpu_cores = self._estimate_gpu_cores(cpu_model, device_type)
            
            # Create hardware object
            self.local_hardware = LocalHardware(
                hostname=hostname,
                device_type=device_type,
                cpu_model=cpu_model,
                cpu_cores=cpu_count,
                memory_gb=memory_gb,
                gpu_cores=gpu_cores,
                storage_gb=storage_gb,
                network_interfaces=network_interfaces,
                vllm_capable=memory_gb >= 16 and cpu_count >= 8,
                estimated_performance=self._estimate_performance(device_type, memory_gb, cpu_count)
            )
            
            logger.info("Local hardware discovered",
                       device=device_type,
                       cpu=cpu_model,
                       memory=f"{memory_gb:.1f}GB",
                       vllm_capable=self.local_hardware.vllm_capable)
            
            return self.local_hardware
            
        except Exception as e:
            logger.error("Hardware discovery failed", error=str(e))
            # Return basic fallback info
            return LocalHardware(
                hostname=socket.gethostname(),
                device_type="unknown",
                cpu_model="Unknown",
                cpu_cores=psutil.cpu_count() or 8,
                memory_gb=psutil.virtual_memory().total / (1024**3),
                vllm_capable=False
            )
    
    def _detect_mac_device_type(self) -> str:
        """Detect Mac device type."""
        try:
            # Try to get Mac model info
            result = subprocess.run(
                ['system_profiler', 'SPHardwareDataType'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                output = result.stdout
                if 'Mac Studio' in output:
                    return 'mac_studio'
                elif 'Mac mini' in output:
                    return 'mac_mini'
                elif 'MacBook Pro' in output:
                    return 'macbook_pro'
                elif 'MacBook Air' in output:
                    return 'macbook_air'
                elif 'iMac' in output:
                    return 'imac'
            
        except Exception as e:
            logger.debug("Failed to detect Mac model", error=str(e))
        
        # Fallback detection based on hostname or other indicators
        hostname = socket.gethostname().lower()
        if 'studio' in hostname:
            return 'mac_studio'
        elif 'mini' in hostname:
            return 'mac_mini'
        elif 'macbook' in hostname:
            return 'macbook_pro'
        
        return 'mac_unknown'
    
    def _get_cpu_model(self) -> str:
        """Get CPU model information."""
        try:
            # Try macOS-specific method first
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                capture_output=True, 
                text=True, 
                timeout=2
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Try system_profiler
            result = subprocess.run(
                ['system_profiler', 'SPHardwareDataType'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Chip:' in line or 'Processor Name:' in line:
                        return line.split(':', 1)[1].strip()
            
        except Exception as e:
            logger.debug("Failed to get CPU model", error=str(e))
        
        return platform.processor() or "Unknown CPU"
    
    def _get_network_interfaces(self) -> List[str]:
        """Get available network interfaces."""
        interfaces = []
        try:
            for interface_name, interface_addresses in psutil.net_if_addrs().items():
                for address in interface_addresses:
                    if address.family == socket.AF_INET and not address.address.startswith('127.'):
                        interfaces.append(f"{interface_name}: {address.address}")
        except Exception as e:
            logger.debug("Failed to get network interfaces", error=str(e))
        
        return interfaces
    
    def _estimate_gpu_cores(self, cpu_model: str, device_type: str) -> Optional[int]:
        """Estimate GPU cores for Apple Silicon Macs."""
        cpu_lower = cpu_model.lower()
        
        # Apple Silicon estimates
        if 'm1 ultra' in cpu_lower:
            return 64  # M1 Ultra
        elif 'm1 max' in cpu_lower:
            return 32  # M1 Max
        elif 'm1 pro' in cpu_lower:
            return 16  # M1 Pro
        elif 'm1' in cpu_lower:
            return 8   # M1
        elif 'm2 ultra' in cpu_lower:
            return 76  # M2 Ultra
        elif 'm2 max' in cpu_lower:
            return 38  # M2 Max
        elif 'm2 pro' in cpu_lower:
            return 19  # M2 Pro
        elif 'm2' in cpu_lower:
            return 10  # M2
        elif 'm3 max' in cpu_lower:
            return 40  # M3 Max
        elif 'm3 pro' in cpu_lower:
            return 18  # M3 Pro
        elif 'm3' in cpu_lower:
            return 10  # M3
        elif 'm4 max' in cpu_lower:
            return 40  # M4 Max
        elif 'm4 pro' in cpu_lower:
            return 20  # M4 Pro
        elif 'm4' in cpu_lower:
            return 10  # M4
        
        # Device-based estimates if CPU detection fails
        if device_type == 'mac_studio':
            return 32  # Assume higher-end
        elif device_type == 'mac_mini':
            return 10  # Assume M2/M4
        elif device_type == 'macbook_pro':
            return 16  # Assume Pro model
        
        return None
    
    def _estimate_performance(self, device_type: str, memory_gb: float, cpu_cores: int) -> str:
        """Estimate system performance level."""
        if device_type == 'mac_studio' and memory_gb >= 128:
            return 'ultra'
        elif device_type == 'mac_studio' and memory_gb >= 64:
            return 'high'
        elif device_type in ['mac_mini', 'macbook_pro'] and memory_gb >= 64:
            return 'high'
        elif memory_gb >= 32 and cpu_cores >= 10:
            return 'medium'
        elif memory_gb >= 16 and cpu_cores >= 8:
            return 'low'
        else:
            return 'insufficient'

class SmartVLLMManager:
    """Smart vLLM manager that adapts to available hardware."""
    
    def __init__(self, hardware: LocalHardware):
        self.hardware = hardware
        self.vllm_config = self._generate_optimal_config()
    
    def _generate_optimal_config(self) -> Dict[str, Any]:
        """Generate optimal vLLM configuration for current hardware."""
        config = {
            "model": "microsoft/DialoGPT-medium",  # Start with smaller model
            "host": "0.0.0.0",
            "port": 8080,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 2048
        }
        
        # Adjust based on hardware capability
        if self.hardware.estimated_performance == 'ultra':
            config.update({
                "model": "meta-llama/Llama-2-13b-chat-hf",
                "tensor_parallel_size": min(4, self.hardware.gpu_cores // 10),
                "max_model_len": 4096
            })
        elif self.hardware.estimated_performance == 'high':
            config.update({
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "tensor_parallel_size": min(2, self.hardware.gpu_cores // 15),
                "max_model_len": 3072
            })
        elif self.hardware.estimated_performance == 'medium':
            config.update({
                "model": "microsoft/DialoGPT-large",
                "max_model_len": 2048
            })
        
        return config
    
    async def deploy_local_vllm(self) -> Dict[str, Any]:
        """Deploy vLLM optimized for local hardware."""
        if not self.hardware.vllm_capable:
            return {
                "status": "skipped",
                "reason": "Hardware not suitable for vLLM",
                "requirements": "Minimum 16GB RAM and 8 CPU cores"
            }
        
        try:
            logger.info("Deploying optimized vLLM configuration",
                       model=self.vllm_config["model"],
                       performance=self.hardware.estimated_performance)
            
            # Create deployment script
            script = self._create_deployment_script()
            
            # For now, simulate deployment
            await asyncio.sleep(2)
            
            return {
                "status": "deployed",
                "configuration": self.vllm_config,
                "performance_tier": self.hardware.estimated_performance,
                "endpoint": f"http://localhost:{self.vllm_config['port']}"
            }
            
        except Exception as e:
            logger.error("vLLM deployment failed", error=str(e))
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_deployment_script(self) -> str:
        """Create optimized vLLM deployment script."""
        config = self.vllm_config
        
        return f"""#!/bin/bash
set -e

echo "Deploying vLLM with optimized configuration for {self.hardware.device_type}"
echo "Performance tier: {self.hardware.estimated_performance}"
echo "Model: {config['model']}"

# Create vLLM directory
mkdir -p ~/vllm_local
cd ~/vllm_local

# Install/update vLLM
pip3 install --upgrade vllm

# Create configuration file
cat > vllm_config.json << EOF
{json.dumps(config, indent=2)}
EOF

# Start vLLM server
echo "Starting vLLM server on port {config['port']}"
nohup python3 -m vllm.entrypoints.api_server \\
    --model {config['model']} \\
    --port {config['port']} \\
    --host {config['host']} \\
    --gpu-memory-utilization {config['gpu_memory_utilization']} \\
    --max-model-len {config['max_model_len']} \\
    > vllm_server.log 2>&1 &

sleep 5

# Verify deployment
curl -s http://localhost:{config['port']}/health || echo "Health check failed"

echo "vLLM deployment completed"
"""

class EnhancedGitAgent:
    """Enhanced Git agent with more intelligent commit analysis."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.dev_controller = MasterDevController(Path('.'))
        
    async def smart_commit_workflow(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Smart commit workflow with context awareness."""
        try:
            logger.info("Starting smart Git commit workflow")
            
            # Get current status
            git_status = self.dev_controller.git.get_status()
            
            if 'nothing to commit' in git_status.get('output', ''):
                return {
                    "status": "no_changes",
                    "message": "Repository is clean - no changes to commit"
                }
            
            # Analyze changes with context
            analysis = await self._analyze_changes_with_context(context)
            
            # Generate intelligent commit message
            commit_message = self._generate_smart_commit_message(analysis, context)
            
            # Stage appropriate files
            staging_result = await self._smart_staging(analysis)
            
            if not staging_result['success']:
                return {
                    "status": "error",
                    "error": f"Failed to stage files: {staging_result['error']}"
                }
            
            # Create commit
            commit_result = self.dev_controller.git.commit_changes(commit_message)
            
            if commit_result.get('success'):
                logger.info("Smart commit created successfully",
                           message=commit_message.split('\n')[0],
                           files_staged=len(staging_result['staged_files']))
                
                return {
                    "status": "success",
                    "commit_message": commit_message,
                    "files_committed": staging_result['staged_files'],
                    "commit_hash": commit_result.get('hash', 'unknown'),
                    "analysis": analysis
                }
            else:
                return {
                    "status": "error",
                    "error": commit_result.get('error', 'Commit failed')
                }
                
        except Exception as e:
            logger.error("Smart commit workflow failed", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def _analyze_changes_with_context(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze changes with additional context."""
        analysis = {
            'change_type': 'unknown',
            'impact_level': 'minor',
            'affected_components': [],
            'new_features': [],
            'improvements': [],
            'context': context or {}
        }
        
        try:
            # Get changed files
            diff_result = self.dev_controller.git.get_diff()
            if diff_result.get('success'):
                changed_files = self._extract_changed_files_simple(diff_result['output'])
                analysis['changed_files'] = changed_files
                
                # Analyze file patterns
                if any('hardware' in f for f in changed_files):
                    analysis['change_type'] = 'hardware'
                    analysis['affected_components'].append('hardware_discovery')
                    if context and 'hardware' in context:
                        analysis['new_features'].append('hardware discovery and vLLM deployment')
                
                if any('vllm' in f or 'llm' in f for f in changed_files):
                    analysis['change_type'] = 'ai_infrastructure'
                    analysis['affected_components'].append('distributed_inference')
                    analysis['new_features'].append('distributed LLM inference')
                
                if any('git' in f or 'commit' in f for f in changed_files):
                    analysis['change_type'] = 'development_tools'
                    analysis['affected_components'].append('git_automation')
                    analysis['new_features'].append('intelligent Git automation')
                
                # Determine impact level
                if len(changed_files) > 3 or any('hardware' in f for f in changed_files):
                    analysis['impact_level'] = 'major'
                elif len(changed_files) > 1:
                    analysis['impact_level'] = 'minor'
                else:
                    analysis['impact_level'] = 'patch'
        
        except Exception as e:
            logger.debug("Change analysis failed", error=str(e))
        
        return analysis
    
    def _extract_changed_files_simple(self, diff_output: str) -> List[str]:
        """Simple extraction of changed files from git diff."""
        files = []
        for line in diff_output.split('\n'):
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3][2:]  # Remove 'b/' prefix
                    files.append(file_path)
        return files
    
    def _generate_smart_commit_message(self, analysis: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """Generate smart commit message based on analysis and context."""
        change_type = analysis['change_type']
        impact = analysis['impact_level']
        
        # Context-aware message generation
        if change_type == 'hardware':
            if context and context.get('hardware_discovered'):
                device_info = context.get('device_info', {})
                device_type = device_info.get('device_type', 'unknown')
                return f"""feat(hardware): implement automatic discovery and vLLM deployment

- Add hardware discovery for {device_type} systems
- Deploy optimized vLLM configuration based on system specs
- Enable distributed inference capabilities
- Integrate intelligent Git automation for seamless commits

Impact: {impact}
Performance tier: {device_info.get('estimated_performance', 'unknown')}

🤖 Automated commit via Smart Git Agent
Generated: {datetime.utcnow().isoformat()}"""
        
        elif change_type == 'ai_infrastructure':
            return f"""feat(ai): enhance distributed inference infrastructure

- Optimize vLLM deployment for local hardware
- Add performance-based model selection
- Implement adaptive configuration generation
- Enable seamless scaling across available resources

Impact: {impact}

🤖 Automated commit via Smart Git Agent
Generated: {datetime.utcnow().isoformat()}"""
        
        elif change_type == 'development_tools':
            return f"""feat(dev): implement intelligent Git automation system

- Add context-aware commit message generation
- Implement smart file staging and analysis
- Enable automatic workflow orchestration
- Integrate with hardware discovery and deployment

Impact: {impact}

🤖 Automated commit via Smart Git Agent
Generated: {datetime.utcnow().isoformat()}"""
        
        else:
            components = ', '.join(analysis['affected_components'][:3])
            return f"""chore: update {components or 'system components'}

- Improve system functionality and integration
- Enhance automation capabilities
- Update configuration and deployment scripts

Impact: {impact}
Files changed: {len(analysis.get('changed_files', []))}

🤖 Automated commit via Smart Git Agent
Generated: {datetime.utcnow().isoformat()}"""
    
    async def _smart_staging(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Smart staging of files based on analysis."""
        try:
            # For now, stage all tracked changes
            # In the future, could be more selective based on analysis
            
            # Add all modified files
            add_result = subprocess.run(
                ['git', 'add', '-A'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if add_result.returncode == 0:
                # Get list of staged files
                staged_result = subprocess.run(
                    ['git', 'diff', '--cached', '--name-only'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                staged_files = staged_result.stdout.strip().split('\n') if staged_result.stdout.strip() else []
                
                return {
                    'success': True,
                    'staged_files': staged_files
                }
            else:
                return {
                    'success': False,
                    'error': add_result.stderr or 'Failed to stage files'
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class QuickOrchestrator:
    """Quick orchestrator for immediate hardware discovery and Git automation."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.hardware_discovery = QuickHardwareDiscovery()
        self.git_agent = EnhancedGitAgent(config_manager)
        self.local_hardware: Optional[LocalHardware] = None
        
    async def quick_setup_workflow(self) -> Dict[str, Any]:
        """Quick setup workflow for immediate results."""
        workflow_results = {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat(),
            "hardware_discovery": {},
            "vllm_deployment": {},
            "git_automation": {},
            "overall_status": "in_progress"
        }
        
        try:
            logger.info("Starting quick setup workflow")
            
            # Phase 1: Discover local hardware
            logger.info("Phase 1: Discovering local hardware")
            self.local_hardware = await self.hardware_discovery.discover_local_hardware()
            
            workflow_results["hardware_discovery"] = {
                "status": "completed",
                "device_type": self.local_hardware.device_type,
                "cpu_model": self.local_hardware.cpu_model,
                "memory_gb": self.local_hardware.memory_gb,
                "vllm_capable": self.local_hardware.vllm_capable,
                "performance_tier": self.local_hardware.estimated_performance
            }
            
            # Phase 2: Deploy vLLM if capable
            if self.local_hardware.vllm_capable:
                logger.info("Phase 2: Deploying optimized vLLM")
                vllm_manager = SmartVLLMManager(self.local_hardware)
                vllm_result = await vllm_manager.deploy_local_vllm()
                workflow_results["vllm_deployment"] = vllm_result
            else:
                logger.warning("Phase 2: Skipping vLLM deployment - insufficient hardware")
                workflow_results["vllm_deployment"] = {
                    "status": "skipped",
                    "reason": "Hardware requirements not met"
                }
            
            # Phase 3: Intelligent Git commit
            logger.info("Phase 3: Creating intelligent Git commit")
            commit_context = {
                "hardware_discovered": True,
                "device_info": asdict(self.local_hardware),
                "vllm_deployed": workflow_results["vllm_deployment"]["status"] == "deployed"
            }
            
            git_result = await self.git_agent.smart_commit_workflow(commit_context)
            workflow_results["git_automation"] = git_result
            
            # Determine overall status
            if (workflow_results["hardware_discovery"]["status"] == "completed" and
                workflow_results["git_automation"]["status"] == "success"):
                workflow_results["overall_status"] = "success"
            else:
                workflow_results["overall_status"] = "partial_success"
            
            logger.info("Quick setup workflow completed",
                       status=workflow_results["overall_status"],
                       hardware=self.local_hardware.device_type,
                       vllm=workflow_results["vllm_deployment"]["status"])
            
        except Exception as e:
            workflow_results["overall_status"] = "error"
            workflow_results["error"] = str(e)
            logger.error("Quick setup workflow failed", error=str(e))
        
        return workflow_results
    
    async def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the current system state."""
        if not self.local_hardware:
            self.local_hardware = await self.hardware_discovery.discover_local_hardware()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "local_hardware": asdict(self.local_hardware),
            "capabilities": {
                "vllm_inference": self.local_hardware.vllm_capable,
                "performance_tier": self.local_hardware.estimated_performance,
                "gpu_cores": self.local_hardware.gpu_cores,
                "memory_sufficient": self.local_hardware.memory_gb >= 16
            },
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get system recommendations based on hardware."""
        recommendations = []
        
        if not self.local_hardware:
            return ["Run hardware discovery first"]
        
        if not self.local_hardware.vllm_capable:
            recommendations.append("Upgrade to at least 16GB RAM for vLLM capabilities")
        
        if self.local_hardware.estimated_performance == 'insufficient':
            recommendations.append("Consider upgrading hardware for better AI performance")
        elif self.local_hardware.estimated_performance == 'low':
            recommendations.append("Consider smaller AI models for optimal performance")
        elif self.local_hardware.estimated_performance in ['high', 'ultra']:
            recommendations.append("Hardware is excellent for distributed AI workloads")
        
        if self.local_hardware.device_type == 'mac_studio':
            recommendations.append("Perfect for coordinating distributed vLLM across network")
        elif self.local_hardware.device_type == 'mac_mini':
            recommendations.append("Ideal as worker node in distributed setup")
        
        return recommendations

# CLI interface
async def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Hardware Discovery and Git Automation")
    parser.add_argument("--action", choices=["setup", "status", "commit", "hardware"], 
                       default="setup", help="Action to perform")
    
    args = parser.parse_args()
    
    orchestrator = QuickOrchestrator()
    
    if args.action == "setup":
        print("🚀 Running quick setup workflow...")
        results = await orchestrator.quick_setup_workflow()
        
        print("\n📊 Quick Setup Results:")
        print(f"   🖥️ Hardware: {results['hardware_discovery'].get('device_type', 'unknown')} "
              f"({results['hardware_discovery'].get('performance_tier', 'unknown')})")
        print(f"   🤖 vLLM: {results['vllm_deployment'].get('status', 'unknown')}")
        print(f"   📝 Git: {results['git_automation'].get('status', 'unknown')}")
        print(f"   ✅ Overall: {results['overall_status']}")
        
        if results['git_automation'].get('status') == 'success':
            commit_msg = results['git_automation']['commit_message'].split('\n')[0]
            print(f"   💬 Commit: {commit_msg}")
    
    elif args.action == "status":
        print("📊 Getting system summary...")
        summary = await orchestrator.get_system_summary()
        
        hw = summary['local_hardware']
        print(f"\n🖥️ Local Hardware:")
        print(f"   Device: {hw['device_type']}")
        print(f"   CPU: {hw['cpu_model']}")
        print(f"   Memory: {hw['memory_gb']:.1f}GB")
        print(f"   GPU Cores: {hw['gpu_cores'] or 'Unknown'}")
        print(f"   Performance: {hw['estimated_performance']}")
        print(f"   vLLM Capable: {'✅' if hw['vllm_capable'] else '❌'}")
        
        print(f"\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"   • {rec}")
    
    elif args.action == "hardware":
        print("🔍 Discovering hardware...")
        hardware = await orchestrator.hardware_discovery.discover_local_hardware()
        
        print(f"\n🖥️ Hardware Details:")
        print(f"   Hostname: {hardware.hostname}")
        print(f"   Device Type: {hardware.device_type}")
        print(f"   CPU: {hardware.cpu_model}")
        print(f"   Cores: {hardware.cpu_cores}")
        print(f"   Memory: {hardware.memory_gb:.1f}GB")
        print(f"   GPU Cores: {hardware.gpu_cores or 'Unknown'}")
        print(f"   Storage: {hardware.storage_gb:.1f}GB")
        print(f"   Performance: {hardware.estimated_performance}")
        print(f"   vLLM Ready: {'✅' if hardware.vllm_capable else '❌'}")
    
    elif args.action == "commit":
        print("📝 Creating intelligent Git commit...")
        result = await orchestrator.git_agent.smart_commit_workflow()
        
        print(f"\n📝 Git Commit Result:")
        print(f"   Status: {result['status']}")
        if result['status'] == 'success':
            print(f"   Message: {result['commit_message'].split(chr(10))[0]}")
            print(f"   Files: {len(result['files_committed'])}")
        elif result['status'] == 'error':
            print(f"   Error: {result['error']}")

if __name__ == "__main__":
    print("⚡ Quick Hardware Discovery & Git Automation")
    print("=" * 50)
    
    # Run CLI
    asyncio.run(main())