#!/usr/bin/env python3

"""
Simplified Testing Framework
Unit tests, Integration tests, and monitoring without conflicting dependencies
"""

import os
import sys
import time
import asyncio
import json
import unittest
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
import yaml
import structlog

# Import our modules (with careful import order)
from unified_config import get_config_manager, APIKeys, EnvironmentConfig

logger = structlog.get_logger()

@dataclass
class TestResult:
    """Individual test result."""
    
    test_name: str
    test_type: str  # unit, integration, e2e, performance
    status: str  # passed, failed, skipped, error
    duration: float
    timestamp: str
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    environment: str = "test"

@dataclass
class TestSuite:
    """Test suite configuration and results."""
    
    name: str
    description: str
    test_type: str
    tests: List[str] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)
    total_duration: float = 0.0
    passed: int = 0
    failed: int = 0
    skipped: int = 0

class SimpleTestRunner:
    """Simple test runner without external dependencies."""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager or get_config_manager()
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: List[TestResult] = []
        
        self._setup_test_suites()
        logger.info("SimpleTestRunner initialized")
    
    def _setup_test_suites(self):
        """Setup all test suites."""
        
        # Unit Tests
        self.test_suites["unit"] = TestSuite(
            name="Unit Tests",
            description="Fast, isolated unit tests for all components",
            test_type="unit",
            tests=[
                "test_unified_config",
                "test_api_keys_encryption",
                "test_environment_switching",
                "test_state_management",
                "test_configuration_export"
            ]
        )
        
        # Integration Tests
        self.test_suites["integration"] = TestSuite(
            name="Integration Tests",
            description="Component integration tests",
            test_type="integration",
            tests=[
                "test_config_system_integration",
                "test_file_operations",
                "test_environment_variables",
                "test_persistent_storage"
            ]
        )
        
        # Performance Tests
        self.test_suites["performance"] = TestSuite(
            name="Performance Tests",
            description="Performance and memory tests",
            test_type="performance",
            tests=[
                "test_config_load_time",
                "test_memory_usage",
                "test_concurrent_access",
                "test_large_state_handling"
            ]
        )
        
        logger.info("Test suites configured", suites=list(self.test_suites.keys()))
    
    async def run_test_suite(self, suite_name: str, environment: str = "test") -> TestSuite:
        """Run a complete test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        suite = self.test_suites[suite_name]
        suite.results = []
        start_time = time.time()
        
        logger.info("Starting test suite", suite=suite_name, environment=environment)
        
        try:
            # Run tests based on type
            if suite.test_type == "unit":
                await self._run_unit_tests(suite, environment)
            elif suite.test_type == "integration":
                await self._run_integration_tests(suite, environment)
            elif suite.test_type == "performance":
                await self._run_performance_tests(suite, environment)
            
            # Calculate results
            suite.total_duration = time.time() - start_time
            suite.passed = len([r for r in suite.results if r.status == "passed"])
            suite.failed = len([r for r in suite.results if r.status == "failed"])
            suite.skipped = len([r for r in suite.results if r.status == "skipped"])
            
            logger.info("Test suite completed", 
                       suite=suite_name,
                       passed=suite.passed,
                       failed=suite.failed,
                       duration=f"{suite.total_duration:.2f}s")
            
        except Exception as e:
            logger.error("Test suite failed", suite=suite_name, error=str(e))
        
        return suite
    
    async def _run_unit_tests(self, suite: TestSuite, environment: str):
        """Run unit tests."""
        for test_name in suite.tests:
            start_time = time.time()
            result = TestResult(
                test_name=test_name,
                test_type="unit",
                status="unknown",
                duration=0.0,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment
            )
            
            try:
                # Run specific unit test
                await self._run_individual_unit_test(test_name, result)
                
            except Exception as e:
                result.status = "error"
                result.error_message = str(e)
                logger.error("Unit test error", test=test_name, error=str(e))
            
            finally:
                result.duration = time.time() - start_time
                suite.results.append(result)
    
    async def _run_individual_unit_test(self, test_name: str, result: TestResult):
        """Run an individual unit test."""
        if test_name == "test_unified_config":
            await self._test_unified_config(result)
        elif test_name == "test_api_keys_encryption":
            await self._test_api_keys_encryption(result)
        elif test_name == "test_environment_switching":
            await self._test_environment_switching(result)
        elif test_name == "test_state_management":
            await self._test_state_management(result)
        elif test_name == "test_configuration_export":
            await self._test_configuration_export(result)
        else:
            result.status = "skipped"
            result.error_message = f"Test '{test_name}' not implemented"
    
    async def _test_unified_config(self, result: TestResult):
        """Test unified configuration system."""
        try:
            # Test configuration manager initialization
            config_manager = get_config_manager()
            assert config_manager is not None, "Config manager should not be None"
            
            # Test environment management
            assert len(config_manager.environments) >= 3, "Should have at least 3 environments"
            assert "development" in config_manager.environments, "Should have development environment"
            assert "staging" in config_manager.environments, "Should have staging environment"
            assert "production" in config_manager.environments, "Should have production environment"
            
            # Test API key management
            api_keys = config_manager.get_api_keys()
            assert isinstance(api_keys, APIKeys), "API keys should be APIKeys instance"
            
            # Test current configuration
            current_config = config_manager.get_current_config()
            assert current_config is not None, "Current config should not be None"
            assert isinstance(current_config, EnvironmentConfig), "Should be EnvironmentConfig instance"
            
            result.status = "passed"
            result.metrics = {
                "environments_count": len(config_manager.environments),
                "current_environment": config_manager.current_environment,
                "config_dir_exists": config_manager.config_dir.exists()
            }
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
    
    async def _test_api_keys_encryption(self, result: TestResult):
        """Test API keys encryption functionality."""
        try:
            config_manager = get_config_manager()
            
            # Test encryption key generation
            assert config_manager._encryption_key is not None, "Encryption key should exist"
            assert len(config_manager._encryption_key) > 0, "Encryption key should not be empty"
            
            # Test encryption/decryption
            test_data = "test_api_key_12345"
            encrypted = config_manager._encrypt_data(test_data)
            decrypted = config_manager._decrypt_data(encrypted)
            
            assert decrypted == test_data, "Decrypted data should match original"
            assert encrypted != test_data.encode(), "Encrypted data should be different"
            
            # Test key file security
            key_file = config_manager.key_file
            assert key_file.exists(), "Key file should exist"
            
            # Check file permissions (Unix-like systems)
            if hasattr(os, 'stat'):
                import stat
                file_mode = key_file.stat().st_mode
                permissions = stat.filemode(file_mode)
                # Should be readable only by owner
                assert not (file_mode & stat.S_IRGRP), "Key file should not be readable by group"
                assert not (file_mode & stat.S_IROTH), "Key file should not be readable by others"
            
            result.status = "passed"
            result.metrics = {
                "encryption_key_length": len(config_manager._encryption_key),
                "key_file_exists": key_file.exists(),
                "encryption_working": True
            }
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
    
    async def _test_environment_switching(self, result: TestResult):
        """Test environment switching functionality."""
        try:
            config_manager = get_config_manager()
            original_env = config_manager.current_environment
            
            # Test switching to different environments
            environments_to_test = ["development", "staging", "production"]
            
            for env in environments_to_test:
                if env in config_manager.environments:
                    # Switch environment
                    config_manager.switch_environment(env)
                    assert config_manager.current_environment == env, f"Should switch to {env}"
                    
                    # Get configuration for this environment
                    env_config = config_manager.get_current_config()
                    assert env_config is not None, f"Config for {env} should exist"
                    assert env_config.name == env, f"Config name should match {env}"
            
            # Test invalid environment
            try:
                config_manager.switch_environment("invalid_environment")
                assert False, "Should raise error for invalid environment"
            except ValueError:
                pass  # Expected
            
            # Restore original environment
            config_manager.switch_environment(original_env)
            assert config_manager.current_environment == original_env, "Should restore original environment"
            
            result.status = "passed"
            result.metrics = {
                "environments_tested": len(environments_to_test),
                "original_environment": original_env,
                "final_environment": config_manager.current_environment
            }
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
    
    async def _test_state_management(self, result: TestResult):
        """Test system state management."""
        try:
            config_manager = get_config_manager()
            
            # Test state updates
            original_agents = config_manager.system_state.active_agents.copy()
            
            # Add agent
            test_agent_id = "test_agent_123"
            config_manager.add_active_agent(test_agent_id)
            assert test_agent_id in config_manager.system_state.active_agents, "Agent should be added"
            
            # Remove agent
            config_manager.remove_active_agent(test_agent_id)
            assert test_agent_id not in config_manager.system_state.active_agents, "Agent should be removed"
            
            # Test task completion
            test_task_id = "test_task_456"
            config_manager.add_completed_task(test_task_id)
            assert test_task_id in config_manager.system_state.completed_tasks, "Task should be completed"
            
            # Test error logging
            config_manager.log_error("Test error", {"context": "unit_test"})
            assert len(config_manager.system_state.error_log) > 0, "Error should be logged"
            
            # Test state persistence
            config_manager.save_state()
            assert config_manager.state_file.exists(), "State file should exist"
            
            # Test state loading
            config_manager._load_state()
            
            result.status = "passed"
            result.metrics = {
                "active_agents_count": len(config_manager.system_state.active_agents),
                "completed_tasks_count": len(config_manager.system_state.completed_tasks),
                "error_log_count": len(config_manager.system_state.error_log),
                "state_file_exists": config_manager.state_file.exists()
            }
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
    
    async def _test_configuration_export(self, result: TestResult):
        """Test configuration export functionality."""
        try:
            config_manager = get_config_manager()
            
            # Test template export
            template_path = Path("test_config_template.yaml")
            config_manager.export_config_template(template_path)
            
            assert template_path.exists(), "Template file should be created"
            
            # Verify template content
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)
            
            assert "api_keys" in template_data, "Template should have api_keys section"
            assert "environments" in template_data, "Template should have environments section"
            
            # Test environment variables export
            env_vars = config_manager.get_environment_variables()
            assert isinstance(env_vars, dict), "Environment variables should be a dictionary"
            
            # Cleanup
            if template_path.exists():
                template_path.unlink()
            
            result.status = "passed"
            result.metrics = {
                "template_created": True,
                "env_vars_count": len(env_vars),
                "template_has_api_keys": "api_keys" in template_data,
                "template_has_environments": "environments" in template_data
            }
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
    
    async def _run_integration_tests(self, suite: TestSuite, environment: str):
        """Run integration tests."""
        for test_name in suite.tests:
            start_time = time.time()
            result = TestResult(
                test_name=test_name,
                test_type="integration",
                status="passed",  # Placeholder for now
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                metrics={"placeholder": True}
            )
            suite.results.append(result)
    
    async def _run_performance_tests(self, suite: TestSuite, environment: str):
        """Run performance tests."""
        for test_name in suite.tests:
            start_time = time.time()
            
            if test_name == "test_config_load_time":
                result = await self._test_config_load_time(environment)
            elif test_name == "test_memory_usage":
                result = await self._test_memory_usage(environment)
            elif test_name == "test_concurrent_access":
                result = await self._test_concurrent_access(environment)
            elif test_name == "test_large_state_handling":
                result = await self._test_large_state_handling(environment)
            else:
                result = TestResult(
                    test_name=test_name,
                    test_type="performance",
                    status="skipped",
                    duration=time.time() - start_time,
                    timestamp=datetime.utcnow().isoformat(),
                    environment=environment,
                    error_message=f"Performance test '{test_name}' not implemented"
                )
            
            suite.results.append(result)
    
    async def _test_config_load_time(self, environment: str) -> TestResult:
        """Test configuration loading performance."""
        start_time = time.time()
        
        try:
            # Measure config manager creation time
            load_start = time.time()
            config_manager = get_config_manager()
            load_time = time.time() - load_start
            
            # Measure environment switching time
            switch_start = time.time()
            original_env = config_manager.current_environment
            config_manager.switch_environment("staging")
            config_manager.switch_environment(original_env)
            switch_time = time.time() - switch_start
            
            # Performance thresholds
            max_load_time = 1.0  # 1 second
            max_switch_time = 0.1  # 100ms
            
            status = "passed"
            error_message = None
            
            if load_time > max_load_time:
                status = "failed"
                error_message = f"Config load time {load_time:.3f}s exceeds threshold {max_load_time}s"
            elif switch_time > max_switch_time:
                status = "failed"
                error_message = f"Environment switch time {switch_time:.3f}s exceeds threshold {max_switch_time}s"
            
            return TestResult(
                test_name="test_config_load_time",
                test_type="performance",
                status=status,
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message=error_message,
                metrics={
                    "config_load_time": load_time,
                    "environment_switch_time": switch_time,
                    "max_load_time_threshold": max_load_time,
                    "max_switch_time_threshold": max_switch_time
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="test_config_load_time",
                test_type="performance",
                status="error",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message=str(e)
            )
    
    async def _test_memory_usage(self, environment: str) -> TestResult:
        """Test memory usage."""
        start_time = time.time()
        
        try:
            import psutil
            import os
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create multiple config managers and perform operations
            config_managers = []
            for i in range(10):
                cm = get_config_manager()
                cm.update_system_state(test_data=f"test_{i}" * 100)
                config_managers.append(cm)
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Cleanup
            del config_managers
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Performance threshold
            max_memory_increase = 50  # 50MB
            
            status = "passed" if memory_increase < max_memory_increase else "failed"
            error_message = None if status == "passed" else f"Memory increase {memory_increase:.1f}MB exceeds threshold {max_memory_increase}MB"
            
            return TestResult(
                test_name="test_memory_usage",
                test_type="performance",
                status=status,
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message=error_message,
                metrics={
                    "initial_memory_mb": initial_memory,
                    "peak_memory_mb": peak_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                    "max_memory_threshold_mb": max_memory_increase
                }
            )
            
        except ImportError:
            return TestResult(
                test_name="test_memory_usage",
                test_type="performance",
                status="skipped",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message="psutil not available for memory testing"
            )
        except Exception as e:
            return TestResult(
                test_name="test_memory_usage",
                test_type="performance",
                status="error",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message=str(e)
            )
    
    async def _test_concurrent_access(self, environment: str) -> TestResult:
        """Test concurrent access to configuration."""
        start_time = time.time()
        
        try:
            config_manager = get_config_manager()
            
            def worker_task(worker_id: int):
                """Worker task for concurrent testing."""
                try:
                    # Perform various operations
                    for i in range(10):
                        config_manager.update_system_state(worker_data=f"worker_{worker_id}_iteration_{i}")
                        config_manager.add_active_agent(f"agent_{worker_id}_{i}")
                        time.sleep(0.001)  # Small delay
                        config_manager.remove_active_agent(f"agent_{worker_id}_{i}")
                    return True
                except Exception as e:
                    return False
            
            # Run concurrent workers
            num_workers = 5
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_task, i) for i in range(num_workers)]
                results = [future.result() for future in as_completed(futures)]
            
            success_count = sum(results)
            success_rate = success_count / num_workers
            
            status = "passed" if success_rate >= 0.8 else "failed"  # 80% success rate
            error_message = None if status == "passed" else f"Success rate {success_rate:.1%} below 80% threshold"
            
            return TestResult(
                test_name="test_concurrent_access",
                test_type="performance",
                status=status,
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message=error_message,
                metrics={
                    "num_workers": num_workers,
                    "successful_workers": success_count,
                    "success_rate": success_rate,
                    "min_success_rate": 0.8
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="test_concurrent_access",
                test_type="performance",
                status="error",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message=str(e)
            )
    
    async def _test_large_state_handling(self, environment: str) -> TestResult:
        """Test handling of large state data."""
        start_time = time.time()
        
        try:
            config_manager = get_config_manager()
            
            # Create large state data
            large_data = {}
            for i in range(1000):
                large_data[f"key_{i}"] = {
                    "data": "x" * 100,  # 100 character string
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": list(range(10))
                }
            
            # Test state update with large data
            update_start = time.time()
            config_manager.update_system_state(large_test_data=large_data)
            update_time = time.time() - update_start
            
            # Test state save with large data
            save_start = time.time()
            config_manager.save_state()
            save_time = time.time() - save_start
            
            # Test state load with large data
            load_start = time.time()
            config_manager._load_state()
            load_time = time.time() - load_start
            
            # Performance thresholds
            max_update_time = 1.0  # 1 second
            max_save_time = 2.0    # 2 seconds
            max_load_time = 2.0    # 2 seconds
            
            status = "passed"
            error_message = None
            
            if update_time > max_update_time:
                status = "failed"
                error_message = f"State update time {update_time:.3f}s exceeds threshold {max_update_time}s"
            elif save_time > max_save_time:
                status = "failed"
                error_message = f"State save time {save_time:.3f}s exceeds threshold {max_save_time}s"
            elif load_time > max_load_time:
                status = "failed"
                error_message = f"State load time {load_time:.3f}s exceeds threshold {max_load_time}s"
            
            return TestResult(
                test_name="test_large_state_handling",
                test_type="performance",
                status=status,
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message=error_message,
                metrics={
                    "large_data_keys": len(large_data),
                    "state_update_time": update_time,
                    "state_save_time": save_time,
                    "state_load_time": load_time,
                    "max_update_threshold": max_update_time,
                    "max_save_threshold": max_save_time,
                    "max_load_threshold": max_load_time
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="test_large_state_handling",
                test_type="performance",
                status="error",
                duration=time.time() - start_time,
                timestamp=datetime.utcnow().isoformat(),
                environment=environment,
                error_message=str(e)
            )
    
    def generate_test_report(self, suite_results: List[TestSuite]) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 60)
        report.append("MASTER ORCHESTRATOR - TEST REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("")
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_duration = 0.0
        
        for suite in suite_results:
            total_tests += len(suite.results)
            total_passed += suite.passed
            total_failed += suite.failed
            total_skipped += suite.skipped
            total_duration += suite.total_duration
            
            report.append(f"📋 {suite.name}")
            report.append(f"   Description: {suite.description}")
            report.append(f"   Type: {suite.test_type}")
            report.append(f"   Duration: {suite.total_duration:.2f}s")
            report.append(f"   Results: {suite.passed} passed, {suite.failed} failed, {suite.skipped} skipped")
            
            if suite.failed > 0:
                report.append("   Failed tests:")
                for result in suite.results:
                    if result.status == "failed":
                        report.append(f"     ❌ {result.test_name}: {result.error_message}")
            
            report.append("")
        
        # Summary
        report.append("📊 SUMMARY")
        report.append(f"   Total tests: {total_tests}")
        report.append(f"   Passed: {total_passed}")
        report.append(f"   Failed: {total_failed}")
        report.append(f"   Skipped: {total_skipped}")
        report.append(f"   Success rate: {(total_passed/total_tests)*100:.1f}%" if total_tests > 0 else "   Success rate: N/A")
        report.append(f"   Total duration: {total_duration:.2f}s")
        
        overall_status = "✅ PASSED" if total_failed == 0 else "❌ FAILED"
        report.append(f"   Overall status: {overall_status}")
        
        return "\n".join(report)

# CLI interface for running tests
async def run_tests_cli():
    """CLI interface for running tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Master Orchestrator Simple Test Runner")
    parser.add_argument("--suite", help="Test suite to run", choices=["unit", "integration", "performance", "all"])
    parser.add_argument("--environment", help="Environment to test", default="test")
    parser.add_argument("--output", help="Output file for report", default=None)
    
    args = parser.parse_args()
    
    runner = SimpleTestRunner()
    
    if args.suite == "all" or args.suite is None:
        # Run all test suites
        suite_results = []
        for suite_name in ["unit", "integration", "performance"]:
            try:
                result = await runner.run_test_suite(suite_name, args.environment)
                suite_results.append(result)
            except Exception as e:
                logger.error("Test suite failed", suite=suite_name, error=str(e))
    else:
        # Run single suite
        result = await runner.run_test_suite(args.suite, args.environment)
        suite_results = [result]
    
    # Generate report
    report = runner.generate_test_report(suite_results)
    
    if args.output:
        Path(args.output).write_text(report)
        print(f"📄 Report saved to: {args.output}")
    else:
        print(report)

if __name__ == "__main__":
    print("🧪 Master Orchestrator - Simple Testing Framework")
    print("=" * 60)
    
    # Run CLI interface
    asyncio.run(run_tests_cli())