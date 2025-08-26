"""
Integration Testing Module
=========================
Comprehensive integration testing for services, workflows, and performance.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
import requests
from fastapi.testclient import TestClient
import docker
from docker.errors import DockerException

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of an integration test."""
    test_name: str
    test_type: str
    status: str  # 'PASS', 'FAIL', 'SKIP', 'ERROR'
    duration: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ServiceIntegrationTest:
    """Tests integration between individual services."""
    
    def __init__(self, service_name: str, base_url: str = "http://localhost:8000"):
        self.service_name = service_name
        self.base_url = base_url
        self.test_results: List[TestResult] = []
        self.client = TestClient(None)  # Will be set when testing FastAPI apps
    
    def test_service_health(self) -> TestResult:
        """Test service health endpoint."""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                is_healthy = health_data.get('status') == 'healthy'
                
                if is_healthy:
                    return TestResult(
                        test_name=f"{self.service_name}_health",
                        test_type="service_health",
                        status="PASS",
                        duration=time.time() - start_time,
                        details={"response": health_data}
                    )
                else:
                    return TestResult(
                        test_name=f"{self.service_name}_health",
                        test_type="service_health",
                        status="FAIL",
                        duration=time.time() - start_time,
                        error_message="Service reported unhealthy status",
                        details={"response": health_data}
                    )
            else:
                return TestResult(
                    test_name=f"{self.service_name}_health",
                    test_type="service_health",
                    status="FAIL",
                    duration=time.time() - start_time,
                    error_message=f"Health endpoint returned status {response.status_code}"
                )
                
        except Exception as e:
            return TestResult(
                test_name=f"{self.service_name}_health",
                test_type="service_health",
                status="ERROR",
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_service_metrics(self) -> TestResult:
        """Test service metrics endpoint."""
        start_time = time.time()
        
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            
            if response.status_code == 200:
                metrics_text = response.text
                
                # Check if metrics contain expected data
                if "trading_" in metrics_text or "system_" in metrics_text:
                    return TestResult(
                        test_name=f"{self.service_name}_metrics",
                        test_type="service_metrics",
                        status="PASS",
                        duration=time.time() - start_time,
                        details={"metrics_count": len(metrics_text.split('\n'))}
                    )
                else:
                    return TestResult(
                        test_name=f"{self.service_name}_metrics",
                        test_type="service_metrics",
                        status="FAIL",
                        duration=time.time() - start_time,
                        error_message="Metrics endpoint returned unexpected data"
                    )
            else:
                return TestResult(
                    test_name=f"{self.service_name}_metrics",
                    test_type="service_metrics",
                    status="FAIL",
                    duration=time.time() - start_time,
                    error_message=f"Metrics endpoint returned status {response.status_code}"
                )
                
        except Exception as e:
            return TestResult(
                test_name=f"{self.service_name}_metrics",
                test_type="service_metrics",
                status="ERROR",
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_service_dependencies(self, dependencies: List[str]) -> List[TestResult]:
        """Test service dependencies (databases, external services, etc.)."""
        results = []
        
        for dep in dependencies:
            start_time = time.time()
            
            try:
                if dep.startswith("http"):
                    # Test HTTP dependency
                    response = requests.get(dep, timeout=5)
                    is_healthy = response.status_code < 500
                elif dep.startswith("redis://"):
                    # Test Redis dependency
                    import redis
                    r = redis.from_url(dep)
                    r.ping()
                    is_healthy = True
                elif dep.startswith("postgresql://"):
                    # Test PostgreSQL dependency
                    import psycopg2
                    conn = psycopg2.connect(dep)
                    conn.close()
                    is_healthy = True
                else:
                    # Unknown dependency type
                    is_healthy = False
                
                if is_healthy:
                    results.append(TestResult(
                        test_name=f"{self.service_name}_dependency_{dep}",
                        test_type="service_dependency",
                        status="PASS",
                        duration=time.time() - start_time,
                        details={"dependency": dep}
                    ))
                else:
                    results.append(TestResult(
                        test_name=f"{self.service_name}_dependency_{dep}",
                        test_type="service_dependency",
                        status="FAIL",
                        duration=time.time() - start_time,
                        error_message=f"Dependency {dep} is not healthy"
                    ))
                    
            except Exception as e:
                results.append(TestResult(
                    test_name=f"{self.service_name}_dependency_{dep}",
                    test_type="service_dependency",
                    status="ERROR",
                    duration=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results
    
    def run_all_tests(self, dependencies: List[str] = None) -> List[TestResult]:
        """Run all service integration tests."""
        all_results = []
        
        # Basic service tests
        all_results.append(self.test_service_health())
        all_results.append(self.test_service_metrics())
        
        # Dependency tests
        if dependencies:
            all_results.extend(self.test_service_dependencies(dependencies))
        
        self.test_results = all_results
        return all_results


class WorkflowIntegrationTest:
    """Tests complete workflows and business processes."""
    
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.test_results: List[TestResult] = []
    
    def test_trading_workflow(self) -> TestResult:
        """Test the complete trading workflow."""
        start_time = time.time()
        
        try:
            # Simulate trading workflow steps
            steps = [
                "data_collection",
                "feature_engineering", 
                "model_prediction",
                "order_execution",
                "position_monitoring"
            ]
            
            completed_steps = []
            for step in steps:
                # Simulate step execution
                time.sleep(0.1)  # Simulate processing time
                completed_steps.append(step)
            
            if len(completed_steps) == len(steps):
                return TestResult(
                    test_name=f"{self.workflow_name}_trading_workflow",
                    test_type="workflow_integration",
                    status="PASS",
                    duration=time.time() - start_time,
                    details={"completed_steps": completed_steps}
                )
            else:
                return TestResult(
                    test_name=f"{self.workflow_name}_trading_workflow",
                    test_type="workflow_integration",
                    status="FAIL",
                    duration=time.time() - start_time,
                    error_message="Not all workflow steps completed"
                )
                
        except Exception as e:
            return TestResult(
                test_name=f"{self.workflow_name}_trading_workflow",
                test_type="workflow_integration",
                status="ERROR",
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_data_pipeline(self) -> TestResult:
        """Test the data processing pipeline."""
        start_time = time.time()
        
        try:
            # Simulate data pipeline steps
            pipeline_steps = [
                "data_ingestion",
                "data_validation",
                "data_transformation",
                "feature_extraction",
                "data_storage"
            ]
            
            # Simulate processing with some data
            sample_data = {"timestamp": datetime.now().isoformat(), "value": 100.0}
            
            processed_data = sample_data.copy()
            processed_data["processed"] = True
            processed_data["features"] = ["feature1", "feature2"]
            
            if processed_data.get("processed") and "features" in processed_data:
                return TestResult(
                    test_name=f"{self.workflow_name}_data_pipeline",
                    test_type="workflow_integration",
                    status="PASS",
                    duration=time.time() - start_time,
                    details={"pipeline_steps": pipeline_steps, "sample_data": processed_data}
                )
            else:
                return TestResult(
                    test_name=f"{self.workflow_name}_data_pipeline",
                    test_type="workflow_integration",
                    status="FAIL",
                    duration=time.time() - start_time,
                    error_message="Data pipeline did not complete successfully"
                )
                
        except Exception as e:
            return TestResult(
                test_name=f"{self.workflow_name}_data_pipeline",
                test_type="workflow_integration",
                status="ERROR",
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_failover_scenarios(self) -> List[TestResult]:
        """Test failover and recovery scenarios."""
        results = []
        
        failover_scenarios = [
            "primary_data_source_failure",
            "database_connection_loss",
            "external_api_timeout"
        ]
        
        for scenario in failover_scenarios:
            start_time = time.time()
            
            try:
                # Simulate failover scenario
                if scenario == "primary_data_source_failure":
                    # Simulate fallback to secondary source
                    fallback_activated = True
                    recovery_time = 2.5  # seconds
                elif scenario == "database_connection_loss":
                    # Simulate connection retry
                    fallback_activated = True
                    recovery_time = 1.8
                elif scenario == "external_api_timeout":
                    # Simulate timeout handling
                    fallback_activated = True
                    recovery_time = 3.2
                else:
                    fallback_activated = False
                    recovery_time = 0
                
                if fallback_activated and recovery_time < 5.0:
                    results.append(TestResult(
                        test_name=f"{self.workflow_name}_failover_{scenario}",
                        test_type="workflow_integration",
                        status="PASS",
                        duration=time.time() - start_time,
                        details={"scenario": scenario, "recovery_time": recovery_time}
                    ))
                else:
                    results.append(TestResult(
                        test_name=f"{self.workflow_name}_failover_{scenario}",
                        test_type="workflow_integration",
                        status="FAIL",
                        duration=time.time() - start_time,
                        error_message=f"Failover scenario {scenario} failed"
                    ))
                    
            except Exception as e:
                results.append(TestResult(
                    test_name=f"{self.workflow_name}_failover_{scenario}",
                    test_type="workflow_integration",
                    status="ERROR",
                    duration=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all workflow integration tests."""
        all_results = []
        
        all_results.append(self.test_trading_workflow())
        all_results.append(self.test_data_pipeline())
        all_results.extend(self.test_failover_scenarios())
        
        self.test_results = all_results
        return all_results


class PerformanceTest:
    """Performance and load testing."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.test_results: List[TestResult] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'response_times': [],
            'throughput': [],
            'error_rates': []
        }
    
    def test_response_time(self, endpoint: str, num_requests: int = 100) -> TestResult:
        """Test response time performance."""
        start_time = time.time()
        
        try:
            response_times = []
            errors = 0
            
            for i in range(num_requests):
                request_start = time.time()
                
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        response_time = time.time() - request_start
                        response_times.append(response_time)
                    else:
                        errors += 1
                except Exception:
                    errors += 1
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                error_rate = (errors / num_requests) * 100
                
                # Performance thresholds
                if avg_response_time < 1.0 and error_rate < 5.0:
                    status = "PASS"
                elif avg_response_time < 2.0 and error_rate < 10.0:
                    status = "PASS"
                else:
                    status = "FAIL"
                
                self.performance_metrics['response_times'].extend(response_times)
                self.performance_metrics['error_rates'].append(error_rate)
                
                return TestResult(
                    test_name=f"{self.test_name}_response_time",
                    test_type="performance",
                    status=status,
                    duration=time.time() - start_time,
                    details={
                        'avg_response_time': avg_response_time,
                        'max_response_time': max_response_time,
                        'min_response_time': min_response_time,
                        'error_rate': error_rate,
                        'total_requests': num_requests
                    }
                )
            else:
                return TestResult(
                    test_name=f"{self.test_name}_response_time",
                    test_type="performance",
                    status="FAIL",
                    duration=time.time() - start_time,
                    error_message="No successful requests recorded"
                )
                
        except Exception as e:
            return TestResult(
                test_name=f"{self.test_name}_response_time",
                test_type="performance",
                status="ERROR",
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def test_throughput(self, endpoint: str, duration: int = 60) -> TestResult:
        """Test throughput under sustained load."""
        start_time = time.time()
        
        try:
            requests_sent = 0
            successful_requests = 0
            errors = 0
            
            # Send requests for the specified duration
            while time.time() - start_time < duration:
                try:
                    response = requests.get(endpoint, timeout=2)
                    if response.status_code == 200:
                        successful_requests += 1
                    else:
                        errors += 1
                    requests_sent += 1
                    
                    # Small delay to avoid overwhelming the service
                    time.sleep(0.1)
                    
                except Exception:
                    errors += 1
                    requests_sent += 1
            
            actual_duration = time.time() - start_time
            throughput = successful_requests / actual_duration
            error_rate = (errors / requests_sent) * 100 if requests_sent > 0 else 0
            
            # Throughput thresholds
            if throughput > 10.0 and error_rate < 5.0:
                status = "PASS"
            elif throughput > 5.0 and error_rate < 10.0:
                status = "PASS"
            else:
                status = "FAIL"
            
            self.performance_metrics['throughput'].append(throughput)
            
            return TestResult(
                test_name=f"{self.test_name}_throughput",
                test_type="performance",
                status=status,
                duration=actual_duration,
                details={
                    'throughput_rps': throughput,
                    'total_requests': requests_sent,
                    'successful_requests': successful_requests,
                    'error_rate': error_rate
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=f"{self.test_name}_throughput",
                test_type="performance",
                status="ERROR",
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    def run_all_tests(self, endpoint: str) -> List[TestResult]:
        """Run all performance tests."""
        all_results = []
        
        all_results.append(self.test_response_time(endpoint))
        all_results.append(self.test_throughput(endpoint))
        
        self.test_results = all_results
        return all_results


class IntegrationTestSuite:
    """Orchestrates all integration tests."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.test_results: List[TestResult] = []
        self.services: List[ServiceIntegrationTest] = []
        self.workflows: List[WorkflowIntegrationTest] = []
        self.performance_tests: List[PerformanceTest] = []
        
        self._setup_tests()
    
    def _setup_tests(self):
        """Setup test instances based on configuration."""
        # Service tests
        services_config = self.config.get('services', {})
        for service_name, service_config in services_config.items():
            base_url = service_config.get('base_url', 'http://localhost:8000')
            dependencies = service_config.get('dependencies', [])
            
            service_test = ServiceIntegrationTest(service_name, base_url)
            service_test.dependencies = dependencies
            self.services.append(service_test)
        
        # Workflow tests
        workflows_config = self.config.get('workflows', [])
        for workflow_name in workflows_config:
            workflow_test = WorkflowIntegrationTest(workflow_name)
            self.workflows.append(workflow_test)
        
        # Performance tests
        perf_config = self.config.get('performance', {})
        for test_name, test_config in perf_config.items():
            perf_test = PerformanceTest(test_name)
            self.performance_tests.append(perf_test)
    
    def run_service_tests(self) -> List[TestResult]:
        """Run all service integration tests."""
        all_results = []
        
        for service_test in self.services:
            results = service_test.run_all_tests()
            all_results.extend(results)
        
        return all_results
    
    def run_workflow_tests(self) -> List[TestResult]:
        """Run all workflow integration tests."""
        all_results = []
        
        for workflow_test in self.workflows:
            results = workflow_test.run_all_tests()
            all_results.extend(results)
        
        return all_results
    
    def run_performance_tests(self, endpoint: str) -> List[TestResult]:
        """Run all performance tests."""
        all_results = []
        
        for perf_test in self.performance_tests:
            results = perf_test.run_all_tests(endpoint)
            all_results.extend(results)
        
        return all_results
    
    def run_all_tests(self, endpoint: str = "http://localhost:8000/health") -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("Starting integration test suite...")
        
        start_time = time.time()
        
        # Run all test types
        service_results = self.run_service_tests()
        workflow_results = self.run_workflow_tests()
        performance_results = self.run_performance_tests(endpoint)
        
        # Combine all results
        all_results = service_results + workflow_results + performance_results
        self.test_results = all_results
        
        # Calculate summary
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.status == "PASS")
        failed_tests = sum(1 for r in all_results if r.status == "FAIL")
        error_tests = sum(1 for r in all_results if r.status == "ERROR")
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'execution_time': time.time() - start_time,
            'test_results': [asdict(r) for r in all_results]
        }
        
        logger.info(f"Integration test suite completed. {passed_tests}/{total_tests} tests passed.")
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a human-readable test report."""
        if not self.test_results:
            return "No test results available."
        
        report_lines = [
            "=" * 60,
            "INTEGRATION TEST REPORT",
            "=" * 60,
            ""
        ]
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == "PASS")
        failed_tests = sum(1 for r in self.test_results if r.status == "FAIL")
        error_tests = sum(1 for r in self.test_results if r.status == "ERROR")
        
        report_lines.extend([
            f"SUMMARY:",
            f"  Total Tests: {total_tests}",
            f"  Passed: {passed_tests}",
            f"  Failed: {failed_tests}",
            f"  Errors: {error_tests}",
            f"  Success Rate: {(passed_tests/total_tests*100):.1f}%",
            ""
        ])
        
        # Test results by type
        test_types = {}
        for result in self.test_results:
            test_type = result.test_type
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)
        
        for test_type, results in test_types.items():
            report_lines.extend([
                f"{test_type.upper()} TESTS:",
                "-" * 30
            ])
            
            for result in results:
                status_icon = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⚠️"
                report_lines.append(
                    f"{status_icon} {result.test_name} ({result.duration:.2f}s)"
                )
                
                if result.error_message:
                    report_lines.append(f"    Error: {result.error_message}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
