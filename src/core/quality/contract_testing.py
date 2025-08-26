"""
Contract Testing Module
======================
Validates contracts between services using Pydantic schemas and OpenAPI specifications.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

import pydantic
from pydantic import BaseModel, ValidationError
import yaml
import requests
from openapi_spec_validator import validate_spec
from openapi_spec_validator.exceptions import OpenAPIError

logger = logging.getLogger(__name__)


@dataclass
class ContractValidationResult:
    """Result of a contract validation."""
    service_name: str
    contract_type: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validation_time: datetime
    details: Dict[str, Any]


class SchemaValidator:
    """Validates Pydantic schemas and data contracts."""
    
    def __init__(self):
        self.validated_schemas: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
    
    def validate_schema(self, schema_class: type, data: Dict[str, Any]) -> bool:
        """Validate data against a Pydantic schema."""
        try:
            if not issubclass(schema_class, BaseModel):
                raise ValueError(f"{schema_class} is not a Pydantic BaseModel")
            
            # Validate the data
            validated_data = schema_class(**data)
            self.validated_schemas[schema_class.__name__] = validated_data
            return True
            
        except ValidationError as e:
            errors = [f"Field '{err['loc'][0]}': {err['msg']}" for err in e.errors()]
            self.validation_errors.extend(errors)
            logger.error(f"Schema validation failed for {schema_class.__name__}: {errors}")
            return False
            
        except Exception as e:
            error_msg = f"Unexpected error validating {schema_class.__name__}: {str(e)}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
            return False
    
    def validate_schema_file(self, schema_file: Path) -> bool:
        """Validate a schema definition file."""
        try:
            if not schema_file.exists():
                raise FileNotFoundError(f"Schema file not found: {schema_file}")
            
            # Load and validate the schema file
            with open(schema_file, 'r') as f:
                if schema_file.suffix == '.json':
                    schema_data = json.load(f)
                elif schema_file.suffix in ['.yml', '.yaml']:
                    schema_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported schema file format: {schema_file.suffix}")
            
            # Basic schema structure validation
            required_fields = ['type', 'properties']
            if not all(field in schema_data for field in required_fields):
                raise ValueError(f"Schema missing required fields: {required_fields}")
            
            return True
            
        except Exception as e:
            error_msg = f"Schema file validation failed: {str(e)}"
            self.validation_errors.append(error_msg)
            logger.error(error_msg)
            return False
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'validated_schemas': len(self.validated_schemas),
            'validation_errors': len(self.validation_errors),
            'errors': self.validation_errors,
            'schema_names': list(self.validated_schemas.keys())
        }


class APIValidator:
    """Validates API contracts and OpenAPI specifications."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_specs: Dict[str, Any] = {}
        self.validation_results: List[ContractValidationResult] = []
    
    def validate_openapi_spec(self, spec_url: str) -> bool:
        """Validate an OpenAPI specification."""
        try:
            response = requests.get(spec_url, timeout=10)
            response.raise_for_status()
            
            spec_data = response.json()
            
            # Validate OpenAPI spec
            validate_spec(spec_data)
            
            # Store the validated spec
            self.api_specs[spec_url] = spec_data
            
            logger.info(f"OpenAPI spec validation successful: {spec_url}")
            return True
            
        except OpenAPIError as e:
            error_msg = f"OpenAPI spec validation failed: {str(e)}"
            logger.error(error_msg)
            self._record_validation_result(spec_url, "openapi", False, [error_msg])
            return False
            
        except Exception as e:
            error_msg = f"Error validating OpenAPI spec: {str(e)}"
            logger.error(error_msg)
            self._record_validation_result(spec_url, "openapi", False, [error_msg])
            return False
    
    def validate_api_endpoint(self, endpoint: str, expected_schema: Dict[str, Any]) -> bool:
        """Validate an API endpoint against expected schema."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, timeout=10)
            
            # Check response status
            if response.status_code != 200:
                error_msg = f"Endpoint {endpoint} returned status {response.status_code}"
                self._record_validation_result(endpoint, "api_endpoint", False, [error_msg])
                return False
            
            # Validate response schema
            response_data = response.json()
            is_valid = self._validate_response_schema(response_data, expected_schema)
            
            self._record_validation_result(
                endpoint, "api_endpoint", is_valid, 
                [] if is_valid else ["Schema validation failed"]
            )
            
            return is_valid
            
        except Exception as e:
            error_msg = f"Error validating API endpoint {endpoint}: {str(e)}"
            logger.error(error_msg)
            self._record_validation_result(endpoint, "api_endpoint", False, [error_msg])
            return False
    
    def _validate_response_schema(self, data: Any, schema: Dict[str, Any]) -> bool:
        """Validate response data against a schema."""
        try:
            # Simple schema validation - can be enhanced with more sophisticated validation
            if schema.get('type') == 'object':
                if not isinstance(data, dict):
                    return False
                
                required_fields = schema.get('required', [])
                if not all(field in data for field in required_fields):
                    return False
                
                # Check property types
                properties = schema.get('properties', {})
                for field, field_schema in properties.items():
                    if field in data:
                        if not self._validate_field_type(data[field], field_schema):
                            return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_field_type(self, value: Any, field_schema: Dict[str, Any]) -> bool:
        """Validate a single field against its schema."""
        expected_type = field_schema.get('type')
        
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        
        return True
    
    def _record_validation_result(self, service_name: str, contract_type: str, 
                                is_valid: bool, errors: List[str]):
        """Record a validation result."""
        result = ContractValidationResult(
            service_name=service_name,
            contract_type=contract_type,
            is_valid=is_valid,
            errors=errors,
            warnings=[],
            validation_time=datetime.now(),
            details={}
        )
        self.validation_results.append(result)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of API validation results."""
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.is_valid)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'results': [asdict(r) for r in self.validation_results]
        }


class ContractValidator:
    """Main contract validation orchestrator."""
    
    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.api_validator = APIValidator()
        self.validation_history: List[ContractValidationResult] = []
    
    def validate_service_contracts(self, service_name: str, 
                                 schemas: List[type] = None,
                                 api_endpoints: List[Dict[str, Any]] = None) -> bool:
        """Validate all contracts for a service."""
        all_valid = True
        
        # Validate schemas
        if schemas:
            for schema in schemas:
                if not self.schema_validator.validate_schema(schema, {}):
                    all_valid = False
        
        # Validate API endpoints
        if api_endpoints:
            for endpoint_info in api_endpoints:
                endpoint = endpoint_info['endpoint']
                expected_schema = endpoint_info.get('schema', {})
                
                if not self.api_validator.validate_api_endpoint(endpoint, expected_schema):
                    all_valid = False
        
        # Record overall result
        result = ContractValidationResult(
            service_name=service_name,
            contract_type="service_contracts",
            is_valid=all_valid,
            errors=self.schema_validator.validation_errors,
            warnings=[],
            validation_time=datetime.now(),
            details={
                'schema_validation': self.schema_validator.get_validation_summary(),
                'api_validation': self.api_validator.get_validation_summary()
            }
        )
        
        self.validation_history.append(result)
        return all_valid
    
    def validate_data_contracts(self, data_contracts: List[Dict[str, Any]]) -> bool:
        """Validate data contracts between services."""
        all_valid = True
        
        for contract in data_contracts:
            try:
                # Validate contract structure
                required_fields = ['service_from', 'service_to', 'data_schema']
                if not all(field in contract for field in required_fields):
                    all_valid = False
                    continue
                
                # Validate data schema
                schema = contract['data_schema']
                if not self.schema_validator.validate_schema_file(Path(schema)):
                    all_valid = False
                
            except Exception as e:
                logger.error(f"Error validating data contract: {str(e)}")
                all_valid = False
        
        return all_valid
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        return {
            'validation_summary': {
                'total_services': len(self.validation_history),
                'valid_services': sum(1 for r in self.validation_history if r.is_valid),
                'invalid_services': sum(1 for r in self.validation_history if not r.is_valid),
                'overall_success_rate': (
                    sum(1 for r in self.validation_history if r.is_valid) / 
                    len(self.validation_history) * 100
                ) if self.validation_history else 0
            },
            'service_details': [asdict(r) for r in self.validation_history],
            'schema_validation': self.schema_validator.get_validation_summary(),
            'api_validation': self.api_validator.get_validation_summary(),
            'timestamp': datetime.now().isoformat()
        }


def validate_contract(service_name: str, contract_type: str, 
                     contract_data: Dict[str, Any]) -> bool:
    """Convenience function for contract validation."""
    validator = ContractValidator()
    
    if contract_type == "service_contracts":
        return validator.validate_service_contracts(service_name, **contract_data)
    elif contract_type == "data_contracts":
        return validator.validate_data_contracts(contract_data.get('contracts', []))
    else:
        raise ValueError(f"Unknown contract type: {contract_type}")


def generate_contract_report(validation_results: List[ContractValidationResult]) -> str:
    """Generate a human-readable contract validation report."""
    report_lines = [
        "=" * 60,
        "CONTRACT VALIDATION REPORT",
        "=" * 60,
        ""
    ]
    
    for result in validation_results:
        status = "✅ PASS" if result.is_valid else "❌ FAIL"
        report_lines.extend([
            f"Service: {result.service_name}",
            f"Type: {result.contract_type}",
            f"Status: {status}",
            f"Time: {result.validation_time.strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        if result.errors:
            report_lines.append("Errors:")
            for error in result.errors:
                report_lines.append(f"  - {error}")
        
        if result.warnings:
            report_lines.append("Warnings:")
            for warning in result.warnings:
                report_lines.append(f"  - {warning}")
        
        report_lines.append("-" * 40)
    
    return "\n".join(report_lines)
