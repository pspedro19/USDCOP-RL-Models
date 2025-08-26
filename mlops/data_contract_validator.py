"""
Data Contract Validator for MLOps Pipeline
===========================================
Validates data contracts between pipeline stages ensuring data quality and consistency.
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from datetime import datetime
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Severity(Enum):
    """Severity levels for validation violations"""
    CRITICAL = "critical"  # Pipeline must stop
    HIGH = "high"         # Requires immediate attention
    MEDIUM = "medium"     # Should be fixed soon
    LOW = "low"          # Minor issue
    WARNING = "warning"  # Informational

@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    rule_name: str
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

@dataclass
class ContractValidation:
    """Complete contract validation results"""
    layer: str
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    critical_failures: int
    results: List[ValidationResult]
    data_fingerprint: str
    validation_timestamp: str

class DataContractValidator:
    """Validates data contracts between pipeline layers"""
    
    def __init__(self, config_path: str = "mlops/config/master_pipeline.yml"):
        """Initialize validator with configuration"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.contracts = self.config.get('data_contracts', {})
        self.quality_gates = {}
        
    def _load_config(self) -> Dict:
        """Load MLOps configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_layer(self, 
                       data: pd.DataFrame, 
                       layer: str,
                       previous_layer_data: Optional[pd.DataFrame] = None) -> ContractValidation:
        """
        Validate data for a specific pipeline layer
        
        Args:
            data: DataFrame to validate
            layer: Layer name (l0_raw, l1_standardized, etc.)
            previous_layer_data: Optional data from previous layer for lineage validation
            
        Returns:
            ContractValidation object with results
        """
        logger.info(f"Validating data contract for layer: {layer}")
        
        if layer not in self.contracts:
            raise ValueError(f"No contract defined for layer: {layer}")
        
        contract = self.contracts[layer]
        results = []
        
        # 1. Schema Validation
        results.extend(self._validate_schema(data, contract))
        
        # 2. Quality Rules Validation
        results.extend(self._validate_quality_rules(data, contract))
        
        # 3. Data Type Validation
        results.extend(self._validate_data_types(data, contract))
        
        # 4. Constraint Validation
        results.extend(self._validate_constraints(data, contract))
        
        # 5. Lineage Validation (if previous layer provided)
        if previous_layer_data is not None:
            results.extend(self._validate_lineage(data, previous_layer_data, layer))
        
        # 6. Statistical Validation
        results.extend(self._validate_statistics(data, contract))
        
        # Calculate summary
        passed_checks = sum(1 for r in results if r.passed)
        failed_checks = len(results) - passed_checks
        critical_failures = sum(1 for r in results if not r.passed and r.severity == Severity.CRITICAL)
        
        # Generate data fingerprint
        data_fingerprint = self._generate_fingerprint(data)
        
        return ContractValidation(
            layer=layer,
            passed=critical_failures == 0,
            total_checks=len(results),
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            critical_failures=critical_failures,
            results=results,
            data_fingerprint=data_fingerprint,
            validation_timestamp=datetime.utcnow().isoformat()
        )
    
    def _validate_schema(self, data: pd.DataFrame, contract: Dict) -> List[ValidationResult]:
        """Validate data schema against contract"""
        results = []
        schema = contract.get('schema', {})
        
        # Check required columns
        required_columns = schema.get('required_columns', [])
        for col_spec in required_columns:
            col_name = col_spec['name']
            
            if col_name not in data.columns:
                results.append(ValidationResult(
                    passed=False,
                    rule_name=f"schema.required_column.{col_name}",
                    severity=Severity.CRITICAL,
                    message=f"Required column '{col_name}' is missing",
                    details={'missing_column': col_name}
                ))
            else:
                results.append(ValidationResult(
                    passed=True,
                    rule_name=f"schema.required_column.{col_name}",
                    severity=Severity.CRITICAL,
                    message=f"Required column '{col_name}' present"
                ))
        
        # Check for unexpected columns
        expected_cols = {col['name'] for col in required_columns}
        unexpected_cols = set(data.columns) - expected_cols
        
        if unexpected_cols and contract.get('strict_schema', False):
            results.append(ValidationResult(
                passed=False,
                rule_name="schema.unexpected_columns",
                severity=Severity.WARNING,
                message=f"Unexpected columns found: {unexpected_cols}",
                details={'unexpected_columns': list(unexpected_cols)}
            ))
        
        return results
    
    def _validate_data_types(self, data: pd.DataFrame, contract: Dict) -> List[ValidationResult]:
        """Validate data types match contract"""
        results = []
        schema = contract.get('schema', {})
        
        for col_spec in schema.get('required_columns', []):
            col_name = col_spec['name']
            expected_type = col_spec.get('type')
            
            if col_name in data.columns and expected_type:
                actual_type = str(data[col_name].dtype)
                
                # Type matching logic
                type_match = self._check_type_compatibility(actual_type, expected_type)
                
                if not type_match:
                    results.append(ValidationResult(
                        passed=False,
                        rule_name=f"dtype.{col_name}",
                        severity=Severity.HIGH,
                        message=f"Column '{col_name}' has type '{actual_type}', expected '{expected_type}'",
                        details={'column': col_name, 'actual': actual_type, 'expected': expected_type}
                    ))
                else:
                    results.append(ValidationResult(
                        passed=True,
                        rule_name=f"dtype.{col_name}",
                        severity=Severity.HIGH,
                        message=f"Column '{col_name}' type valid"
                    ))
        
        return results
    
    def _validate_constraints(self, data: pd.DataFrame, contract: Dict) -> List[ValidationResult]:
        """Validate data constraints (min, max, nullable, etc.)"""
        results = []
        schema = contract.get('schema', {})
        
        for col_spec in schema.get('required_columns', []):
            col_name = col_spec['name']
            
            if col_name not in data.columns:
                continue
            
            col_data = data[col_name]
            constraints = col_spec.get('constraints', {})
            
            # Nullable check
            if not col_spec.get('nullable', True):
                null_count = col_data.isnull().sum()
                if null_count > 0:
                    results.append(ValidationResult(
                        passed=False,
                        rule_name=f"constraint.nullable.{col_name}",
                        severity=Severity.HIGH,
                        message=f"Column '{col_name}' has {null_count} null values but nullable=false",
                        details={'null_count': int(null_count)}
                    ))
            
            # Min/Max constraints
            if pd.api.types.is_numeric_dtype(col_data):
                if 'min' in constraints:
                    min_val = col_data.min()
                    if min_val < constraints['min']:
                        results.append(ValidationResult(
                            passed=False,
                            rule_name=f"constraint.min.{col_name}",
                            severity=Severity.HIGH,
                            message=f"Column '{col_name}' min value {min_val} < {constraints['min']}",
                            details={'min_value': float(min_val), 'constraint': constraints['min']}
                        ))
                
                if 'max' in constraints:
                    max_val = col_data.max()
                    if max_val > constraints['max']:
                        results.append(ValidationResult(
                            passed=False,
                            rule_name=f"constraint.max.{col_name}",
                            severity=Severity.HIGH,
                            message=f"Column '{col_name}' max value {max_val} > {constraints['max']}",
                            details={'max_value': float(max_val), 'constraint': constraints['max']}
                        ))
        
        return results
    
    def _validate_quality_rules(self, data: pd.DataFrame, contract: Dict) -> List[ValidationResult]:
        """Validate custom quality rules"""
        results = []
        
        for rule_spec in contract.get('quality_rules', []):
            rule = rule_spec['rule']
            severity = Severity(rule_spec.get('severity', 'warning'))
            
            try:
                # Evaluate rule (safely)
                # In production, use a safe expression evaluator
                valid_rows = data.query(rule, engine='python')
                violation_rate = 1 - (len(valid_rows) / len(data))
                
                if violation_rate > 0:
                    results.append(ValidationResult(
                        passed=False,
                        rule_name=f"quality.{rule}",
                        severity=severity,
                        message=f"Quality rule '{rule}' violated by {violation_rate:.2%} of rows",
                        details={'violation_rate': violation_rate, 'rule': rule}
                    ))
                else:
                    results.append(ValidationResult(
                        passed=True,
                        rule_name=f"quality.{rule}",
                        severity=severity,
                        message=f"Quality rule '{rule}' passed"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    passed=False,
                    rule_name=f"quality.{rule}",
                    severity=Severity.WARNING,
                    message=f"Could not evaluate rule '{rule}': {e}",
                    details={'error': str(e)}
                ))
        
        return results
    
    def _validate_lineage(self, data: pd.DataFrame, previous_data: pd.DataFrame, layer: str) -> List[ValidationResult]:
        """Validate data lineage between layers"""
        results = []
        
        # Check row count changes
        row_diff = len(data) - len(previous_data)
        row_change_pct = abs(row_diff) / len(previous_data)
        
        if row_change_pct > 0.1:  # More than 10% change
            results.append(ValidationResult(
                passed=False,
                rule_name="lineage.row_count",
                severity=Severity.WARNING,
                message=f"Row count changed by {row_change_pct:.2%} from previous layer",
                details={'previous_rows': len(previous_data), 'current_rows': len(data)}
            ))
        
        # Check for data loss in key columns
        if 'inherits' in self.contracts.get(layer, {}):
            parent_layer = self.contracts[layer]['inherits']
            parent_contract = self.contracts.get(parent_layer, {})
            
            for col_spec in parent_contract.get('schema', {}).get('required_columns', []):
                col_name = col_spec['name']
                if col_name in previous_data.columns and col_name not in data.columns:
                    results.append(ValidationResult(
                        passed=False,
                        rule_name=f"lineage.column_loss.{col_name}",
                        severity=Severity.CRITICAL,
                        message=f"Column '{col_name}' from parent layer is missing",
                        details={'missing_column': col_name}
                    ))
        
        return results
    
    def _validate_statistics(self, data: pd.DataFrame, contract: Dict) -> List[ValidationResult]:
        """Validate statistical properties of data"""
        results = []
        
        # Check for data drift using simple statistics
        for col in data.select_dtypes(include=[np.number]).columns:
            stats = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'null_rate': data[col].isnull().mean()
            }
            
            # Check for extreme null rates
            if stats['null_rate'] > 0.5:
                results.append(ValidationResult(
                    passed=False,
                    rule_name=f"statistics.null_rate.{col}",
                    severity=Severity.HIGH,
                    message=f"Column '{col}' has {stats['null_rate']:.2%} null values",
                    details={'column': col, 'null_rate': stats['null_rate']}
                ))
            
            # Check for zero variance (constant column)
            if stats['std'] == 0:
                results.append(ValidationResult(
                    passed=False,
                    rule_name=f"statistics.zero_variance.{col}",
                    severity=Severity.WARNING,
                    message=f"Column '{col}' has zero variance (constant values)",
                    details={'column': col, 'value': stats['mean']}
                ))
        
        return results
    
    def _check_type_compatibility(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type"""
        type_mappings = {
            'float64': ['float', 'float64', 'double'],
            'float32': ['float32', 'float'],
            'int64': ['int', 'int64', 'integer'],
            'int32': ['int32', 'int'],
            'int16': ['int16'],
            'bool': ['bool', 'boolean'],
            'object': ['string', 'str', 'object'],
            'datetime64[ns]': ['datetime', 'datetime64[ns]', 'datetime64[ns, UTC]']
        }
        
        for actual_type, compatible_types in type_mappings.items():
            if actual_type in actual and any(t in expected for t in compatible_types):
                return True
        
        return False
    
    def _generate_fingerprint(self, data: pd.DataFrame) -> str:
        """Generate a fingerprint for the data"""
        fingerprint_data = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'sample_hash': hashlib.sha256(
                data.head(100).to_json().encode()
            ).hexdigest()
        }
        
        return hashlib.sha256(
            json.dumps(fingerprint_data, sort_keys=True).encode()
        ).hexdigest()
    
    def generate_validation_report(self, validation: ContractValidation) -> Dict[str, Any]:
        """Generate a comprehensive validation report"""
        report = {
            'layer': validation.layer,
            'status': 'PASSED' if validation.passed else 'FAILED',
            'timestamp': validation.validation_timestamp,
            'data_fingerprint': validation.data_fingerprint,
            'summary': {
                'total_checks': validation.total_checks,
                'passed': validation.passed_checks,
                'failed': validation.failed_checks,
                'critical_failures': validation.critical_failures
            },
            'violations_by_severity': {},
            'detailed_results': []
        }
        
        # Group violations by severity
        for severity in Severity:
            violations = [r for r in validation.results if not r.passed and r.severity == severity]
            if violations:
                report['violations_by_severity'][severity.value] = [
                    {
                        'rule': v.rule_name,
                        'message': v.message,
                        'details': v.details
                    }
                    for v in violations
                ]
        
        # Add all results
        report['detailed_results'] = [
            {
                'rule': r.rule_name,
                'passed': r.passed,
                'severity': r.severity.value,
                'message': r.message,
                'details': r.details
            }
            for r in validation.results
        ]
        
        return report
    
    def save_validation_report(self, validation: ContractValidation, output_path: str):
        """Save validation report to file"""
        report = self.generate_validation_report(validation)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to: {output_file}")
        
        return output_file


def validate_pipeline_data(data_path: str, layer: str, config_path: Optional[str] = None) -> bool:
    """
    Convenience function to validate pipeline data
    
    Args:
        data_path: Path to data file (CSV or Parquet)
        layer: Layer name to validate against
        config_path: Optional path to configuration file
        
    Returns:
        True if validation passed, False otherwise
    """
    # Load data
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Create validator
    validator = DataContractValidator(config_path) if config_path else DataContractValidator()
    
    # Validate
    validation = validator.validate_layer(data, layer)
    
    # Generate report
    report_path = f"validation_reports/{layer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    validator.save_validation_report(validation, report_path)
    
    # Print summary
    print(f"\nValidation Summary for {layer}:")
    print(f"Status: {'✅ PASSED' if validation.passed else '❌ FAILED'}")
    print(f"Total Checks: {validation.total_checks}")
    print(f"Passed: {validation.passed_checks}")
    print(f"Failed: {validation.failed_checks}")
    print(f"Critical Failures: {validation.critical_failures}")
    print(f"Report saved to: {report_path}")
    
    return validation.passed


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python data_contract_validator.py <data_file> <layer>")
        print("Example: python data_contract_validator.py data.parquet l1_standardized")
        sys.exit(1)
    
    data_file = sys.argv[1]
    layer = sys.argv[2]
    
    passed = validate_pipeline_data(data_file, layer)
    sys.exit(0 if passed else 1)