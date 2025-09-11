#!/usr/bin/env python3
"""
Practical Examples for USDCOP MLOps Bucket Management System
============================================================
This file contains complete working examples demonstrating how to use
the bucket management system in various scenarios
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.mlops.bucket_provisioner import MLOpsBucketProvisioner, BucketProvisionerError
from scripts.mlops.bucket_validator import BucketValidator, ValidationReport

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BucketManagementExamples:
    """
    Collection of practical examples for bucket management
    """
    
    def __init__(self):
        self.config_path = project_root / "config" / "minio-buckets.yaml"
        self.examples_output_dir = project_root / "examples" / "output"
        self.examples_output_dir.mkdir(exist_ok=True)
    
    def example_1_basic_provisioning(self):
        """
        Example 1: Basic bucket provisioning for development environment
        """
        print("\n" + "="*60)
        print("EXAMPLE 1: Basic Bucket Provisioning")
        print("="*60)
        
        try:
            # Initialize provisioner for development
            provisioner = MLOpsBucketProvisioner(
                config_path=str(self.config_path),
                environment="development"
            )
            
            print("🚀 Starting bucket provisioning for development environment...")
            
            # Provision all buckets
            results = provisioner.provision_all_buckets()
            
            print(f"✅ Provisioning completed!")
            print(f"   Created: {results['summary']['created']} buckets")
            print(f"   Updated: {results['summary']['updated']} buckets")
            print(f"   Skipped: {results['summary']['skipped']} buckets")
            print(f"   Failed:  {results['summary']['failed']} buckets")
            
            # Save detailed results
            output_file = self.examples_output_dir / "basic_provisioning_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"📄 Detailed results saved to: {output_file}")
            
            return results
            
        except BucketProvisionerError as e:
            print(f"❌ Provisioning failed: {e}")
            return None
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            return None
    
    def example_2_comprehensive_validation(self):
        """
        Example 2: Comprehensive bucket validation with detailed reporting
        """
        print("\n" + "="*60)
        print("EXAMPLE 2: Comprehensive Validation")
        print("="*60)
        
        try:
            # Initialize validator
            validator = BucketValidator(
                config_path=str(self.config_path),
                environment="production"
            )
            
            print("🔍 Running comprehensive bucket validation...")
            
            # Run validation
            report = validator.validate_all_buckets()
            
            print(f"📊 Validation Results:")
            print(f"   Total Buckets: {report.total_buckets}")
            print(f"   Healthy: {report.healthy_buckets}")
            print(f"   Degraded: {report.degraded_buckets}")
            print(f"   Unhealthy: {report.unhealthy_buckets}")
            print(f"   Overall Health Score: {report.overall_health_score:.2f}")
            
            # Display recommendations
            if report.recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(report.recommendations, 1):
                    print(f"   {i}. {rec}")
            
            # Export reports in different formats
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # JSON report
            json_file = self.examples_output_dir / f"validation_report_{timestamp}.json"
            validator.export_validation_report(report, str(json_file), "json")
            
            # HTML report
            html_file = self.examples_output_dir / f"validation_report_{timestamp}.html"
            validator.export_validation_report(report, str(html_file), "html")
            
            print(f"📄 Reports exported:")
            print(f"   JSON: {json_file}")
            print(f"   HTML: {html_file}")
            
            return report
            
        except Exception as e:
            print(f"💥 Validation failed: {e}")
            return None
    
    def example_3_monitoring_setup(self):
        """
        Example 3: Set up continuous monitoring of bucket health
        """
        print("\n" + "="*60)
        print("EXAMPLE 3: Continuous Monitoring Setup")
        print("="*60)
        
        try:
            validator = BucketValidator(
                config_path=str(self.config_path),
                environment="production"
            )
            
            print("📈 Starting 2-minute bucket health monitoring...")
            
            # Monitor for 2 minutes
            monitoring_results = validator.monitor_bucket_health(duration_minutes=2)
            
            print(f"📊 Monitoring Results:")
            print(f"   Duration: {monitoring_results['duration_minutes']} minutes")
            print(f"   Samples: {len(monitoring_results['samples'])}")
            
            if monitoring_results['trends']:
                trends = monitoring_results['trends']
                print(f"   Connectivity Rate: {trends.get('connectivity_rate', 0):.2%}")
                print(f"   Avg Response Time: {trends.get('avg_response_time_ms', 0):.2f}ms")
            
            if monitoring_results['alerts']:
                print(f"⚠️  Alerts: {len(monitoring_results['alerts'])}")
                for alert in monitoring_results['alerts']:
                    print(f"     {alert['severity'].upper()}: {alert['message']}")
            
            # Save monitoring results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            monitoring_file = self.examples_output_dir / f"monitoring_results_{timestamp}.json"
            with open(monitoring_file, 'w') as f:
                json.dump(monitoring_results, f, indent=2, default=str)
            
            print(f"📄 Monitoring results saved to: {monitoring_file}")
            
            return monitoring_results
            
        except Exception as e:
            print(f"💥 Monitoring failed: {e}")
            return None
    
    def example_4_custom_bucket_creation(self):
        """
        Example 4: Create custom buckets programmatically
        """
        print("\n" + "="*60)
        print("EXAMPLE 4: Custom Bucket Creation")
        print("="*60)
        
        try:
            # Create custom configuration
            custom_config = {
                "metadata": {
                    "version": "v1.0.0",
                    "description": "Custom bucket configuration",
                    "environment": "development"
                },
                "minio_config": {
                    "endpoint": "localhost:9000",
                    "access_key": "minioadmin",
                    "secret_key": "minioadmin123",
                    "secure": False,
                    "region": "us-east-1"
                },
                "bucket_groups": {
                    "experimental": {
                        "description": "Experimental data buckets",
                        "retention_days": 7,
                        "versioning": False,
                        "buckets": [
                            {
                                "name": "experiment-data-sandbox",
                                "description": "Sandbox for experimental data",
                                "tags": {
                                    "environment": "development",
                                    "purpose": "experimentation",
                                    "auto_created": "true"
                                }
                            },
                            {
                                "name": "temp-analysis-results",
                                "description": "Temporary analysis results",
                                "tags": {
                                    "environment": "development",
                                    "purpose": "temporary",
                                    "auto_cleanup": "true"
                                }
                            }
                        ]
                    }
                },
                "bucket_policies": {
                    "default_policy": {
                        "version": "2012-10-17",
                        "statements": [
                            {
                                "effect": "Allow",
                                "principal": {"AWS": "*"},
                                "actions": [
                                    "s3:GetBucketLocation",
                                    "s3:ListBucket",
                                    "s3:GetObject",
                                    "s3:PutObject",
                                    "s3:DeleteObject"
                                ],
                                "resources": [
                                    "arn:aws:s3:::{bucket_name}",
                                    "arn:aws:s3:::{bucket_name}/*"
                                ]
                            }
                        ]
                    }
                }
            }
            
            # Save custom configuration
            custom_config_file = self.examples_output_dir / "custom_buckets.yaml"
            with open(custom_config_file, 'w') as f:
                yaml.dump(custom_config, f, default_flow_style=False)
            
            print(f"📝 Custom configuration created: {custom_config_file}")
            
            # Use custom configuration with provisioner
            provisioner = MLOpsBucketProvisioner(
                config_path=str(custom_config_file),
                environment="development"
            )
            
            print("🚀 Provisioning custom buckets...")
            results = provisioner.provision_all_buckets()
            
            print(f"✅ Custom bucket provisioning completed!")
            print(f"   Created: {results['summary']['created']} buckets")
            
            # Validate custom buckets
            validator = BucketValidator(
                config_path=str(custom_config_file),
                environment="development"
            )
            
            validation_results = validator.validate_all_buckets()
            print(f"🔍 Validation: {validation_results.healthy_buckets}/{validation_results.total_buckets} buckets healthy")
            
            return results
            
        except Exception as e:
            print(f"💥 Custom bucket creation failed: {e}")
            return None
    
    def example_5_environment_migration(self):
        """
        Example 5: Migrate bucket configuration between environments
        """
        print("\n" + "="*60)
        print("EXAMPLE 5: Environment Migration")
        print("="*60)
        
        try:
            print("🔄 Simulating environment migration...")
            
            # Validate source environment (development)
            dev_validator = BucketValidator(
                config_path=str(self.config_path),
                environment="development"
            )
            
            print("1️⃣ Validating development environment...")
            dev_report = dev_validator.validate_all_buckets()
            print(f"   Development: {dev_report.healthy_buckets}/{dev_report.total_buckets} buckets healthy")
            
            # Create staging configuration
            staging_provisioner = MLOpsBucketProvisioner(
                config_path=str(self.config_path),
                environment="staging"
            )
            
            print("2️⃣ Setting up staging environment...")
            staging_results = staging_provisioner.provision_all_buckets()
            print(f"   Staging setup: {staging_results['summary']['created']} buckets created")
            
            # Validate staging environment
            staging_validator = BucketValidator(
                config_path=str(self.config_path),
                environment="staging"
            )
            
            print("3️⃣ Validating staging environment...")
            staging_report = staging_validator.validate_all_buckets()
            print(f"   Staging: {staging_report.healthy_buckets}/{staging_report.total_buckets} buckets healthy")
            
            # Generate migration report
            migration_report = {
                "migration_timestamp": datetime.now().isoformat(),
                "source_environment": "development",
                "target_environment": "staging",
                "source_health_score": dev_report.overall_health_score,
                "target_health_score": staging_report.overall_health_score,
                "migration_successful": staging_report.unhealthy_buckets == 0,
                "recommendations": staging_report.recommendations
            }
            
            migration_file = self.examples_output_dir / "migration_report.json"
            with open(migration_file, 'w') as f:
                json.dump(migration_report, f, indent=2)
            
            print(f"📄 Migration report saved: {migration_file}")
            print(f"✅ Migration completed successfully: {migration_report['migration_successful']}")
            
            return migration_report
            
        except Exception as e:
            print(f"💥 Environment migration failed: {e}")
            return None
    
    def example_6_integration_with_airflow(self):
        """
        Example 6: Integration example with Airflow DAGs
        """
        print("\n" + "="*60)
        print("EXAMPLE 6: Airflow Integration")
        print("="*60)
        
        # Create example Airflow DAG code
        airflow_integration_code = '''
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from scripts.mlops.bucket_validator import BucketValidator
from scripts.mlops.bucket_provisioner import MLOpsBucketProvisioner

# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 11),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Create DAG
dag = DAG(
    'bucket_health_monitoring',
    default_args=default_args,
    description='Monitor bucket health and ensure availability',
    schedule_interval='@hourly',
    catchup=False,
    tags=['mlops', 'infrastructure', 'monitoring']
)

def validate_bucket_health(**context):
    """Validate bucket health and raise alert if issues found"""
    validator = BucketValidator(
        config_path='/opt/airflow/config/minio-buckets.yaml',
        environment='production'
    )
    
    report = validator.validate_all_buckets()
    
    # Log results
    print(f"Bucket health check completed:")
    print(f"  Total buckets: {report.total_buckets}")
    print(f"  Healthy: {report.healthy_buckets}")
    print(f"  Unhealthy: {report.unhealthy_buckets}")
    print(f"  Health score: {report.overall_health_score:.2f}")
    
    # Store results in XCom
    context['task_instance'].xcom_push(
        key='health_report',
        value={
            'total_buckets': report.total_buckets,
            'healthy_buckets': report.healthy_buckets,
            'unhealthy_buckets': report.unhealthy_buckets,
            'health_score': report.overall_health_score,
            'recommendations': report.recommendations
        }
    )
    
    # Fail if unhealthy buckets found
    if report.unhealthy_buckets > 0:
        raise Exception(f"Found {report.unhealthy_buckets} unhealthy buckets!")
    
    return report.overall_health_score

def provision_missing_buckets(**context):
    """Provision any missing buckets"""
    provisioner = MLOpsBucketProvisioner(
        config_path='/opt/airflow/config/minio-buckets.yaml',
        environment='production'
    )
    
    results = provisioner.provision_all_buckets()
    
    print(f"Bucket provisioning completed:")
    print(f"  Created: {results['summary']['created']}")
    print(f"  Failed: {results['summary']['failed']}")
    
    return results['summary']

def send_health_report(**context):
    """Send health report via email or Slack"""
    health_report = context['task_instance'].xcom_pull(
        task_ids='validate_buckets',
        key='health_report'
    )
    
    if health_report:
        # Here you would integrate with your notification system
        print(f"Health report ready for notification:")
        print(f"  Health Score: {health_report['health_score']:.2f}")
        print(f"  Recommendations: {len(health_report['recommendations'])}")
    
    return "notification_sent"

# Define tasks
validate_task = PythonOperator(
    task_id='validate_buckets',
    python_callable=validate_bucket_health,
    dag=dag
)

provision_task = PythonOperator(
    task_id='provision_missing_buckets',
    python_callable=provision_missing_buckets,
    dag=dag
)

report_task = PythonOperator(
    task_id='send_health_report',
    python_callable=send_health_report,
    dag=dag
)

cleanup_task = BashOperator(
    task_id='cleanup_old_reports',
    bash_command='''
        # Clean up old validation reports (keep last 30 days)
        find /opt/airflow/reports -name "validation_report_*.json" -mtime +30 -delete
        find /opt/airflow/reports -name "monitoring_results_*.json" -mtime +7 -delete
        echo "Cleanup completed"
    ''',
    dag=dag
)

# Define task dependencies
validate_task >> provision_task >> report_task >> cleanup_task
        '''
        
        # Save the DAG example
        dag_file = self.examples_output_dir / "bucket_health_monitoring_dag.py"
        with open(dag_file, 'w') as f:
            f.write(airflow_integration_code)
        
        print(f"📝 Airflow DAG example created: {dag_file}")
        print("🔗 This DAG provides:")
        print("   • Hourly bucket health monitoring")
        print("   • Automatic provisioning of missing buckets")
        print("   • Health report notifications")
        print("   • Automated cleanup of old reports")
        print("")
        print("📋 To use this DAG:")
        print("   1. Copy the file to your Airflow DAGs directory")
        print("   2. Update the config path to match your setup")
        print("   3. Configure email/Slack notifications")
        print("   4. Enable the DAG in Airflow UI")
        
        return dag_file

def main():
    """Run all examples"""
    print("🚀 USDCOP MLOps Bucket Management Examples")
    print("=" * 80)
    
    examples = BucketManagementExamples()
    
    # Run all examples
    examples_to_run = [
        ("Basic Provisioning", examples.example_1_basic_provisioning),
        ("Comprehensive Validation", examples.example_2_comprehensive_validation),
        ("Monitoring Setup", examples.example_3_monitoring_setup),
        ("Custom Bucket Creation", examples.example_4_custom_bucket_creation),
        ("Environment Migration", examples.example_5_environment_migration),
        ("Airflow Integration", examples.example_6_integration_with_airflow),
    ]
    
    results = {}
    
    for example_name, example_func in examples_to_run:
        try:
            print(f"\n⏳ Running: {example_name}")
            result = example_func()
            results[example_name] = {
                "status": "success" if result is not None else "failed",
                "result": result
            }
        except Exception as e:
            print(f"❌ {example_name} failed: {e}")
            results[example_name] = {
                "status": "error",
                "error": str(e)
            }
    
    # Generate summary
    print("\n" + "="*80)
    print("📊 EXAMPLES SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    
    for example_name, result in results.items():
        status_emoji = "✅" if result["status"] == "success" else "❌"
        print(f"   {status_emoji} {example_name}: {result['status']}")
    
    # Save complete results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = examples.examples_output_dir / f"examples_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Complete results saved to: {summary_file}")
    print(f"📁 All example outputs in: {examples.examples_output_dir}")

if __name__ == "__main__":
    main()