"""
Enhanced Pipeline Orchestrator for USDCOP Trading System
=========================================================
Provides granular control over data pipeline execution with numbered steps,
dependency management, retry logic, and comprehensive monitoring.

Features:
- 7-step data cycle pipeline
- Step-by-step execution control
- Dependency resolution
- Retry and resume capabilities
- Real-time monitoring
- Pipeline state persistence
"""

import json
import logging
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

import pandas as pd
import numpy as np

# Internal imports
from ..config.unified_config import get_system_config
from ..database.db_integration import db_integration
from ..events.bus import event_bus, Event, EventType
from ..monitoring.health_checks import health_checker
from ..errors.handlers import error_handler, ErrorContext

logger = logging.getLogger(__name__)


class PipelineStepStatus(Enum):
    """Pipeline step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class PipelineStep:
    """Individual pipeline step definition"""
    id: int
    name: str
    description: str
    function: Callable
    dependencies: List[int] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 300  # seconds
    required: bool = True
    parallel: bool = False
    
    # Runtime state
    status: PipelineStepStatus = PipelineStepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'attempts': self.attempts,
            'error': self.error
        }


@dataclass
class PipelineRun:
    """Pipeline execution run metadata"""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    results: Dict[int, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    

class DataPipelineOrchestrator:
    """
    Main orchestrator for the 7-step data pipeline with full control
    and monitoring capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the pipeline orchestrator"""
        self.config = config or get_system_config()
        self.steps: Dict[int, PipelineStep] = {}
        self.current_run: Optional[PipelineRun] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.state_file = Path("data/pipeline_state.json")
        
        # Initialize pipeline steps
        self._initialize_pipeline_steps()
        
    def _initialize_pipeline_steps(self):
        """Define the 7-step data pipeline"""
        
        # Import necessary modules
        from ...markets.usdcop.pipeline import USDCOPPipeline
        from ...markets.usdcop.feature_engine import FeatureEngine
        from ...markets.usdcop.environment import TradingEnvironment
        from ...markets.usdcop.agent import RLTradingAgent
        from ...markets.usdcop.backtester import Backtester
        
        # Create pipeline instance
        self.pipeline = USDCOPPipeline(self.config)
        self.feature_engine = FeatureEngine(self.config)
        
        # Define the 7 pipeline steps
        self.steps = {
            1: PipelineStep(
                id=1,
                name="Data Extraction",
                description="Extract raw data from MT5/CCXT/Simulation sources",
                function=self._step1_data_extraction,
                dependencies=[],
                retry_count=5,
                timeout=600,
                required=True
            ),
            
            2: PipelineStep(
                id=2,
                name="Data Validation",
                description="Validate data quality, check for gaps and anomalies",
                function=self._step2_data_validation,
                dependencies=[1],
                retry_count=3,
                timeout=300,
                required=True
            ),
            
            3: PipelineStep(
                id=3,
                name="Data Transformation",
                description="Clean, normalize, and transform data (Silver stage)",
                function=self._step3_data_transformation,
                dependencies=[2],
                retry_count=3,
                timeout=300,
                required=True
            ),
            
            4: PipelineStep(
                id=4,
                name="Feature Engineering",
                description="Generate technical indicators and features (Gold stage)",
                function=self._step4_feature_engineering,
                dependencies=[3],
                retry_count=2,
                timeout=600,
                required=True
            ),
            
            5: PipelineStep(
                id=5,
                name="Model Prediction",
                description="Run RL model predictions and generate signals",
                function=self._step5_model_prediction,
                dependencies=[4],
                retry_count=2,
                timeout=300,
                required=False  # Can skip if no model
            ),
            
            6: PipelineStep(
                id=6,
                name="Risk Analysis",
                description="Perform risk assessment and position sizing",
                function=self._step6_risk_analysis,
                dependencies=[5],
                retry_count=1,
                timeout=200,
                required=False
            ),
            
            7: PipelineStep(
                id=7,
                name="Decision Making",
                description="Make final trading decisions and execute orders",
                function=self._step7_decision_making,
                dependencies=[6],
                retry_count=1,
                timeout=100,
                required=False
            )
        }
    
    # -------------------------------------------------------------------------
    # Pipeline Steps Implementation
    # -------------------------------------------------------------------------
    
    def _step1_data_extraction(self, **kwargs) -> Dict[str, Any]:
        """Step 1: Extract raw data from sources"""
        logger.info("🔄 Pipeline Step 1: Data Extraction")
        
        start_date = kwargs.get('start_date', datetime.now() - timedelta(days=30))
        end_date = kwargs.get('end_date', datetime.now())
        symbol = kwargs.get('symbol', 'USDCOP')
        timeframe = kwargs.get('timeframe', 'M5')
        
        # Use existing pipeline's bronze stage
        result = self.pipeline.run_bronze(start_date, end_date)
        
        if result['success']:
            logger.info(f"✅ Extracted {result['bars_fetched']} bars")
            
            # Save to database
            db_integration.save_market_data(result['data'], symbol, timeframe)
            
            # Publish event
            if event_bus:
                event_bus.publish(Event(
                    event="pipeline.extraction.complete",
                    data={'bars': result['bars_fetched'], 'symbol': symbol}
                ))
            
            return {
                'success': True,
                'data': result['data'],
                'metadata': {
                    'bars_fetched': result['bars_fetched'],
                    'source': result['source'],
                    'duration': result['duration']
                }
            }
        else:
            raise Exception(f"Data extraction failed: {result.get('error')}")
    
    def _step2_data_validation(self, **kwargs) -> Dict[str, Any]:
        """Step 2: Validate data quality"""
        logger.info("🔍 Pipeline Step 2: Data Validation")
        
        data = kwargs.get('data')
        if data is None:
            data = self.get_step_result(1).get('data')
        
        validation_results = {
            'total_rows': len(data),
            'null_count': data.isnull().sum().sum(),
            'duplicate_count': data.duplicated().sum(),
            'gaps_detected': 0,
            'anomalies': []
        }
        
        # Check for time gaps
        time_diff = data.index.to_series().diff()
        expected_interval = pd.Timedelta(minutes=5)  # For M5 timeframe
        gaps = time_diff[time_diff > expected_interval * 1.5]
        validation_results['gaps_detected'] = len(gaps)
        
        # Check for price anomalies (>5% jumps)
        price_changes = data['close'].pct_change()
        anomalies = price_changes[abs(price_changes) > 0.05]
        validation_results['anomalies'] = anomalies.index.tolist()
        
        # Determine if data is valid
        is_valid = (
            validation_results['null_count'] < len(data) * 0.01 and  # <1% nulls
            validation_results['duplicate_count'] == 0 and
            validation_results['gaps_detected'] < 10  # Tolerate some gaps
        )
        
        logger.info(f"✅ Validation complete: {'PASSED' if is_valid else 'FAILED'}")
        logger.info(f"   - Nulls: {validation_results['null_count']}")
        logger.info(f"   - Gaps: {validation_results['gaps_detected']}")
        logger.info(f"   - Anomalies: {len(validation_results['anomalies'])}")
        
        return {
            'success': is_valid,
            'data': data,
            'validation': validation_results
        }
    
    def _step3_data_transformation(self, **kwargs) -> Dict[str, Any]:
        """Step 3: Transform and clean data (Silver stage)"""
        logger.info("🔧 Pipeline Step 3: Data Transformation")
        
        data = kwargs.get('data')
        if data is None:
            data = self.get_step_result(2).get('data')
        
        # Use pipeline's silver stage
        result = self.pipeline.run_silver(data)
        
        if result['success']:
            transformed_data = result['data']
            
            logger.info(f"✅ Transformation complete:")
            logger.info(f"   - Rows processed: {len(transformed_data)}")
            logger.info(f"   - Gaps filled: {result.get('gaps_filled', 0)}")
            logger.info(f"   - Outliers handled: {result.get('outliers_handled', 0)}")
            
            # Save transformed data
            db_integration.save_processed_data(
                transformed_data, 
                'USDCOP', 
                'silver',
                metadata={'transformation_time': datetime.now().isoformat()}
            )
            
            return {
                'success': True,
                'data': transformed_data,
                'metadata': result
            }
        else:
            raise Exception(f"Data transformation failed: {result.get('error')}")
    
    def _step4_feature_engineering(self, **kwargs) -> Dict[str, Any]:
        """Step 4: Generate features (Gold stage)"""
        logger.info("🎯 Pipeline Step 4: Feature Engineering")
        
        data = kwargs.get('data')
        if data is None:
            data = self.get_step_result(3).get('data')
        
        # Use pipeline's gold stage
        result = self.pipeline.run_gold(data)
        
        if result['success']:
            featured_data = result['data']
            
            logger.info(f"✅ Feature engineering complete:")
            logger.info(f"   - Features generated: {result.get('features_added', 0)}")
            logger.info(f"   - Final columns: {len(featured_data.columns)}")
            
            # Save featured data
            db_integration.save_features(
                featured_data,
                'USDCOP',
                metadata={'feature_version': '2.0'}
            )
            
            return {
                'success': True,
                'data': featured_data,
                'features': list(featured_data.columns),
                'metadata': result
            }
        else:
            raise Exception(f"Feature engineering failed: {result.get('error')}")
    
    def _step5_model_prediction(self, **kwargs) -> Dict[str, Any]:
        """Step 5: Generate model predictions"""
        logger.info("🤖 Pipeline Step 5: Model Prediction")
        
        data = kwargs.get('data')
        if data is None:
            data = self.get_step_result(4).get('data')
        
        try:
            # Load model if available
            model_path = Path(self.config.get('model_dir', 'models')) / 'best_model.zip'
            
            if not model_path.exists():
                logger.warning("No trained model found, skipping predictions")
                return {
                    'success': True,
                    'skipped': True,
                    'reason': 'No model available'
                }
            
            # Import and load model
            from ...markets.usdcop.agent import RLTradingAgent
            agent = RLTradingAgent.load(str(model_path))
            
            # Generate predictions
            predictions = agent.predict(data)
            
            # Generate signals
            signals = []
            for idx, pred in enumerate(predictions):
                if pred['confidence'] > 0.6:
                    signals.append({
                        'timestamp': data.index[idx],
                        'action': pred['action'],
                        'confidence': pred['confidence'],
                        'predicted_return': pred.get('expected_return', 0)
                    })
            
            logger.info(f"✅ Predictions complete:")
            logger.info(f"   - Total predictions: {len(predictions)}")
            logger.info(f"   - High confidence signals: {len(signals)}")
            
            return {
                'success': True,
                'predictions': predictions,
                'signals': signals,
                'model_version': agent.version if hasattr(agent, 'version') else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _step6_risk_analysis(self, **kwargs) -> Dict[str, Any]:
        """Step 6: Perform risk analysis"""
        logger.info("⚠️ Pipeline Step 6: Risk Analysis")
        
        signals = kwargs.get('signals')
        if signals is None:
            prev_result = self.get_step_result(5)
            if prev_result and not prev_result.get('skipped'):
                signals = prev_result.get('signals', [])
            else:
                return {'success': True, 'skipped': True, 'reason': 'No signals to analyze'}
        
        try:
            from ...core.risk_manager import RiskManager
            risk_manager = RiskManager(self.config.get('risk', {}))
            
            risk_results = []
            for signal in signals:
                risk_assessment = risk_manager.assess_signal(signal)
                risk_results.append({
                    'signal': signal,
                    'risk_score': risk_assessment['risk_score'],
                    'position_size': risk_assessment['position_size'],
                    'stop_loss': risk_assessment['stop_loss'],
                    'take_profit': risk_assessment['take_profit'],
                    'approved': risk_assessment['approved']
                })
            
            approved_signals = [r for r in risk_results if r['approved']]
            
            logger.info(f"✅ Risk analysis complete:")
            logger.info(f"   - Signals analyzed: {len(signals)}")
            logger.info(f"   - Approved signals: {len(approved_signals)}")
            logger.info(f"   - Average risk score: {np.mean([r['risk_score'] for r in risk_results]):.2f}")
            
            return {
                'success': True,
                'risk_results': risk_results,
                'approved_signals': approved_signals
            }
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _step7_decision_making(self, **kwargs) -> Dict[str, Any]:
        """Step 7: Make final trading decisions"""
        logger.info("💰 Pipeline Step 7: Decision Making")
        
        approved_signals = kwargs.get('approved_signals')
        if approved_signals is None:
            prev_result = self.get_step_result(6)
            if prev_result and not prev_result.get('skipped'):
                approved_signals = prev_result.get('approved_signals', [])
            else:
                return {'success': True, 'skipped': True, 'reason': 'No approved signals'}
        
        try:
            decisions = []
            for signal_data in approved_signals:
                decision = {
                    'timestamp': datetime.now(),
                    'signal': signal_data['signal'],
                    'position_size': signal_data['position_size'],
                    'stop_loss': signal_data['stop_loss'],
                    'take_profit': signal_data['take_profit'],
                    'action': 'EXECUTE' if self.config.get('live_trading', False) else 'SIMULATE',
                    'status': 'pending'
                }
                decisions.append(decision)
                
                # Save decision to database
                db_integration.save_trading_decision(decision)
                
                # Publish event
                if event_bus:
                    event_bus.publish(Event(
                        event="pipeline.decision.made",
                        data=decision
                    ))
            
            logger.info(f"✅ Decision making complete:")
            logger.info(f"   - Decisions made: {len(decisions)}")
            logger.info(f"   - Mode: {'LIVE' if self.config.get('live_trading', False) else 'SIMULATION'}")
            
            return {
                'success': True,
                'decisions': decisions,
                'execution_mode': 'live' if self.config.get('live_trading', False) else 'simulation'
            }
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # -------------------------------------------------------------------------
    # Pipeline Execution Control
    # -------------------------------------------------------------------------
    
    def run_pipeline(self, 
                    steps: Optional[List[int]] = None,
                    config: Optional[Dict[str, Any]] = None,
                    resume_from: Optional[int] = None) -> PipelineRun:
        """
        Execute the pipeline with specified steps.
        
        Args:
            steps: List of step IDs to execute (default: all steps)
            config: Runtime configuration overrides
            resume_from: Resume from specific step (useful for retries)
            
        Returns:
            PipelineRun object with results
        """
        # Create new run
        run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self.current_run = PipelineRun(
            run_id=run_id,
            start_time=datetime.now(),
            config=config or {}
        )
        
        # Determine steps to execute
        if steps is None:
            steps = list(range(1, 8))
        
        if resume_from:
            steps = [s for s in steps if s >= resume_from]
        
        self.current_run.total_steps = len(steps)
        
        logger.info(f"🚀 Starting pipeline run: {run_id}")
        logger.info(f"   Steps to execute: {steps}")
        
        try:
            # Execute steps in order
            for step_id in steps:
                if step_id not in self.steps:
                    logger.warning(f"Step {step_id} not found, skipping")
                    continue
                
                step = self.steps[step_id]
                
                # Check dependencies
                if not self._check_dependencies(step):
                    logger.warning(f"Dependencies not met for step {step_id}, skipping")
                    step.status = PipelineStepStatus.SKIPPED
                    self.current_run.skipped_steps += 1
                    continue
                
                # Execute step
                success = self._execute_step(step, config)
                
                if success:
                    self.current_run.completed_steps += 1
                elif step.required:
                    # Required step failed, stop pipeline
                    logger.error(f"Required step {step_id} failed, stopping pipeline")
                    self.current_run.status = "failed"
                    break
                else:
                    # Optional step failed, continue
                    logger.warning(f"Optional step {step_id} failed, continuing")
                    self.current_run.failed_steps += 1
            
            # Finalize run
            self.current_run.end_time = datetime.now()
            if self.current_run.status == "running":
                self.current_run.status = "completed"
            
            # Save state
            self._save_state()
            
            # Generate report
            self._generate_report()
            
            logger.info(f"✅ Pipeline run completed: {run_id}")
            logger.info(f"   Status: {self.current_run.status}")
            logger.info(f"   Completed: {self.current_run.completed_steps}/{self.current_run.total_steps}")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.current_run.status = "error"
            self.current_run.errors.append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            })
        
        return self.current_run
    
    def run_step(self, step_id: int, config: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a single pipeline step"""
        if step_id not in self.steps:
            logger.error(f"Step {step_id} not found")
            return False
        
        step = self.steps[step_id]
        return self._execute_step(step, config)
    
    def _execute_step(self, step: PipelineStep, config: Optional[Dict[str, Any]] = None) -> bool:
        """Execute a single step with retry logic"""
        logger.info(f"Executing step {step.id}: {step.name}")
        
        step.status = PipelineStepStatus.RUNNING
        step.start_time = datetime.now()
        step.attempts = 0
        
        while step.attempts < step.retry_count:
            step.attempts += 1
            
            try:
                # Prepare kwargs
                kwargs = config or {}
                
                # Add results from dependencies
                for dep_id in step.dependencies:
                    if dep_id in self.steps and self.steps[dep_id].result:
                        kwargs[f'step{dep_id}_result'] = self.steps[dep_id].result
                
                # Execute with timeout
                result = self._execute_with_timeout(step.function, kwargs, step.timeout)
                
                # Store result
                step.result = result
                step.status = PipelineStepStatus.COMPLETED
                step.end_time = datetime.now()
                
                if self.current_run:
                    self.current_run.results[step.id] = result
                
                # Publish success event
                if event_bus:
                    event_bus.publish(Event(
                        event=f"pipeline.step.{step.id}.complete",
                        data={'step': step.name, 'duration': (step.end_time - step.start_time).total_seconds()}
                    ))
                
                return True
                
            except Exception as e:
                logger.error(f"Step {step.id} attempt {step.attempts} failed: {e}")
                step.error = str(e)
                
                if step.attempts < step.retry_count:
                    step.status = PipelineStepStatus.RETRYING
                    time.sleep(min(2 ** step.attempts, 30))  # Exponential backoff
                else:
                    step.status = PipelineStepStatus.FAILED
                    step.end_time = datetime.now()
                    
                    if self.current_run:
                        self.current_run.errors.append({
                            'step_id': step.id,
                            'error': str(e),
                            'attempts': step.attempts
                        })
                    
                    # Publish failure event
                    if event_bus:
                        event_bus.publish(Event(
                            event=f"pipeline.step.{step.id}.failed",
                            data={'step': step.name, 'error': str(e)}
                        ))
                    
                    return False
        
        return False
    
    def _execute_with_timeout(self, func: Callable, kwargs: Dict, timeout: int) -> Any:
        """Execute function with timeout"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Step execution exceeded {timeout} seconds")
    
    def _check_dependencies(self, step: PipelineStep) -> bool:
        """Check if step dependencies are satisfied"""
        for dep_id in step.dependencies:
            if dep_id in self.steps:
                dep_step = self.steps[dep_id]
                if dep_step.status != PipelineStepStatus.COMPLETED:
                    return False
        return True
    
    def get_step_result(self, step_id: int) -> Optional[Any]:
        """Get the result of a completed step"""
        if step_id in self.steps:
            return self.steps[step_id].result
        return None
    
    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------
    
    def _save_state(self):
        """Save pipeline state to disk"""
        try:
            state = {
                'run_id': self.current_run.run_id if self.current_run else None,
                'timestamp': datetime.now().isoformat(),
                'steps': {
                    step_id: step.to_dict() 
                    for step_id, step in self.steps.items()
                },
                'current_run': asdict(self.current_run) if self.current_run else None
            }
            
            self.state_file.parent.mkdir(exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.debug(f"Pipeline state saved to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline state: {e}")
    
    def load_state(self) -> bool:
        """Load previous pipeline state"""
        try:
            if not self.state_file.exists():
                return False
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore step states
            for step_id, step_data in state.get('steps', {}).items():
                if int(step_id) in self.steps:
                    step = self.steps[int(step_id)]
                    step.status = PipelineStepStatus(step_data['status'])
                    step.attempts = step_data.get('attempts', 0)
                    step.error = step_data.get('error')
            
            logger.info(f"Pipeline state loaded from {state['timestamp']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load pipeline state: {e}")
            return False
    
    def reset_pipeline(self):
        """Reset all pipeline steps to initial state"""
        for step in self.steps.values():
            step.status = PipelineStepStatus.PENDING
            step.start_time = None
            step.end_time = None
            step.result = None
            step.error = None
            step.attempts = 0
        
        self.current_run = None
        logger.info("Pipeline reset to initial state")
    
    # -------------------------------------------------------------------------
    # Monitoring and Reporting
    # -------------------------------------------------------------------------
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'current_run': self.current_run.run_id if self.current_run else None,
            'status': self.current_run.status if self.current_run else 'idle',
            'steps': {
                step_id: {
                    'name': step.name,
                    'status': step.status.value,
                    'attempts': step.attempts,
                    'error': step.error
                }
                for step_id, step in self.steps.items()
            }
        }
    
    def _generate_report(self):
        """Generate pipeline execution report"""
        if not self.current_run:
            return
        
        report = {
            'run_id': self.current_run.run_id,
            'status': self.current_run.status,
            'duration': (self.current_run.end_time - self.current_run.start_time).total_seconds() if self.current_run.end_time else None,
            'summary': {
                'total_steps': self.current_run.total_steps,
                'completed': self.current_run.completed_steps,
                'failed': self.current_run.failed_steps,
                'skipped': self.current_run.skipped_steps
            },
            'steps': []
        }
        
        for step_id, step in self.steps.items():
            if step.status != PipelineStepStatus.PENDING:
                step_report = {
                    'id': step.id,
                    'name': step.name,
                    'status': step.status.value,
                    'duration': (step.end_time - step.start_time).total_seconds() if step.end_time and step.start_time else None,
                    'attempts': step.attempts
                }
                if step.error:
                    step_report['error'] = step.error
                report['steps'].append(step_report)
        
        # Save report
        report_file = Path(f"reports/pipeline_{self.current_run.run_id}.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Pipeline report saved to {report_file}")
        
        return report


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def create_orchestrator(config: Optional[Dict[str, Any]] = None) -> DataPipelineOrchestrator:
    """Create and initialize a pipeline orchestrator"""
    return DataPipelineOrchestrator(config)


def run_full_pipeline(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the complete 7-step pipeline"""
    orchestrator = create_orchestrator(config)
    run = orchestrator.run_pipeline()
    return {
        'run_id': run.run_id,
        'status': run.status,
        'results': run.results
    }


def run_pipeline_steps(steps: List[int], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run specific pipeline steps"""
    orchestrator = create_orchestrator(config)
    run = orchestrator.run_pipeline(steps=steps, config=config)
    return {
        'run_id': run.run_id,
        'status': run.status,
        'results': run.results
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument('--steps', nargs='+', type=int, help='Steps to run (1-7)')
    parser.add_argument('--resume', type=int, help='Resume from step')
    parser.add_argument('--reset', action='store_true', help='Reset pipeline state')
    
    args = parser.parse_args()
    
    orchestrator = create_orchestrator()
    
    if args.reset:
        orchestrator.reset_pipeline()
        print("Pipeline reset complete")
    elif args.steps:
        run = orchestrator.run_pipeline(steps=args.steps)
        print(f"Pipeline run {run.run_id} completed with status: {run.status}")
    elif args.resume:
        orchestrator.load_state()
        run = orchestrator.run_pipeline(resume_from=args.resume)
        print(f"Pipeline resumed from step {args.resume}, status: {run.status}")
    else:
        # Run full pipeline
        run = orchestrator.run_pipeline()
        print(f"Full pipeline run {run.run_id} completed with status: {run.status}")