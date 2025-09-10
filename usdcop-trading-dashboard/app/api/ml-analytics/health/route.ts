import { NextRequest, NextResponse } from 'next/server';

interface ModelHealthStatus {
  model_id: string;
  model_name: string;
  status: 'healthy' | 'warning' | 'critical' | 'offline';
  last_prediction_time: string;
  health_score: number; // 0-100
  alerts: ModelAlert[];
  metrics: {
    prediction_latency: number; // milliseconds
    throughput: number; // predictions per minute
    error_rate: number; // percentage
    drift_score: number; // 0-1
    confidence_avg: number; // 0-1
  };
  resource_usage: {
    cpu_usage: number; // percentage
    memory_usage: number; // MB
    disk_usage: number; // MB
  };
}

interface ModelAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  type: 'drift' | 'performance' | 'resource' | 'availability';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  threshold_value?: number;
  current_value?: number;
}

interface SystemHealthSummary {
  overall_status: 'healthy' | 'warning' | 'critical';
  total_models: number;
  healthy_models: number;
  models_with_warnings: number;
  critical_models: number;
  offline_models: number;
  total_alerts: number;
  critical_alerts: number;
  last_updated: string;
}

// Mock data generators
function generateModelHealth(modelId: string, modelName: string): ModelHealthStatus {
  const now = new Date();
  const lastPredictionTime = new Date(now.getTime() - Math.random() * 10 * 60 * 1000); // Within last 10 minutes
  
  // Generate random but realistic health metrics
  const errorRate = Math.random() * 15; // 0-15%
  const driftScore = Math.random() * 0.3; // 0-0.3
  const confidenceAvg = 0.6 + Math.random() * 0.35; // 0.6-0.95
  const predictionLatency = 50 + Math.random() * 200; // 50-250ms
  
  // Calculate health score based on metrics
  let healthScore = 100;
  healthScore -= errorRate * 2; // Penalize high error rate
  healthScore -= driftScore * 100; // Penalize drift
  healthScore -= Math.max(0, predictionLatency - 100) * 0.1; // Penalize high latency
  healthScore = Math.max(10, Math.min(100, healthScore));
  
  // Determine status based on health score
  let status: 'healthy' | 'warning' | 'critical' | 'offline';
  if (Math.random() > 0.95) {
    status = 'offline';
    healthScore = 0;
  } else if (healthScore < 60) {
    status = 'critical';
  } else if (healthScore < 80) {
    status = 'warning';
  } else {
    status = 'healthy';
  }
  
  // Generate alerts based on status and metrics
  const alerts: ModelAlert[] = [];
  
  if (errorRate > 10) {
    alerts.push({
      id: `alert_${modelId}_error_rate`,
      severity: errorRate > 15 ? 'critical' : 'warning',
      type: 'performance',
      title: 'High Error Rate',
      message: `Model error rate is ${errorRate.toFixed(1)}%, exceeding threshold of 10%`,
      timestamp: new Date(now.getTime() - Math.random() * 60 * 60 * 1000).toISOString(),
      acknowledged: Math.random() > 0.7,
      threshold_value: 10,
      current_value: errorRate
    });
  }
  
  if (driftScore > 0.2) {
    alerts.push({
      id: `alert_${modelId}_drift`,
      severity: driftScore > 0.25 ? 'critical' : 'warning',
      type: 'drift',
      title: 'Model Drift Detected',
      message: `Data drift score is ${(driftScore * 100).toFixed(1)}%, indicating potential model degradation`,
      timestamp: new Date(now.getTime() - Math.random() * 2 * 60 * 60 * 1000).toISOString(),
      acknowledged: Math.random() > 0.6,
      threshold_value: 20,
      current_value: driftScore * 100
    });
  }
  
  if (predictionLatency > 200) {
    alerts.push({
      id: `alert_${modelId}_latency`,
      severity: predictionLatency > 300 ? 'critical' : 'warning',
      type: 'performance',
      title: 'High Prediction Latency',
      message: `Average prediction latency is ${predictionLatency.toFixed(0)}ms, exceeding SLA of 200ms`,
      timestamp: new Date(now.getTime() - Math.random() * 30 * 60 * 1000).toISOString(),
      acknowledged: Math.random() > 0.8,
      threshold_value: 200,
      current_value: predictionLatency
    });
  }
  
  if (status === 'offline') {
    alerts.push({
      id: `alert_${modelId}_offline`,
      severity: 'critical',
      type: 'availability',
      title: 'Model Offline',
      message: `Model has not produced predictions in the last ${Math.floor(Math.random() * 30 + 10)} minutes`,
      timestamp: lastPredictionTime.toISOString(),
      acknowledged: false
    });
  }
  
  return {
    model_id: modelId,
    model_name: modelName,
    status,
    last_prediction_time: status === 'offline' ? 
      new Date(now.getTime() - (20 + Math.random() * 40) * 60 * 1000).toISOString() : 
      lastPredictionTime.toISOString(),
    health_score: Math.round(healthScore),
    alerts,
    metrics: {
      prediction_latency: Math.round(predictionLatency),
      throughput: Math.round(60 + Math.random() * 120), // 60-180 predictions per minute
      error_rate: Number(errorRate.toFixed(2)),
      drift_score: Number(driftScore.toFixed(3)),
      confidence_avg: Number(confidenceAvg.toFixed(3))
    },
    resource_usage: {
      cpu_usage: Math.round(20 + Math.random() * 60), // 20-80%
      memory_usage: Math.round(500 + Math.random() * 1500), // 500-2000 MB
      disk_usage: Math.round(100 + Math.random() * 900) // 100-1000 MB
    }
  };
}

function generateSystemHealthSummary(models: ModelHealthStatus[]): SystemHealthSummary {
  const totalModels = models.length;
  const healthyModels = models.filter(m => m.status === 'healthy').length;
  const modelsWithWarnings = models.filter(m => m.status === 'warning').length;
  const criticalModels = models.filter(m => m.status === 'critical').length;
  const offlineModels = models.filter(m => m.status === 'offline').length;
  
  const totalAlerts = models.reduce((sum, model) => sum + model.alerts.length, 0);
  const criticalAlerts = models.reduce((sum, model) => 
    sum + model.alerts.filter(alert => alert.severity === 'critical').length, 0
  );
  
  // Determine overall status
  let overallStatus: 'healthy' | 'warning' | 'critical';
  if (criticalModels > 0 || offlineModels > 0 || criticalAlerts > 0) {
    overallStatus = 'critical';
  } else if (modelsWithWarnings > 0) {
    overallStatus = 'warning';
  } else {
    overallStatus = 'healthy';
  }
  
  return {
    overall_status: overallStatus,
    total_models: totalModels,
    healthy_models: healthyModels,
    models_with_warnings: modelsWithWarnings,
    critical_models: criticalModels,
    offline_models: offlineModels,
    total_alerts: totalAlerts,
    critical_alerts: criticalAlerts,
    last_updated: new Date().toISOString()
  };
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');
    const modelId = searchParams.get('modelId');

    switch (action) {
      case 'summary':
        // Generate mock models and their health status
        const models = [
          generateModelHealth('model_001', 'USDCOP_PPO_v1.2'),
          generateModelHealth('model_002', 'USDCOP_A2C_v1.1'),
          generateModelHealth('model_003', 'USDCOP_DQN_v2.0'),
          generateModelHealth('model_004', 'USDCOP_LSTM_v1.5'),
          generateModelHealth('model_005', 'USDCOP_Ensemble_v1.0')
        ];
        
        const summary = generateSystemHealthSummary(models);
        
        return NextResponse.json({
          success: true,
          data: {
            summary,
            models: models.map(model => ({
              model_id: model.model_id,
              model_name: model.model_name,
              status: model.status,
              health_score: model.health_score,
              alert_count: model.alerts.length,
              critical_alerts: model.alerts.filter(a => a.severity === 'critical').length
            }))
          }
        });

      case 'detail':
        if (!modelId) {
          return NextResponse.json({
            success: false,
            error: 'modelId parameter is required for detail view'
          }, { status: 400 });
        }
        
        const modelHealth = generateModelHealth(modelId, `Model_${modelId}`);
        
        return NextResponse.json({
          success: true,
          data: modelHealth
        });

      case 'alerts':
        // Get all alerts across all models
        const allModels = [
          generateModelHealth('model_001', 'USDCOP_PPO_v1.2'),
          generateModelHealth('model_002', 'USDCOP_A2C_v1.1'),
          generateModelHealth('model_003', 'USDCOP_DQN_v2.0'),
          generateModelHealth('model_004', 'USDCOP_LSTM_v1.5'),
          generateModelHealth('model_005', 'USDCOP_Ensemble_v1.0')
        ];
        
        const allAlerts = allModels.flatMap(model => 
          model.alerts.map(alert => ({
            ...alert,
            model_id: model.model_id,
            model_name: model.model_name
          }))
        );
        
        // Sort by timestamp (most recent first)
        allAlerts.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
        
        return NextResponse.json({
          success: true,
          data: {
            alerts: allAlerts,
            summary: {
              total: allAlerts.length,
              critical: allAlerts.filter(a => a.severity === 'critical').length,
              warning: allAlerts.filter(a => a.severity === 'warning').length,
              info: allAlerts.filter(a => a.severity === 'info').length,
              unacknowledged: allAlerts.filter(a => !a.acknowledged).length
            }
          }
        });

      case 'metrics-history':
        // Generate historical health metrics for a model
        const historyData = [];
        const now = new Date();
        
        for (let i = 23; i >= 0; i--) { // Last 24 hours
          const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
          historyData.push({
            timestamp: timestamp.toISOString(),
            health_score: 70 + Math.random() * 25 + Math.sin(i * 0.2) * 10,
            error_rate: 2 + Math.random() * 5 + Math.sin(i * 0.3) * 2,
            prediction_latency: 80 + Math.random() * 40 + Math.sin(i * 0.1) * 20,
            throughput: 100 + Math.random() * 50 + Math.sin(i * 0.15) * 25,
            drift_score: 0.05 + Math.random() * 0.1 + Math.sin(i * 0.25) * 0.05
          });
        }
        
        return NextResponse.json({
          success: true,
          data: {
            model_id: modelId || 'model_001',
            time_range: '24h',
            metrics: historyData
          }
        });

      default:
        return NextResponse.json({
          success: false,
          error: 'Invalid action. Supported actions: summary, detail, alerts, metrics-history'
        }, { status: 400 });
    }

  } catch (error) {
    console.error('Model Health API Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Internal server error',
      details: process.env.NODE_ENV === 'development' ? error : undefined
    }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, alert_id, model_id } = body;

    switch (action) {
      case 'acknowledge-alert':
        if (!alert_id) {
          return NextResponse.json({
            success: false,
            error: 'alert_id is required'
          }, { status: 400 });
        }
        
        return NextResponse.json({
          success: true,
          data: {
            alert_id,
            acknowledged: true,
            acknowledged_at: new Date().toISOString(),
            acknowledged_by: 'current_user'
          }
        });

      case 'silence-alerts':
        if (!model_id) {
          return NextResponse.json({
            success: false,
            error: 'model_id is required'
          }, { status: 400 });
        }
        
        const silenceDuration = body.duration || 3600; // Default 1 hour
        
        return NextResponse.json({
          success: true,
          data: {
            model_id,
            silenced: true,
            silence_duration: silenceDuration,
            silenced_until: new Date(Date.now() + silenceDuration * 1000).toISOString()
          }
        });

      default:
        return NextResponse.json({
          success: false,
          error: 'Invalid action. Supported actions: acknowledge-alert, silence-alerts'
        }, { status: 400 });
    }

  } catch (error) {
    console.error('Model Health POST API Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Internal server error'
    }, { status: 500 });
  }
}