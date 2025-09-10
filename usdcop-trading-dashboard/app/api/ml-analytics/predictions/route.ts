import { NextRequest, NextResponse } from 'next/server';

interface PredictionData {
  timestamp: string;
  actual: number;
  predicted: number;
  confidence?: number;
  feature_values?: { [key: string]: number };
  model_version?: string;
  error?: number;
  absolute_error?: number;
  percentage_error?: number;
}

interface PredictionMetrics {
  mse: number;
  mae: number;
  rmse: number;
  mape: number;
  accuracy: number;
  correlation: number;
  total_predictions: number;
  correct_direction: number;
  direction_accuracy: number;
}

// Mock data generator for predictions vs actuals
function generateMockPredictions(count: number = 100): PredictionData[] {
  const predictions: PredictionData[] = [];
  const basePrice = 4200; // Base USD/COP rate
  let currentActual = basePrice;
  
  for (let i = 0; i < count; i++) {
    const timestamp = new Date(Date.now() - (count - i) * 5 * 60 * 1000).toISOString(); // 5-minute intervals
    
    // Simulate price movement with some trend and volatility
    const trend = 0.0001 * Math.sin(i * 0.1); // Small trend component
    const noise = (Math.random() - 0.5) * 0.01; // Random volatility
    const actualChange = trend + noise;
    currentActual += currentActual * actualChange;
    
    // Model prediction with some error
    const predictionError = (Math.random() - 0.5) * 0.005; // Model error
    const predicted = currentActual * (1 + predictionError);
    
    const error = predicted - currentActual;
    const absoluteError = Math.abs(error);
    const percentageError = (absoluteError / currentActual) * 100;
    
    predictions.push({
      timestamp,
      actual: Number(currentActual.toFixed(4)),
      predicted: Number(predicted.toFixed(4)),
      confidence: 0.7 + Math.random() * 0.3, // Random confidence between 0.7-1.0
      feature_values: {
        rsi: 30 + Math.random() * 40,
        macd: (Math.random() - 0.5) * 100,
        bollinger_position: Math.random(),
        volume_ratio: 0.5 + Math.random() * 1.5,
        volatility: 0.01 + Math.random() * 0.05
      },
      model_version: '1.0.0',
      error: Number(error.toFixed(4)),
      absolute_error: Number(absoluteError.toFixed(4)),
      percentage_error: Number(percentageError.toFixed(4))
    });
  }
  
  return predictions;
}

function calculatePredictionMetrics(predictions: PredictionData[]): PredictionMetrics {
  if (predictions.length === 0) {
    return {
      mse: 0, mae: 0, rmse: 0, mape: 0, accuracy: 0, correlation: 0,
      total_predictions: 0, correct_direction: 0, direction_accuracy: 0
    };
  }

  // Calculate basic metrics
  const errors = predictions.map(p => p.error || 0);
  const absoluteErrors = predictions.map(p => p.absolute_error || 0);
  const percentageErrors = predictions.map(p => p.percentage_error || 0);
  
  const mse = errors.reduce((sum, error) => sum + error * error, 0) / errors.length;
  const mae = absoluteErrors.reduce((sum, ae) => sum + ae, 0) / absoluteErrors.length;
  const rmse = Math.sqrt(mse);
  const mape = percentageErrors.reduce((sum, pe) => sum + pe, 0) / percentageErrors.length;
  
  // Calculate correlation
  const actuals = predictions.map(p => p.actual);
  const predicteds = predictions.map(p => p.predicted);
  const correlation = calculateCorrelation(actuals, predicteds);
  
  // Calculate direction accuracy
  let correctDirection = 0;
  for (let i = 1; i < predictions.length; i++) {
    const actualDirection = predictions[i].actual > predictions[i-1].actual;
    const predictedDirection = predictions[i].predicted > predictions[i-1].predicted;
    if (actualDirection === predictedDirection) {
      correctDirection++;
    }
  }
  const directionAccuracy = correctDirection / (predictions.length - 1) * 100;
  
  // Simplified accuracy based on MAPE
  const accuracy = Math.max(0, 100 - mape);
  
  return {
    mse: Number(mse.toFixed(6)),
    mae: Number(mae.toFixed(6)),
    rmse: Number(rmse.toFixed(6)),
    mape: Number(mape.toFixed(2)),
    accuracy: Number(accuracy.toFixed(2)),
    correlation: Number(correlation.toFixed(4)),
    total_predictions: predictions.length,
    correct_direction: correctDirection,
    direction_accuracy: Number(directionAccuracy.toFixed(2))
  };
}

function calculateCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length === 0) return 0;
  
  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
  const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);
  
  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
  
  return denominator === 0 ? 0 : numerator / denominator;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');
    const runId = searchParams.get('runId');
    const limit = parseInt(searchParams.get('limit') || '100');
    const timeRange = searchParams.get('timeRange') || '24h';

    switch (action) {
      case 'data':
        // Get prediction vs actual data
        const predictions = generateMockPredictions(limit);
        
        return NextResponse.json({
          success: true,
          data: predictions,
          metadata: {
            total_count: predictions.length,
            time_range: timeRange,
            model_run_id: runId || 'latest',
            generated_at: new Date().toISOString()
          }
        });

      case 'metrics':
        // Calculate and return prediction metrics
        const predictionData = generateMockPredictions(limit);
        const metrics = calculatePredictionMetrics(predictionData);
        
        return NextResponse.json({
          success: true,
          data: {
            metrics,
            sample_predictions: predictionData.slice(-10), // Last 10 predictions
            calculation_timestamp: new Date().toISOString(),
            model_run_id: runId || 'latest'
          }
        });

      case 'accuracy-over-time':
        // Return accuracy metrics over time windows
        const allPredictions = generateMockPredictions(limit);
        const windowSize = Math.max(10, Math.floor(limit / 10)); // 10 time windows
        const accuracyOverTime = [];
        
        for (let i = 0; i < allPredictions.length; i += windowSize) {
          const windowPredictions = allPredictions.slice(i, i + windowSize);
          if (windowPredictions.length > 0) {
            const windowMetrics = calculatePredictionMetrics(windowPredictions);
            accuracyOverTime.push({
              window_start: windowPredictions[0].timestamp,
              window_end: windowPredictions[windowPredictions.length - 1].timestamp,
              accuracy: windowMetrics.accuracy,
              mape: windowMetrics.mape,
              direction_accuracy: windowMetrics.direction_accuracy,
              correlation: windowMetrics.correlation,
              sample_count: windowPredictions.length
            });
          }
        }
        
        return NextResponse.json({
          success: true,
          data: accuracyOverTime,
          metadata: {
            window_size: windowSize,
            total_windows: accuracyOverTime.length,
            model_run_id: runId || 'latest'
          }
        });

      case 'feature-impact':
        // Analyze feature impact on prediction accuracy
        const featurePredictions = generateMockPredictions(limit);
        const featureNames = ['rsi', 'macd', 'bollinger_position', 'volume_ratio', 'volatility'];
        const featureImpact = featureNames.map(feature => {
          // Simple correlation between feature values and prediction accuracy
          const featureValues = featurePredictions.map(p => p.feature_values?.[feature] || 0);
          const accuracies = featurePredictions.map(p => 100 - (p.percentage_error || 0));
          const impact = Math.abs(calculateCorrelation(featureValues, accuracies));
          
          return {
            feature_name: feature,
            impact_score: Number((impact * 100).toFixed(2)),
            correlation: Number(impact.toFixed(4)),
            importance_rank: 0 // Will be set after sorting
          };
        });
        
        // Sort by impact and assign ranks
        featureImpact.sort((a, b) => b.impact_score - a.impact_score);
        featureImpact.forEach((item, index) => {
          item.importance_rank = index + 1;
        });
        
        return NextResponse.json({
          success: true,
          data: featureImpact,
          metadata: {
            model_run_id: runId || 'latest',
            analysis_timestamp: new Date().toISOString()
          }
        });

      default:
        return NextResponse.json({
          success: false,
          error: 'Invalid action. Supported actions: data, metrics, accuracy-over-time, feature-impact'
        }, { status: 400 });
    }

  } catch (error) {
    console.error('Predictions API Error:', error);
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
    const { predictions, model_run_id } = body;

    if (!predictions || !Array.isArray(predictions)) {
      return NextResponse.json({
        success: false,
        error: 'predictions array is required'
      }, { status: 400 });
    }

    // In a real implementation, this would store predictions to a database
    // For now, we'll just validate and return metrics
    const metrics = calculatePredictionMetrics(predictions);

    return NextResponse.json({
      success: true,
      data: {
        stored_predictions: predictions.length,
        metrics,
        model_run_id: model_run_id || 'unknown',
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Predictions POST API Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Internal server error'
    }, { status: 500 });
  }
}