/**
 * Data Visualization Optimizer
 * Advanced optimization techniques for rendering 92k+ data points
 * Implements LTTB (Largest Triangle Three Buckets) and other sampling algorithms
 */

interface DataPoint {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  [key: string]: any;
}

interface OptimizationConfig {
  maxPoints: number;
  algorithm: 'lttb' | 'uniform' | 'adaptive' | 'none';
  preserveExtremes: boolean;
  volumeWeighted: boolean;
  timeWeighted: boolean;
}

interface OptimizationResult {
  data: DataPoint[];
  originalCount: number;
  optimizedCount: number;
  compressionRatio: number;
  algorithm: string;
  processingTime: number;
}

class DataVisualizationOptimizer {
  private defaultConfig: OptimizationConfig = {
    maxPoints: 2000,
    algorithm: 'lttb',
    preserveExtremes: true,
    volumeWeighted: false,
    timeWeighted: true
  };

  /**
   * Optimize dataset for visualization based on viewport and zoom level
   */
  optimizeForVisualization(
    data: DataPoint[],
    config: Partial<OptimizationConfig> = {}
  ): OptimizationResult {
    const startTime = performance.now();
    const finalConfig = { ...this.defaultConfig, ...config };

    if (data.length <= finalConfig.maxPoints) {
      return {
        data: [...data],
        originalCount: data.length,
        optimizedCount: data.length,
        compressionRatio: 1,
        algorithm: 'none',
        processingTime: performance.now() - startTime
      };
    }

    let optimizedData: DataPoint[];

    switch (finalConfig.algorithm) {
      case 'lttb':
        optimizedData = this.largestTriangleThreeBuckets(data, finalConfig.maxPoints);
        break;
      case 'uniform':
        optimizedData = this.uniformSampling(data, finalConfig.maxPoints);
        break;
      case 'adaptive':
        optimizedData = this.adaptiveSampling(data, finalConfig);
        break;
      default:
        optimizedData = [...data];
    }

    // Preserve extremes if requested
    if (finalConfig.preserveExtremes && optimizedData.length > 2) {
      optimizedData = this.preserveExtremes(data, optimizedData);
    }

    const processingTime = performance.now() - startTime;

    return {
      data: optimizedData,
      originalCount: data.length,
      optimizedCount: optimizedData.length,
      compressionRatio: data.length / optimizedData.length,
      algorithm: finalConfig.algorithm,
      processingTime
    };
  }

  /**
   * Largest Triangle Three Buckets (LTTB) algorithm
   * Best for preserving visual shape of the data
   */
  private largestTriangleThreeBuckets(data: DataPoint[], threshold: number): DataPoint[] {
    if (threshold >= data.length || threshold <= 2) {
      return [...data];
    }

    const sampled: DataPoint[] = [];
    const bucketSize = (data.length - 2) / (threshold - 2);

    // Always include first point
    sampled.push(data[0]);

    let a = 0; // Initially a is the first point in the triangle

    for (let i = 0; i < threshold - 2; i++) {
      // Calculate point average for next bucket (containing c)
      let avgX = 0;
      let avgY = 0;
      let avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
      let avgRangeEnd = Math.floor((i + 2) * bucketSize) + 1;
      avgRangeEnd = avgRangeEnd < data.length ? avgRangeEnd : data.length;

      const avgRangeLength = avgRangeEnd - avgRangeStart;

      for (; avgRangeStart < avgRangeEnd; avgRangeStart++) {
        avgX += data[avgRangeStart].time;
        avgY += data[avgRangeStart].close; // Use close price for LTTB
      }
      avgX /= avgRangeLength;
      avgY /= avgRangeLength;

      // Get the range for this bucket
      let rangeOffs = Math.floor(i * bucketSize) + 1;
      const rangeTo = Math.floor((i + 1) * bucketSize) + 1;

      // Point a
      const pointAX = data[a].time;
      const pointAY = data[a].close;

      let maxArea = -1;
      let maxAreaPoint = -1;

      for (; rangeOffs < rangeTo; rangeOffs++) {
        // Calculate triangle area over three buckets
        const area = Math.abs(
          (pointAX - avgX) * (data[rangeOffs].close - pointAY) -
          (pointAX - data[rangeOffs].time) * (avgY - pointAY)
        ) * 0.5;

        if (area > maxArea) {
          maxArea = area;
          maxAreaPoint = rangeOffs;
        }
      }

      sampled.push(data[maxAreaPoint]);
      a = maxAreaPoint; // This a is the next a (chosen b)
    }

    // Always include last point
    sampled.push(data[data.length - 1]);

    return sampled;
  }

  /**
   * Uniform sampling - evenly spaced points
   */
  private uniformSampling(data: DataPoint[], threshold: number): DataPoint[] {
    if (threshold >= data.length) {
      return [...data];
    }

    const sampled: DataPoint[] = [];
    const step = data.length / threshold;

    for (let i = 0; i < threshold; i++) {
      const index = Math.floor(i * step);
      sampled.push(data[index]);
    }

    return sampled;
  }

  /**
   * Adaptive sampling based on volatility and volume
   */
  private adaptiveSampling(data: DataPoint[], config: OptimizationConfig): DataPoint[] {
    if (data.length <= config.maxPoints) {
      return [...data];
    }

    // Calculate importance scores for each point
    const scores = this.calculateImportanceScores(data, config);

    // Sort by importance and take top points
    const indexed = data.map((point, index) => ({ point, score: scores[index], index }));
    indexed.sort((a, b) => b.score - a.score);

    const selected = indexed.slice(0, config.maxPoints);
    selected.sort((a, b) => a.index - b.index); // Restore time order

    return selected.map(item => item.point);
  }

  /**
   * Calculate importance scores for adaptive sampling
   */
  private calculateImportanceScores(data: DataPoint[], config: OptimizationConfig): number[] {
    const scores: number[] = new Array(data.length).fill(0);

    for (let i = 1; i < data.length - 1; i++) {
      let score = 0;

      // Price volatility contribution
      const priceChange = Math.abs(data[i].close - data[i - 1].close);
      const prevPriceChange = Math.abs(data[i - 1].close - data[i - 2]?.close || data[i - 1].close);
      score += Math.abs(priceChange - prevPriceChange);

      // Volume contribution
      if (config.volumeWeighted && data[i].volume > 0) {
        const volumeRatio = data[i].volume / Math.max(data[i - 1].volume, 1);
        score += Math.log(volumeRatio + 1);
      }

      // High/Low range contribution
      const range = data[i].high - data[i].low;
      const avgPrice = (data[i].high + data[i].low) / 2;
      score += (range / avgPrice) * 100; // Percentage range

      // Time weight (more recent = more important)
      if (config.timeWeighted) {
        const timeWeight = i / data.length;
        score *= (1 + timeWeight * 0.5);
      }

      scores[i] = score;
    }

    // Always include first and last points
    scores[0] = Infinity;
    scores[data.length - 1] = Infinity;

    return scores;
  }

  /**
   * Preserve extreme values (peaks and troughs)
   */
  private preserveExtremes(originalData: DataPoint[], sampledData: DataPoint[]): DataPoint[] {
    const extremes = this.findExtremes(originalData);
    const result = [...sampledData];

    // Add extremes that aren't already in the sampled data
    extremes.forEach(extreme => {
      const exists = result.some(point =>
        Math.abs(point.time - extreme.time) < 1000 // Within 1 second
      );

      if (!exists) {
        result.push(extreme);
      }
    });

    // Sort by time
    result.sort((a, b) => a.time - b.time);

    return result;
  }

  /**
   * Find local extremes (peaks and troughs)
   */
  private findExtremes(data: DataPoint[]): DataPoint[] {
    const extremes: DataPoint[] = [];
    const windowSize = Math.max(5, Math.floor(data.length / 1000));

    for (let i = windowSize; i < data.length - windowSize; i++) {
      const current = data[i];
      let isMaximum = true;
      let isMinimum = true;

      // Check window around current point
      for (let j = i - windowSize; j <= i + windowSize; j++) {
        if (j === i) continue;

        if (data[j].high > current.high) isMaximum = false;
        if (data[j].low < current.low) isMinimum = false;
      }

      if (isMaximum || isMinimum) {
        extremes.push(current);
      }
    }

    return extremes;
  }

  /**
   * Optimize for different zoom levels
   */
  optimizeForZoomLevel(
    data: DataPoint[],
    zoomLevel: number,
    viewportWidth: number = 1920
  ): OptimizationResult {
    // Calculate appropriate point density based on zoom
    const pixelsPerPoint = 2; // Minimum pixels per data point for smooth visualization
    const maxPoints = Math.floor(viewportWidth / pixelsPerPoint);

    // Adjust for zoom level
    const adjustedMaxPoints = Math.floor(maxPoints * Math.sqrt(zoomLevel));

    const config: OptimizationConfig = {
      maxPoints: Math.min(adjustedMaxPoints, 5000), // Cap at 5000 points
      algorithm: zoomLevel > 2 ? 'lttb' : 'adaptive',
      preserveExtremes: zoomLevel < 0.5, // Preserve extremes when zoomed out
      volumeWeighted: zoomLevel > 1,
      timeWeighted: true
    };

    return this.optimizeForVisualization(data, config);
  }

  /**
   * Create multi-resolution data structure for smooth zooming
   */
  createMultiResolutionData(data: DataPoint[]): Map<number, DataPoint[]> {
    const resolutions = new Map<number, DataPoint[]>();
    const levels = [100, 500, 1000, 2000, 5000];

    levels.forEach(level => {
      if (data.length > level) {
        const result = this.optimizeForVisualization(data, {
          maxPoints: level,
          algorithm: 'lttb',
          preserveExtremes: true
        });
        resolutions.set(level, result.data);
      } else {
        resolutions.set(level, [...data]);
      }
    });

    return resolutions;
  }

  /**
   * Get optimal resolution for current viewport
   */
  getOptimalResolution(
    multiResData: Map<number, DataPoint[]>,
    viewportSize: number,
    zoomLevel: number
  ): DataPoint[] {
    const targetPoints = Math.floor(viewportSize * Math.sqrt(zoomLevel));

    // Find the best matching resolution
    const availableResolutions = Array.from(multiResData.keys()).sort((a, b) => a - b);

    let bestResolution = availableResolutions[0];
    for (const resolution of availableResolutions) {
      if (resolution <= targetPoints * 1.5) {
        bestResolution = resolution;
      } else {
        break;
      }
    }

    return multiResData.get(bestResolution) || [];
  }

  /**
   * Performance benchmark for optimization algorithms
   */
  benchmark(data: DataPoint[], iterations: number = 10): Record<string, number> {
    const algorithms: Array<OptimizationConfig['algorithm']> = ['lttb', 'uniform', 'adaptive'];
    const results: Record<string, number> = {};

    algorithms.forEach(algorithm => {
      const times: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const startTime = performance.now();
        this.optimizeForVisualization(data, { algorithm, maxPoints: 2000 });
        times.push(performance.now() - startTime);
      }

      results[algorithm] = times.reduce((a, b) => a + b, 0) / times.length;
    });

    return results;
  }
}

// Create and export singleton instance
export const dataVisualizationOptimizer = new DataVisualizationOptimizer();
export default dataVisualizationOptimizer;