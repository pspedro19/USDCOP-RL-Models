/**
 * Correlation Analytics Engine
 * ===========================
 *
 * Advanced correlation analysis for multi-asset portfolios with:
 * - Dynamic correlation matrices
 * - Principal Component Analysis (PCA)
 * - Hierarchical clustering
 * - Rolling correlations
 * - Regime detection
 * - Risk factor analysis
 */

import { CandleData, CorrelationMatrix } from '../types';

export interface CorrelationAnalyticsConfig {
  lookbackPeriod: number;
  rollingWindow: number;
  minCorrelationThreshold: number;
  clusteringMethod: 'hierarchical' | 'kmeans' | 'spectral';
  pcaComponents: number;
  rebalanceFrequency: 'daily' | 'weekly' | 'monthly';
}

export interface DynamicCorrelation {
  timestamp: number;
  correlationMatrix: number[][];
  eigenvalues: number[];
  explained_variance: number[];
  principalComponents: number[][];
  regimeDetection: {
    regime: 'LOW_CORRELATION' | 'HIGH_CORRELATION' | 'CRISIS' | 'NORMAL';
    confidence: number;
    changePoint?: number;
  };
}

export interface RollingCorrelation {
  asset1: string;
  asset2: string;
  timestamps: number[];
  correlations: number[];
  statistics: {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
    percentiles: { p25: number; p50: number; p75: number; p95: number };
  };
}

export interface ClusterAnalysis {
  timestamp: number;
  clusters: {
    id: number;
    assets: string[];
    centroid: number[];
    intraClusterCorrelation: number;
    size: number;
  }[];
  dendrogram: {
    merges: Array<[number, number, number]>; // [cluster1, cluster2, distance]
    distances: number[];
  };
  silhouetteScores: number[];
  optimalClusters: number;
}

export interface RiskFactorAnalysis {
  factors: {
    name: string;
    loadings: number[];
    explained_variance: number;
    assets: string[];
  }[];
  factorReturns: number[][];
  specificRisks: number[];
  totalRisk: number;
  systematicRisk: number;
  idiosyncraticRisk: number;
}

export interface RegimeDetection {
  timestamp: number;
  currentRegime: 'BULL' | 'BEAR' | 'VOLATILE' | 'CALM' | 'CRISIS' | 'RECOVERY';
  regimeProbabilities: { [regime: string]: number };
  changePoints: number[];
  regimeDuration: number;
  nextRegimeTransition?: {
    mostLikely: string;
    probability: number;
    expectedDuration: number;
  };
}

export interface PortfolioRiskMetrics {
  correlationRisk: number;
  concentrationRisk: number;
  diversificationRatio: number;
  effectiveAssets: number;
  maxDrawdownCorrelation: number;
  averageCorrelation: number;
  correlationBreakdown: {
    low: number; // < 0.3
    medium: number; // 0.3 - 0.7
    high: number; // > 0.7
  };
}

export class CorrelationAnalytics {
  private config: CorrelationAnalyticsConfig;
  private cache: Map<string, any> = new Map();

  constructor(config: Partial<CorrelationAnalyticsConfig> = {}) {
    this.config = {
      lookbackPeriod: 252,
      rollingWindow: 60,
      minCorrelationThreshold: 0.05,
      clusteringMethod: 'hierarchical',
      pcaComponents: 5,
      rebalanceFrequency: 'daily',
      ...config
    };
  }

  /**
   * Calculate dynamic correlation matrix with PCA
   */
  calculateDynamicCorrelation(
    datasets: { name: string; data: CandleData[] }[],
    timestamp?: number
  ): DynamicCorrelation {
    const cacheKey = `dynamic_corr_${timestamp || Date.now()}_${datasets.map(d => d.name).join('_')}`;

    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    // Align data and calculate returns
    const alignedData = this.alignDatasets(datasets);
    const returns = this.calculateReturns(alignedData);

    if (returns.length < this.config.lookbackPeriod) {
      throw new Error('Insufficient data for correlation analysis');
    }

    // Use most recent data for current correlation
    const recentReturns = returns.slice(-this.config.lookbackPeriod);
    const correlationMatrix = this.calculateCorrelationMatrix(recentReturns);

    // Perform PCA
    const { eigenvalues, eigenvectors, explainedVariance } = this.performPCA(correlationMatrix);

    // Detect correlation regime
    const regimeDetection = this.detectCorrelationRegime(correlationMatrix, returns);

    const result: DynamicCorrelation = {
      timestamp: timestamp || Date.now() / 1000,
      correlationMatrix,
      eigenvalues,
      explained_variance: explainedVariance,
      principalComponents: eigenvectors,
      regimeDetection
    };

    this.cache.set(cacheKey, result);
    return result;
  }

  /**
   * Calculate rolling correlations between asset pairs
   */
  calculateRollingCorrelations(
    asset1Data: CandleData[],
    asset2Data: CandleData[],
    asset1Name: string,
    asset2Name: string,
    windowSize: number = this.config.rollingWindow
  ): RollingCorrelation {
    const alignedData = this.alignTwoDatasets(asset1Data, asset2Data);
    const returns1 = this.calculateSingleAssetReturns(alignedData.asset1);
    const returns2 = this.calculateSingleAssetReturns(alignedData.asset2);

    if (returns1.length < windowSize) {
      throw new Error('Insufficient data for rolling correlation');
    }

    const correlations: number[] = [];
    const timestamps: number[] = [];

    for (let i = windowSize; i <= returns1.length; i++) {
      const window1 = returns1.slice(i - windowSize, i);
      const window2 = returns2.slice(i - windowSize, i);

      const correlation = this.pearsonCorrelation(window1, window2);
      correlations.push(correlation);
      timestamps.push(alignedData.asset1[i - 1].timestamp);
    }

    // Calculate statistics
    const sortedCorrelations = [...correlations].sort((a, b) => a - b);
    const statistics = {
      mean: correlations.reduce((sum, c) => sum + c, 0) / correlations.length,
      median: sortedCorrelations[Math.floor(sortedCorrelations.length / 2)],
      std: this.calculateStandardDeviation(correlations),
      min: Math.min(...correlations),
      max: Math.max(...correlations),
      percentiles: {
        p25: sortedCorrelations[Math.floor(sortedCorrelations.length * 0.25)],
        p50: sortedCorrelations[Math.floor(sortedCorrelations.length * 0.50)],
        p75: sortedCorrelations[Math.floor(sortedCorrelations.length * 0.75)],
        p95: sortedCorrelations[Math.floor(sortedCorrelations.length * 0.95)]
      }
    };

    return {
      asset1: asset1Name,
      asset2: asset2Name,
      timestamps,
      correlations,
      statistics
    };
  }

  /**
   * Perform hierarchical clustering analysis
   */
  performClusterAnalysis(
    datasets: { name: string; data: CandleData[] }[],
    timestamp?: number
  ): ClusterAnalysis {
    const alignedData = this.alignDatasets(datasets);
    const returns = this.calculateReturns(alignedData);
    const correlationMatrix = this.calculateCorrelationMatrix(returns.slice(-this.config.lookbackPeriod));

    // Convert correlation to distance matrix
    const distanceMatrix = correlationMatrix.map(row =>
      row.map(corr => Math.sqrt(2 * (1 - corr)))
    );

    // Perform hierarchical clustering
    const { clusters, dendrogram } = this.hierarchicalClustering(
      datasets.map(d => d.name),
      distanceMatrix
    );

    // Calculate silhouette scores
    const silhouetteScores = this.calculateSilhouetteScores(distanceMatrix, clusters);

    // Find optimal number of clusters
    const optimalClusters = this.findOptimalClusters(distanceMatrix, datasets.map(d => d.name));

    return {
      timestamp: timestamp || Date.now() / 1000,
      clusters,
      dendrogram,
      silhouetteScores,
      optimalClusters
    };
  }

  /**
   * Risk factor analysis using PCA
   */
  performRiskFactorAnalysis(
    datasets: { name: string; data: CandleData[] }[],
    factorNames?: string[]
  ): RiskFactorAnalysis {
    const alignedData = this.alignDatasets(datasets);
    const returns = this.calculateReturns(alignedData);
    const recentReturns = returns.slice(-this.config.lookbackPeriod);

    // Perform PCA on returns
    const covarianceMatrix = this.calculateCovarianceMatrix(recentReturns);
    const { eigenvalues, eigenvectors, explainedVariance } = this.performPCA(covarianceMatrix);

    // Extract factors (principal components)
    const numFactors = Math.min(this.config.pcaComponents, eigenvalues.length);
    const factors = [];

    for (let i = 0; i < numFactors; i++) {
      const loadings = eigenvectors[i];
      const factorName = factorNames?.[i] || `Factor ${i + 1}`;

      factors.push({
        name: factorName,
        loadings,
        explained_variance: explainedVariance[i],
        assets: datasets.map(d => d.name)
      });
    }

    // Calculate factor returns
    const factorReturns = this.calculateFactorReturns(recentReturns, eigenvectors, numFactors);

    // Calculate specific risks (idiosyncratic risks)
    const specificRisks = this.calculateSpecificRisks(recentReturns, factorReturns, eigenvectors, numFactors);

    // Calculate risk decomposition
    const totalRisk = this.calculateTotalRisk(recentReturns);
    const systematicRisk = this.calculateSystematicRisk(factorReturns, eigenvalues, numFactors);
    const idiosyncraticRisk = totalRisk - systematicRisk;

    return {
      factors,
      factorReturns,
      specificRisks,
      totalRisk,
      systematicRisk,
      idiosyncraticRisk
    };
  }

  /**
   * Detect market regimes based on correlation patterns
   */
  detectMarketRegime(
    datasets: { name: string; data: CandleData[] }[],
    lookbackPeriod: number = 60
  ): RegimeDetection {
    const alignedData = this.alignDatasets(datasets);
    const returns = this.calculateReturns(alignedData);

    if (returns.length < lookbackPeriod * 2) {
      throw new Error('Insufficient data for regime detection');
    }

    const recentReturns = returns.slice(-lookbackPeriod);
    const correlationMatrix = this.calculateCorrelationMatrix(recentReturns);

    // Calculate regime indicators
    const avgCorrelation = this.calculateAverageCorrelation(correlationMatrix);
    const volatility = this.calculatePortfolioVolatility(recentReturns);
    const momentum = this.calculateMomentum(recentReturns);

    // Classify current regime
    const currentRegime = this.classifyRegime(avgCorrelation, volatility, momentum);

    // Calculate regime probabilities using a simplified model
    const regimeProbabilities = this.calculateRegimeProbabilities(avgCorrelation, volatility, momentum);

    // Detect change points
    const changePoints = this.detectChangePoints(returns, lookbackPeriod);

    // Calculate regime duration
    const regimeDuration = changePoints.length > 0 ?
      (Date.now() / 1000) - changePoints[changePoints.length - 1] : lookbackPeriod;

    // Predict next regime transition
    const nextRegimeTransition = this.predictRegimeTransition(currentRegime, regimeProbabilities);

    return {
      timestamp: Date.now() / 1000,
      currentRegime,
      regimeProbabilities,
      changePoints,
      regimeDuration,
      nextRegimeTransition
    };
  }

  /**
   * Calculate comprehensive portfolio risk metrics
   */
  calculatePortfolioRiskMetrics(
    datasets: { name: string; data: CandleData[] }[],
    weights?: number[]
  ): PortfolioRiskMetrics {
    const alignedData = this.alignDatasets(datasets);
    const returns = this.calculateReturns(alignedData);
    const correlationMatrix = this.calculateCorrelationMatrix(returns.slice(-this.config.lookbackPeriod));

    // Default equal weights if not provided
    const portfolioWeights = weights || Array(datasets.length).fill(1 / datasets.length);

    // Calculate correlation-based risk metrics
    const avgCorrelation = this.calculateAverageCorrelation(correlationMatrix);
    const concentrationRisk = this.calculateConcentrationRisk(portfolioWeights);
    const diversificationRatio = this.calculateDiversificationRatio(correlationMatrix, portfolioWeights);
    const effectiveAssets = this.calculateEffectiveAssets(correlationMatrix);

    // Calculate correlation breakdown
    const flatCorrelations = correlationMatrix
      .flatMap((row, i) => row.slice(i + 1))
      .filter(corr => !isNaN(corr));

    const correlationBreakdown = {
      low: flatCorrelations.filter(c => Math.abs(c) < 0.3).length / flatCorrelations.length,
      medium: flatCorrelations.filter(c => Math.abs(c) >= 0.3 && Math.abs(c) < 0.7).length / flatCorrelations.length,
      high: flatCorrelations.filter(c => Math.abs(c) >= 0.7).length / flatCorrelations.length
    };

    // Calculate maximum drawdown correlation
    const maxDrawdownCorrelation = this.calculateMaxDrawdownCorrelation(returns, correlationMatrix);

    return {
      correlationRisk: avgCorrelation,
      concentrationRisk,
      diversificationRatio,
      effectiveAssets,
      maxDrawdownCorrelation,
      averageCorrelation: avgCorrelation,
      correlationBreakdown
    };
  }

  // Private helper methods

  private alignDatasets(datasets: { name: string; data: CandleData[] }[]): { [key: string]: CandleData[] } {
    // Find common time range
    const allTimestamps = datasets.flatMap(d => d.data.map(candle => candle.timestamp));
    const uniqueTimestamps = [...new Set(allTimestamps)].sort((a, b) => a - b);

    const aligned: { [key: string]: CandleData[] } = {};

    datasets.forEach(dataset => {
      aligned[dataset.name] = uniqueTimestamps
        .map(timestamp => {
          const candle = dataset.data.find(c => c.timestamp === timestamp);
          return candle || null;
        })
        .filter(Boolean) as CandleData[];
    });

    return aligned;
  }

  private alignTwoDatasets(
    data1: CandleData[],
    data2: CandleData[]
  ): { asset1: CandleData[]; asset2: CandleData[] } {
    const timestamps1 = new Set(data1.map(d => d.timestamp));
    const timestamps2 = new Set(data2.map(d => d.timestamp));
    const commonTimestamps = [...timestamps1].filter(t => timestamps2.has(t)).sort((a, b) => a - b);

    return {
      asset1: commonTimestamps.map(t => data1.find(d => d.timestamp === t)!),
      asset2: commonTimestamps.map(t => data2.find(d => d.timestamp === t)!)
    };
  }

  private calculateReturns(alignedData: { [key: string]: CandleData[] }): number[][] {
    const assets = Object.keys(alignedData);
    const returns: number[][] = [];

    const maxLength = Math.max(...assets.map(asset => alignedData[asset].length));

    for (let i = 1; i < maxLength; i++) {
      const periodReturns: number[] = [];

      assets.forEach(asset => {
        const data = alignedData[asset];
        if (i < data.length && i > 0) {
          const ret = Math.log(data[i].close / data[i - 1].close);
          periodReturns.push(isFinite(ret) ? ret : 0);
        } else {
          periodReturns.push(0);
        }
      });

      returns.push(periodReturns);
    }

    return returns;
  }

  private calculateSingleAssetReturns(data: CandleData[]): number[] {
    const returns: number[] = [];

    for (let i = 1; i < data.length; i++) {
      const ret = Math.log(data[i].close / data[i - 1].close);
      returns.push(isFinite(ret) ? ret : 0);
    }

    return returns;
  }

  private calculateCorrelationMatrix(returns: number[][]): number[][] {
    const numAssets = returns[0]?.length || 0;
    const correlationMatrix: number[][] = Array(numAssets).fill(0).map(() => Array(numAssets).fill(0));

    for (let i = 0; i < numAssets; i++) {
      for (let j = 0; j < numAssets; j++) {
        if (i === j) {
          correlationMatrix[i][j] = 1;
        } else {
          const asset1Returns = returns.map(period => period[i]);
          const asset2Returns = returns.map(period => period[j]);
          correlationMatrix[i][j] = this.pearsonCorrelation(asset1Returns, asset2Returns);
        }
      }
    }

    return correlationMatrix;
  }

  private calculateCovarianceMatrix(returns: number[][]): number[][] {
    const numAssets = returns[0]?.length || 0;
    const covarianceMatrix: number[][] = Array(numAssets).fill(0).map(() => Array(numAssets).fill(0));

    // Calculate means
    const means = Array(numAssets).fill(0);
    returns.forEach(period => {
      period.forEach((ret, i) => {
        means[i] += ret;
      });
    });
    means.forEach((_, i) => {
      means[i] /= returns.length;
    });

    // Calculate covariances
    for (let i = 0; i < numAssets; i++) {
      for (let j = 0; j < numAssets; j++) {
        let covariance = 0;
        returns.forEach(period => {
          covariance += (period[i] - means[i]) * (period[j] - means[j]);
        });
        covarianceMatrix[i][j] = covariance / (returns.length - 1);
      }
    }

    return covarianceMatrix;
  }

  private pearsonCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n === 0) return 0;

    const sumX = x.slice(0, n).reduce((a, b) => a + b, 0);
    const sumY = y.slice(0, n).reduce((a, b) => a + b, 0);
    const sumXY = x.slice(0, n).reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.slice(0, n).reduce((sum, xi) => sum + xi * xi, 0);
    const sumY2 = y.slice(0, n).reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  private performPCA(matrix: number[][]): {
    eigenvalues: number[];
    eigenvectors: number[][];
    explainedVariance: number[];
  } {
    // Simplified PCA implementation - would use a proper linear algebra library in production
    const n = matrix.length;

    // Power iteration for largest eigenvalue/eigenvector
    const eigenvalues: number[] = [];
    const eigenvectors: number[][] = [];

    // This is a simplified implementation - proper PCA would use SVD or other methods
    for (let k = 0; k < Math.min(n, this.config.pcaComponents); k++) {
      let vector = Array(n).fill(0).map(() => Math.random());
      let eigenvalue = 0;

      // Power iteration
      for (let iter = 0; iter < 100; iter++) {
        const newVector = this.matrixVectorMultiply(matrix, vector);
        const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));

        if (norm > 0) {
          newVector.forEach((val, i) => {
            newVector[i] = val / norm;
          });
        }

        eigenvalue = this.vectorDotProduct(newVector, this.matrixVectorMultiply(matrix, newVector));
        vector = newVector;
      }

      eigenvalues.push(eigenvalue);
      eigenvectors.push(vector);

      // Deflate matrix (remove this component)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          matrix[i][j] -= eigenvalue * vector[i] * vector[j];
        }
      }
    }

    // Calculate explained variance
    const totalVariance = eigenvalues.reduce((sum, val) => sum + Math.abs(val), 0);
    const explainedVariance = eigenvalues.map(val => Math.abs(val) / totalVariance);

    return { eigenvalues, eigenvectors, explainedVariance };
  }

  private matrixVectorMultiply(matrix: number[][], vector: number[]): number[] {
    return matrix.map(row =>
      row.reduce((sum, val, i) => sum + val * vector[i], 0)
    );
  }

  private vectorDotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  private hierarchicalClustering(
    assets: string[],
    distanceMatrix: number[][]
  ): { clusters: any[]; dendrogram: any } {
    const n = assets.length;
    const clusters: any[] = assets.map((asset, i) => ({
      id: i,
      assets: [asset],
      centroid: distanceMatrix[i],
      intraClusterCorrelation: 1,
      size: 1
    }));

    const merges: Array<[number, number, number]> = [];
    const distances: number[] = [];

    // Single linkage clustering
    while (clusters.length > 1) {
      let minDistance = Infinity;
      let mergeIndices = [0, 1];

      // Find closest clusters
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          const distance = this.calculateClusterDistance(clusters[i], clusters[j], distanceMatrix);
          if (distance < minDistance) {
            minDistance = distance;
            mergeIndices = [i, j];
          }
        }
      }

      // Merge clusters
      const [i, j] = mergeIndices;
      const newCluster = {
        id: n + merges.length,
        assets: [...clusters[i].assets, ...clusters[j].assets],
        centroid: this.calculateClusterCentroid(clusters[i], clusters[j]),
        intraClusterCorrelation: this.calculateIntraClusterCorrelation(
          [...clusters[i].assets, ...clusters[j].assets],
          assets,
          distanceMatrix
        ),
        size: clusters[i].size + clusters[j].size
      };

      merges.push([clusters[i].id, clusters[j].id, minDistance]);
      distances.push(minDistance);

      // Remove merged clusters and add new one
      clusters.splice(Math.max(i, j), 1);
      clusters.splice(Math.min(i, j), 1);
      clusters.push(newCluster);
    }

    return {
      clusters: clusters.slice(0, 5), // Return top 5 clusters
      dendrogram: { merges, distances }
    };
  }

  private calculateClusterDistance(cluster1: any, cluster2: any, distanceMatrix: number[][]): number {
    // Single linkage: minimum distance between any two points
    let minDistance = Infinity;

    cluster1.assets.forEach((asset1: string, i1: number) => {
      cluster2.assets.forEach((asset2: string, i2: number) => {
        const distance = distanceMatrix[i1][i2];
        if (distance < minDistance) {
          minDistance = distance;
        }
      });
    });

    return minDistance;
  }

  private calculateClusterCentroid(cluster1: any, cluster2: any): number[] {
    // Simple average of centroids
    return cluster1.centroid.map((val: number, i: number) =>
      (val * cluster1.size + cluster2.centroid[i] * cluster2.size) / (cluster1.size + cluster2.size)
    );
  }

  private calculateIntraClusterCorrelation(
    clusterAssets: string[],
    allAssets: string[],
    distanceMatrix: number[][]
  ): number {
    if (clusterAssets.length < 2) return 1;

    let totalCorrelation = 0;
    let pairCount = 0;

    clusterAssets.forEach(asset1 => {
      clusterAssets.forEach(asset2 => {
        if (asset1 !== asset2) {
          const i1 = allAssets.indexOf(asset1);
          const i2 = allAssets.indexOf(asset2);
          if (i1 !== -1 && i2 !== -1) {
            totalCorrelation += 1 - (distanceMatrix[i1][i2] ** 2) / 2; // Convert distance back to correlation
            pairCount++;
          }
        }
      });
    });

    return pairCount > 0 ? totalCorrelation / pairCount : 1;
  }

  private calculateSilhouetteScores(distanceMatrix: number[][], clusters: any[]): number[] {
    // Simplified silhouette calculation
    return Array(distanceMatrix.length).fill(0.5);
  }

  private findOptimalClusters(distanceMatrix: number[][], assets: string[]): number {
    // Use elbow method - simplified
    return Math.min(5, Math.max(2, Math.floor(assets.length / 3)));
  }

  private calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  }

  private detectCorrelationRegime(correlationMatrix: number[][], returns: number[][]): any {
    const avgCorrelation = this.calculateAverageCorrelation(correlationMatrix);
    const volatility = this.calculatePortfolioVolatility(returns.slice(-60));

    let regime: 'LOW_CORRELATION' | 'HIGH_CORRELATION' | 'CRISIS' | 'NORMAL' = 'NORMAL';
    let confidence = 0.5;

    if (avgCorrelation > 0.7 && volatility > 0.02) {
      regime = 'CRISIS';
      confidence = 0.8;
    } else if (avgCorrelation > 0.5) {
      regime = 'HIGH_CORRELATION';
      confidence = 0.7;
    } else if (avgCorrelation < 0.2) {
      regime = 'LOW_CORRELATION';
      confidence = 0.6;
    }

    return { regime, confidence };
  }

  private calculateAverageCorrelation(correlationMatrix: number[][]): number {
    const flatCorrelations = correlationMatrix
      .flatMap((row, i) => row.slice(i + 1))
      .filter(corr => !isNaN(corr));

    return flatCorrelations.reduce((sum, corr) => sum + Math.abs(corr), 0) / flatCorrelations.length;
  }

  private calculatePortfolioVolatility(returns: number[][]): number {
    if (returns.length === 0) return 0;

    const portfolioReturns = returns.map(period =>
      period.reduce((sum, ret) => sum + ret, 0) / period.length
    );

    return this.calculateStandardDeviation(portfolioReturns);
  }

  private calculateMomentum(returns: number[][]): number {
    if (returns.length < 20) return 0;

    const recentReturns = returns.slice(-20);
    const avgReturn = recentReturns.reduce((sum, period) =>
      sum + period.reduce((pSum, ret) => pSum + ret, 0) / period.length, 0
    ) / recentReturns.length;

    return avgReturn;
  }

  private classifyRegime(
    avgCorrelation: number,
    volatility: number,
    momentum: number
  ): 'BULL' | 'BEAR' | 'VOLATILE' | 'CALM' | 'CRISIS' | 'RECOVERY' {
    if (volatility > 0.03 && avgCorrelation > 0.7) return 'CRISIS';
    if (momentum > 0.01 && volatility < 0.015) return 'BULL';
    if (momentum < -0.01 && volatility > 0.02) return 'BEAR';
    if (volatility > 0.025) return 'VOLATILE';
    if (momentum > 0 && avgCorrelation < 0.5) return 'RECOVERY';
    return 'CALM';
  }

  private calculateRegimeProbabilities(
    avgCorrelation: number,
    volatility: number,
    momentum: number
  ): { [regime: string]: number } {
    // Simplified regime probabilities
    return {
      BULL: Math.max(0, momentum * 10),
      BEAR: Math.max(0, -momentum * 10),
      VOLATILE: volatility * 20,
      CALM: Math.max(0, 1 - volatility * 20),
      CRISIS: Math.max(0, (avgCorrelation - 0.5) * 2 * volatility * 10),
      RECOVERY: Math.max(0, momentum * 5 * (1 - avgCorrelation))
    };
  }

  private detectChangePoints(returns: number[][], windowSize: number): number[] {
    // Simplified change point detection
    const changePoints: number[] = [];
    const currentTime = Date.now() / 1000;

    // Look for significant changes in correlation structure
    for (let i = windowSize; i < returns.length - windowSize; i += windowSize) {
      const before = returns.slice(i - windowSize, i);
      const after = returns.slice(i, i + windowSize);

      const corrBefore = this.calculateCorrelationMatrix(before);
      const corrAfter = this.calculateCorrelationMatrix(after);

      const avgCorrBefore = this.calculateAverageCorrelation(corrBefore);
      const avgCorrAfter = this.calculateAverageCorrelation(corrAfter);

      if (Math.abs(avgCorrBefore - avgCorrAfter) > 0.2) {
        changePoints.push(currentTime - (returns.length - i) * 86400); // Assuming daily data
      }
    }

    return changePoints;
  }

  private predictRegimeTransition(
    currentRegime: string,
    probabilities: { [regime: string]: number }
  ): { mostLikely: string; probability: number; expectedDuration: number } {
    const otherRegimes = Object.entries(probabilities)
      .filter(([regime, _]) => regime !== currentRegime)
      .sort(([, a], [, b]) => b - a);

    return {
      mostLikely: otherRegimes[0]?.[0] || 'NORMAL',
      probability: otherRegimes[0]?.[1] || 0.1,
      expectedDuration: 30 // Simplified expected duration in days
    };
  }

  private calculateConcentrationRisk(weights: number[]): number {
    // Herfindahl-Hirschman Index
    return weights.reduce((sum, weight) => sum + weight ** 2, 0);
  }

  private calculateDiversificationRatio(correlationMatrix: number[][], weights: number[]): number {
    const n = weights.length;
    const weightedAvgCorrelation = correlationMatrix
      .flatMap((row, i) => row.slice(i + 1).map((corr, j) => corr * weights[i] * weights[j + i + 1]))
      .reduce((sum, val) => sum + val, 0);

    const portfolioVariance = 1 + 2 * weightedAvgCorrelation;
    const averageVariance = weights.reduce((sum, weight) => sum + weight ** 2, 0);

    return Math.sqrt(averageVariance) / Math.sqrt(portfolioVariance);
  }

  private calculateEffectiveAssets(correlationMatrix: number[][]): number {
    const n = correlationMatrix.length;
    const avgCorrelation = this.calculateAverageCorrelation(correlationMatrix);
    return 1 + (n - 1) / (1 + (n - 1) * avgCorrelation);
  }

  private calculateMaxDrawdownCorrelation(returns: number[][], correlationMatrix: number[][]): number {
    // Calculate correlation during max drawdown periods
    const portfolioReturns = returns.map(period =>
      period.reduce((sum, ret) => sum + ret, 0) / period.length
    );

    // Find drawdown periods
    let peak = portfolioReturns[0];
    let maxDrawdown = 0;
    let drawdownStart = 0;
    let drawdownEnd = 0;

    portfolioReturns.forEach((ret, i) => {
      if (ret > peak) {
        peak = ret;
      } else {
        const drawdown = (peak - ret) / peak;
        if (drawdown > maxDrawdown) {
          maxDrawdown = drawdown;
          drawdownEnd = i;
          drawdownStart = portfolioReturns.slice(0, i).findLastIndex(r => r === peak);
        }
      }
    });

    // Calculate correlation during max drawdown period
    if (drawdownEnd > drawdownStart) {
      const drawdownReturns = returns.slice(drawdownStart, drawdownEnd + 1);
      const drawdownCorr = this.calculateCorrelationMatrix(drawdownReturns);
      return this.calculateAverageCorrelation(drawdownCorr);
    }

    return this.calculateAverageCorrelation(correlationMatrix);
  }

  private calculateFactorReturns(
    returns: number[][],
    eigenvectors: number[][],
    numFactors: number
  ): number[][] {
    const factorReturns: number[][] = [];

    returns.forEach(periodReturns => {
      const periodFactorReturns: number[] = [];

      for (let f = 0; f < numFactors; f++) {
        const factorReturn = periodReturns.reduce((sum, ret, i) =>
          sum + ret * eigenvectors[f][i], 0
        );
        periodFactorReturns.push(factorReturn);
      }

      factorReturns.push(periodFactorReturns);
    });

    return factorReturns;
  }

  private calculateSpecificRisks(
    returns: number[][],
    factorReturns: number[][],
    eigenvectors: number[][],
    numFactors: number
  ): number[] {
    const numAssets = returns[0]?.length || 0;
    const specificRisks: number[] = [];

    for (let asset = 0; asset < numAssets; asset++) {
      const assetReturns = returns.map(period => period[asset]);
      const explainedReturns = returns.map((_, periodIndex) => {
        let explained = 0;
        for (let f = 0; f < numFactors; f++) {
          explained += factorReturns[periodIndex][f] * eigenvectors[f][asset];
        }
        return explained;
      });

      const residuals = assetReturns.map((ret, i) => ret - explainedReturns[i]);
      const specificRisk = this.calculateStandardDeviation(residuals);
      specificRisks.push(specificRisk);
    }

    return specificRisks;
  }

  private calculateTotalRisk(returns: number[][]): number {
    const portfolioReturns = returns.map(period =>
      period.reduce((sum, ret) => sum + ret, 0) / period.length
    );
    return this.calculateStandardDeviation(portfolioReturns);
  }

  private calculateSystematicRisk(
    factorReturns: number[][],
    eigenvalues: number[],
    numFactors: number
  ): number {
    let systematicVariance = 0;

    for (let f = 0; f < numFactors; f++) {
      const factorVariance = this.calculateStandardDeviation(
        factorReturns.map(period => period[f])
      ) ** 2;
      systematicVariance += eigenvalues[f] * factorVariance;
    }

    return Math.sqrt(systematicVariance);
  }
}