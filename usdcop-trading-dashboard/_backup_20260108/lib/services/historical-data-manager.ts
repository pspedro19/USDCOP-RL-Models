/**
 * Historical Data Manager
 * Optimized service for handling 92k+ historical records with smart caching
 * and progressive loading for smooth navigation through 2020-2025 data
 */

interface MarketDataPoint {
  timestamp: string;
  symbol: string;
  price: number;
  bid?: number;
  ask?: number;
  volume?: number;
  source: string;
}

interface CandlestickData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  datetime: string;
}

interface DataRange {
  startDate: Date;
  endDate: Date;
  timeframe: string;
  limit?: number;
}

class HistoricalDataManager {
  private cache = new Map<string, CandlestickData[]>();
  private maxCacheSize = 50; // Maximum cached chunks
  private baseUrl = '/api/proxy/trading';

  // Date constants for USDCOP data
  public readonly MIN_DATE = new Date('2020-01-02T07:30:00Z');
  public readonly MAX_DATE = new Date('2025-10-10T18:55:00Z');

  /**
   * Get cache key for data chunk
   */
  private getCacheKey(range: DataRange): string {
    return `${range.timeframe}_${range.startDate.getTime()}_${range.endDate.getTime()}_${range.limit || 'all'}`;
  }

  /**
   * Clear old cache entries when limit reached
   */
  private manageCacheSize(): void {
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
  }

  /**
   * Fetch candlestick data from API with smart parameters
   */
  private async fetchFromAPI(range: DataRange): Promise<CandlestickData[]> {
    const params = new URLSearchParams({
      timeframe: range.timeframe,
      start_date: range.startDate.toISOString().split('T')[0],
      end_date: range.endDate.toISOString().split('T')[0],
      include_indicators: 'true'
    });

    if (range.limit) {
      params.append('limit', range.limit.toString());
    }

    const response = await fetch(`${this.baseUrl}/candlesticks/USDCOP?${params}`);

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    if (!data.data || !Array.isArray(data.data)) {
      throw new Error('Invalid API response format');
    }

    return data.data.map((item: any) => ({
      time: item.time,
      open: item.open || item.price,
      high: item.high || item.price,
      low: item.low || item.price,
      close: item.close || item.price,
      volume: item.volume || 0,
      datetime: new Date(item.time).toISOString(),
      // Include technical indicators if available
      ...item
    }));
  }

  /**
   * Get optimal chunk size based on timeframe
   */
  private getOptimalChunkSize(timeframe: string): number {
    switch (timeframe) {
      case '5m': return 2000;   // ~7 days of 5min data
      case '15m': return 1500;  // ~15 days of 15min data
      case '1h': return 1000;   // ~40 days of hourly data
      case '4h': return 500;    // ~80 days of 4h data
      case '1d': return 365;    // 1 year of daily data
      case '1w': return 100;    // ~2 years of weekly data
      case '1M': return 60;     // 5 years of monthly data
      default: return 1000;
    }
  }

  /**
   * Calculate intelligent date ranges for progressive loading
   */
  private calculateDateRanges(
    startDate: Date,
    endDate: Date,
    timeframe: string,
    maxPoints: number = 5000
  ): DataRange[] {
    const ranges: DataRange[] = [];
    const chunkSize = this.getOptimalChunkSize(timeframe);

    // Calculate time span per chunk based on timeframe
    let timeSpanPerChunk: number;
    switch (timeframe) {
      case '5m': timeSpanPerChunk = 5 * 60 * 1000 * chunkSize; break;
      case '15m': timeSpanPerChunk = 15 * 60 * 1000 * chunkSize; break;
      case '1h': timeSpanPerChunk = 60 * 60 * 1000 * chunkSize; break;
      case '4h': timeSpanPerChunk = 4 * 60 * 60 * 1000 * chunkSize; break;
      case '1d': timeSpanPerChunk = 24 * 60 * 60 * 1000 * chunkSize; break;
      case '1w': timeSpanPerChunk = 7 * 24 * 60 * 60 * 1000 * chunkSize; break;
      case '1M': timeSpanPerChunk = 30 * 24 * 60 * 60 * 1000 * chunkSize; break;
      default: timeSpanPerChunk = 24 * 60 * 60 * 1000 * chunkSize;
    }

    let currentStart = new Date(startDate);
    while (currentStart < endDate) {
      const currentEnd = new Date(Math.min(
        currentStart.getTime() + timeSpanPerChunk,
        endDate.getTime()
      ));

      ranges.push({
        startDate: new Date(currentStart),
        endDate: new Date(currentEnd),
        timeframe,
        limit: chunkSize
      });

      currentStart = new Date(currentEnd.getTime() + 1);
    }

    return ranges;
  }

  /**
   * Load data for specific time range with caching
   */
  async loadDataRange(range: DataRange): Promise<CandlestickData[]> {
    const cacheKey = this.getCacheKey(range);

    // Check cache first
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey)!;
    }

    try {
      const data = await this.fetchFromAPI(range);

      // Cache the result
      this.manageCacheSize();
      this.cache.set(cacheKey, data);

      return data;
    } catch (error) {
      console.error('Error loading data range:', error);
      throw error;
    }
  }

  /**
   * Load data progressively for smooth navigation
   */
  async loadDataProgressive(
    startDate: Date,
    endDate: Date,
    timeframe: string,
    onProgress?: (loaded: number, total: number) => void
  ): Promise<CandlestickData[]> {
    const ranges = this.calculateDateRanges(startDate, endDate, timeframe);
    const allData: CandlestickData[] = [];

    for (let i = 0; i < ranges.length; i++) {
      try {
        const chunkData = await this.loadDataRange(ranges[i]);
        allData.push(...chunkData);

        onProgress?.(i + 1, ranges.length);

        // Add small delay to prevent overwhelming the API
        if (i < ranges.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 50));
        }
      } catch (error) {
        console.warn(`Failed to load chunk ${i + 1}/${ranges.length}:`, error);
        // Continue with other chunks
      }
    }

    // Sort by timestamp to ensure proper order
    return allData.sort((a, b) => a.time - b.time);
  }

  /**
   * Get data for current viewport with smart loading
   */
  async getViewportData(
    centerDate: Date,
    timeframe: string,
    viewportSize: number = 1000
  ): Promise<CandlestickData[]> {
    // Calculate viewport range
    const timespan = this.getTimespan(timeframe, viewportSize);
    const startDate = new Date(centerDate.getTime() - timespan / 2);
    const endDate = new Date(centerDate.getTime() + timespan / 2);

    // Ensure dates are within bounds
    const clampedStart = new Date(Math.max(startDate.getTime(), this.MIN_DATE.getTime()));
    const clampedEnd = new Date(Math.min(endDate.getTime(), this.MAX_DATE.getTime()));

    return this.loadDataRange({
      startDate: clampedStart,
      endDate: clampedEnd,
      timeframe,
      limit: viewportSize
    });
  }

  /**
   * Calculate timespan for given number of data points
   */
  private getTimespan(timeframe: string, pointCount: number): number {
    const multipliers = {
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000,
      '1w': 7 * 24 * 60 * 60 * 1000,
      '1M': 30 * 24 * 60 * 60 * 1000
    };

    return (multipliers[timeframe as keyof typeof multipliers] || multipliers['1d']) * pointCount;
  }

  /**
   * Get latest data for real-time updates
   */
  async getLatestData(timeframe: string, limit: number = 100): Promise<CandlestickData[]> {
    const now = new Date();
    const startDate = new Date(now.getTime() - this.getTimespan(timeframe, limit));

    return this.loadDataRange({
      startDate,
      endDate: now,
      timeframe,
      limit
    });
  }

  /**
   * Get data summary for overview visualization
   */
  async getDataSummary(): Promise<{
    totalRecords: number;
    dateRange: { min: Date; max: Date };
    availableTimeframes: string[];
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const health = await response.json();

      return {
        totalRecords: health.total_records || 0,
        dateRange: {
          min: this.MIN_DATE,
          max: this.MAX_DATE
        },
        availableTimeframes: ['5m', '15m', '1h', '4h', '1d', '1w', '1M']
      };
    } catch (error) {
      console.error('Error getting data summary:', error);
      return {
        totalRecords: 0,
        dateRange: { min: this.MIN_DATE, max: this.MAX_DATE },
        availableTimeframes: ['5m', '15m', '1h', '4h', '1d', '1w', '1M']
      };
    }
  }

  /**
   * Preload adjacent data for smooth navigation
   */
  async preloadAdjacentData(
    currentDate: Date,
    timeframe: string,
    direction: 'forward' | 'backward' | 'both' = 'both'
  ): Promise<void> {
    const viewportSize = this.getOptimalChunkSize(timeframe);
    const timespan = this.getTimespan(timeframe, viewportSize);

    const promises: Promise<any>[] = [];

    if (direction === 'forward' || direction === 'both') {
      const forwardStart = new Date(currentDate.getTime() + timespan);
      const forwardEnd = new Date(forwardStart.getTime() + timespan);

      if (forwardEnd <= this.MAX_DATE) {
        promises.push(this.loadDataRange({
          startDate: forwardStart,
          endDate: forwardEnd,
          timeframe,
          limit: viewportSize
        }));
      }
    }

    if (direction === 'backward' || direction === 'both') {
      const backwardEnd = new Date(currentDate.getTime() - timespan);
      const backwardStart = new Date(backwardEnd.getTime() - timespan);

      if (backwardStart >= this.MIN_DATE) {
        promises.push(this.loadDataRange({
          startDate: backwardStart,
          endDate: backwardEnd,
          timeframe,
          limit: viewportSize
        }));
      }
    }

    // Execute preloading in background
    Promise.all(promises).catch(error => {
      console.warn('Preloading failed:', error);
    });
  }

  /**
   * Get data for a specific date range (main method for navigation)
   */
  async getDataForRange(
    startDate: Date,
    endDate: Date,
    timeframe: string,
    limit?: number
  ): Promise<CandlestickData[]> {
    const range: DataRange = {
      startDate,
      endDate,
      timeframe,
      limit
    };

    return this.loadDataRange(range);
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; maxSize: number; keys: string[] } {
    return {
      size: this.cache.size,
      maxSize: this.maxCacheSize,
      keys: Array.from(this.cache.keys())
    };
  }
}

// Create and export singleton instance
export const historicalDataManager = new HistoricalDataManager();
export default historicalDataManager;