export interface PipelineDataPoint {
  timestamp: number
  value: number
  layer: string
}

export interface CandleData {
  datetime: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  timestamp: number
}

export async function getPipelineData(): Promise<PipelineDataPoint[]> {
  return [
    { timestamp: Date.now(), value: 100, layer: 'L0' },
    { timestamp: Date.now() - 1000, value: 95, layer: 'L1' }
  ]
}

export const pipelineDataService = {
  getPipelineData,

  async getHealthStatus() {
    return { status: 'healthy', timestamp: Date.now() }
  },

  async getMetrics() {
    return { processed: 1000, errors: 0, latency: 45 }
  },

  /**
   * Get real-time update - fetches latest 5-minute bar
   */
  async getRealtimeUpdate(): Promise<CandleData | null> {
    try {
      const response = await fetch('/api/pipeline/l0/raw-data?limit=1', {
        credentials: 'include',  // Include session cookies for authentication
      });
      if (!response.ok) {
        console.error(`[Pipeline Data Client] HTTP ${response.status}`);
        return null;
      }

      const result = await response.json();

      // API response format: { success, data: { count, data: [...], metadata, pagination }, metadata }
      if (result.success && result.data && result.data.data && result.data.data.length > 0) {
        const latest = result.data.data[0];
        const timeValue = latest.timestamp || latest.time;
        return {
          datetime: timeValue,
          open: Number(latest.open),
          high: Number(latest.high),
          low: Number(latest.low),
          close: Number(latest.close),
          volume: Number(latest.volume || 0),
          timestamp: new Date(timeValue).getTime()
        };
      }

      return null;
    } catch (error) {
      console.error('[Pipeline Data Client] Error in getRealtimeUpdate:', error);
      return null;
    }
  },

  /**
   * Load historical L0 data with date range
   * @param startDate - Start date for data range
   * @param endDate - End date for data range
   * @param mode - 'align' for aligned data, 'raw' for raw data
   */
  async loadL0Data(
    startDate: Date,
    endDate: Date,
    mode: 'align' | 'raw' = 'align'
  ): Promise<CandleData[]> {
    try {
      // Format dates to ISO
      const start = startDate.toISOString();
      const end = endDate.toISOString();

      console.log(`[Pipeline Data Client] Loading L0 data from ${start} to ${end}`);

      // Fetch data from L0 raw-data endpoint (reasonable limit for performance)
      const response = await fetch(
        `/api/pipeline/l0/raw-data?start_date=${start}&end_date=${end}&limit=5000`,
        {
          credentials: 'include',  // Include session cookies for authentication
        }
      );

      if (!response.ok) {
        console.error(`[Pipeline Data Client] HTTP ${response.status}`);
        return [];
      }

      const result = await response.json();

      // API response format: { success, data: { count, data: [...], metadata, pagination }, metadata }
      if (result.success && result.data && result.data.data) {
        // Transform to CandleData format - note: actual array is in result.data.data
        return result.data.data.map((item: any) => {
          const timeValue = item.timestamp || item.time;
          return {
            datetime: timeValue,
            open: Number(item.open),
            high: Number(item.high),
            low: Number(item.low),
            close: Number(item.close),
            volume: Number(item.volume || 0),
            timestamp: new Date(timeValue).getTime()
          };
        }).sort((a: CandleData, b: CandleData) => a.timestamp - b.timestamp);
      }

      return [];
    } catch (error) {
      console.error('[Pipeline Data Client] Error in loadL0Data:', error);
      return [];
    }
  }
}
