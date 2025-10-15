export interface PipelineDataPoint {
  timestamp: number
  value: number
  layer: string
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
  }
}
