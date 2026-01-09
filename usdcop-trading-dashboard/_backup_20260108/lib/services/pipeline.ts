export interface PipelineData {
  layer: string
  status: string
  records: number
  lastUpdate: string
}

export async function getPipelineStatus(): Promise<PipelineData[]> {
  return [
    { layer: 'L0-Acquire', status: 'running', records: 1000, lastUpdate: new Date().toISOString() },
    { layer: 'L1-Standardize', status: 'completed', records: 950, lastUpdate: new Date().toISOString() },
    { layer: 'L2-Prepare', status: 'pending', records: 0, lastUpdate: new Date().toISOString() }
  ]
}

export async function fetchLatestPipelineOutput(): Promise<any> {
  return {
    status: 'success',
    data: [],
    lastUpdate: new Date().toISOString()
  }
}
