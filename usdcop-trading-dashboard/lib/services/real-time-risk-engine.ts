export interface RiskMetrics {
  var: number
  exposure: number
  leverage: number
  positions: number
}

export async function getRiskMetrics(): Promise<RiskMetrics> {
  return {
    var: 0.05,
    exposure: 0.75,
    leverage: 2.5,
    positions: 15
  }
}
