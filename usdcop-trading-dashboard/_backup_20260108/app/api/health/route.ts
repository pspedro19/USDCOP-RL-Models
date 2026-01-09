import { createApiResponse } from '@/lib/types/api';
import { getCircuitBreakerStatus } from '@/lib/utils/circuit-breaker';

export async function GET() {
  const startTime = Date.now();

  // Get circuit breaker status for monitoring
  const circuitBreakerStatus = getCircuitBreakerStatus();

  // Determine overall health based on circuit breaker states
  const hasOpenCircuits = Object.values(circuitBreakerStatus).some(
    (cb) => cb.state === 'OPEN'
  );

  return Response.json(
    createApiResponse(true, 'live', {
      data: {
        status: hasOpenCircuits ? 'degraded' : 'healthy',
        service: 'USDCOP Trading Dashboard',
        circuitBreakers: circuitBreakerStatus,
        timestamp: new Date().toISOString(),
      },
      latency: Date.now() - startTime,
    })
  );
}