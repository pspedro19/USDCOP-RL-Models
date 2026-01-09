// Health Monitoring Types for USDCOP Trading System

export type ServiceStatus = 'healthy' | 'degraded' | 'unhealthy';

export interface ServiceHealth {
  service: string;
  status: ServiceStatus;
  latency_ms: number;
  last_check: string;
  details: {
    uptime: number;
    requests_per_minute: number;
    error_rate: number;
    memory_usage_mb: number;
  };
  metadata?: Record<string, any>;
}

export interface HealthCheckResult {
  success: boolean;
  latency_ms: number;
  error?: string;
  metadata?: Record<string, any>;
}

export interface ServiceConfig {
  name: string;
  url?: string;
  checkInterval?: number; // milliseconds
  timeout?: number; // milliseconds
  healthCheck?: () => Promise<HealthCheckResult>;
}

export interface LatencyMeasurement {
  timestamp: string;
  service: string;
  operation: string;
  latency_ms: number;
  success: boolean;
  metadata?: Record<string, any>;
}

export interface LatencyStats {
  service: string;
  operation: string;
  count: number;
  avg_ms: number;
  min_ms: number;
  max_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  success_rate: number;
}

export interface SystemHealth {
  overall_status: ServiceStatus;
  timestamp: string;
  services: ServiceHealth[];
  summary: {
    total_services: number;
    healthy_services: number;
    degraded_services: number;
    unhealthy_services: number;
  };
}

export interface IHealthChecker {
  registerService(config: ServiceConfig): void;
  unregisterService(serviceName: string): void;
  checkService(serviceName: string): Promise<ServiceHealth>;
  checkAllServices(): Promise<SystemHealth>;
  getServiceHealth(serviceName: string): ServiceHealth | null;
  getSystemHealth(): SystemHealth;
  startMonitoring(): void;
  stopMonitoring(): void;
}

export interface ILatencyMonitor {
  recordLatency(
    service: string,
    operation: string,
    latencyMs: number,
    success: boolean,
    metadata?: Record<string, any>
  ): void;
  getLatencyStats(service: string, operation?: string): LatencyStats | LatencyStats[];
  getRecentMeasurements(limit?: number): LatencyMeasurement[];
  clearMeasurements(service?: string): void;
  getAverageLatency(service: string, minutes?: number): number;
}

export interface IServiceRegistry {
  register(service: ServiceConfig): void;
  unregister(serviceName: string): void;
  getService(serviceName: string): ServiceConfig | null;
  getAllServices(): ServiceConfig[];
  updateServiceStatus(serviceName: string, health: ServiceHealth): void;
  getServiceStatus(serviceName: string): ServiceHealth | null;
  getAllStatuses(): ServiceHealth[];
}
