// Health Checker for USDCOP Trading System

import type {
  IHealthChecker,
  ServiceConfig,
  ServiceHealth,
  SystemHealth,
  HealthCheckResult,
  ServiceStatus,
} from './types';
import { getLogger } from '../logging';
import { ServiceRegistry } from './ServiceRegistry';

/**
 * Health checker for monitoring service health
 * Performs periodic health checks and tracks service status
 */
export class HealthChecker implements IHealthChecker {
  private registry: ServiceRegistry;
  private logger = getLogger({ service: 'HealthChecker' });
  private monitoringIntervals: Map<string, NodeJS.Timeout> = new Map();
  private serviceMetrics: Map<
    string,
    { requests: number; errors: number; startTime: number }
  > = new Map();

  constructor(registry?: ServiceRegistry) {
    this.registry = registry || new ServiceRegistry();
  }

  registerService(config: ServiceConfig): void {
    this.registry.register(config);
    this.initializeMetrics(config.name);

    this.logger.info(`Service registered: ${config.name}`, {
      service_name: config.name,
    });
  }

  unregisterService(serviceName: string): void {
    this.stopServiceMonitoring(serviceName);
    this.registry.unregister(serviceName);
    this.serviceMetrics.delete(serviceName);

    this.logger.info(`Service unregistered: ${serviceName}`, {
      service_name: serviceName,
    });
  }

  private initializeMetrics(serviceName: string): void {
    this.serviceMetrics.set(serviceName, {
      requests: 0,
      errors: 0,
      startTime: Date.now(),
    });
  }

  async checkService(serviceName: string): Promise<ServiceHealth> {
    const service = this.registry.getService(serviceName);

    if (!service) {
      throw new Error(`Service not found: ${serviceName}`);
    }

    const metrics = this.serviceMetrics.get(serviceName);
    if (!metrics) {
      this.initializeMetrics(serviceName);
    }

    const startTime = Date.now();
    let result: HealthCheckResult;

    try {
      if (service.healthCheck) {
        // Use custom health check
        result = await Promise.race([
          service.healthCheck(),
          this.timeout(service.timeout || 5000),
        ]);
      } else if (service.url) {
        // Use default HTTP health check
        result = await this.defaultHealthCheck(service.url, service.timeout);
      } else {
        // No health check available
        result = {
          success: true,
          latency_ms: 0,
          metadata: { type: 'no_check' },
        };
      }

      const latencyMs = Date.now() - startTime;

      // Update metrics
      const currentMetrics = this.serviceMetrics.get(serviceName)!;
      currentMetrics.requests++;

      if (!result.success) {
        currentMetrics.errors++;
      }

      // Calculate status
      const status = this.calculateStatus(result, latencyMs);

      // Calculate uptime
      const uptime = Math.floor((Date.now() - currentMetrics.startTime) / 1000);

      // Calculate error rate
      const errorRate =
        currentMetrics.requests > 0
          ? currentMetrics.errors / currentMetrics.requests
          : 0;

      // Calculate requests per minute
      const requestsPerMinute =
        uptime > 0 ? (currentMetrics.requests / uptime) * 60 : 0;

      const health: ServiceHealth = {
        service: serviceName,
        status,
        latency_ms: result.latency_ms || latencyMs,
        last_check: new Date().toISOString(),
        details: {
          uptime,
          requests_per_minute: requestsPerMinute,
          error_rate: errorRate,
          memory_usage_mb: this.getMemoryUsage(),
        },
        metadata: result.metadata,
      };

      this.registry.updateServiceStatus(serviceName, health);

      return health;
    } catch (error) {
      const latencyMs = Date.now() - startTime;

      const currentMetrics = this.serviceMetrics.get(serviceName)!;
      currentMetrics.requests++;
      currentMetrics.errors++;

      const health: ServiceHealth = {
        service: serviceName,
        status: 'unhealthy',
        latency_ms: latencyMs,
        last_check: new Date().toISOString(),
        details: {
          uptime: Math.floor((Date.now() - currentMetrics.startTime) / 1000),
          requests_per_minute: 0,
          error_rate: currentMetrics.errors / currentMetrics.requests,
          memory_usage_mb: this.getMemoryUsage(),
        },
        metadata: {
          error: (error as Error).message,
        },
      };

      this.registry.updateServiceStatus(serviceName, health);

      return health;
    }
  }

  async checkAllServices(): Promise<SystemHealth> {
    const services = this.registry.getAllServices();
    const healthChecks = await Promise.allSettled(
      services.map((s) => this.checkService(s.name))
    );

    const serviceHealths: ServiceHealth[] = healthChecks
      .filter((result) => result.status === 'fulfilled')
      .map((result) => (result as PromiseFulfilledResult<ServiceHealth>).value);

    const summary = {
      total_services: serviceHealths.length,
      healthy_services: serviceHealths.filter((s) => s.status === 'healthy').length,
      degraded_services: serviceHealths.filter((s) => s.status === 'degraded')
        .length,
      unhealthy_services: serviceHealths.filter((s) => s.status === 'unhealthy')
        .length,
    };

    const overallStatus: ServiceStatus =
      summary.unhealthy_services > 0
        ? 'unhealthy'
        : summary.degraded_services > 0
        ? 'degraded'
        : 'healthy';

    return {
      overall_status: overallStatus,
      timestamp: new Date().toISOString(),
      services: serviceHealths,
      summary,
    };
  }

  getServiceHealth(serviceName: string): ServiceHealth | null {
    return this.registry.getServiceStatus(serviceName);
  }

  getSystemHealth(): SystemHealth {
    const services = this.registry.getAllStatuses();

    const summary = {
      total_services: services.length,
      healthy_services: services.filter((s) => s.status === 'healthy').length,
      degraded_services: services.filter((s) => s.status === 'degraded').length,
      unhealthy_services: services.filter((s) => s.status === 'unhealthy').length,
    };

    const overallStatus: ServiceStatus =
      summary.unhealthy_services > 0
        ? 'unhealthy'
        : summary.degraded_services > 0
        ? 'degraded'
        : 'healthy';

    return {
      overall_status: overallStatus,
      timestamp: new Date().toISOString(),
      services,
      summary,
    };
  }

  startMonitoring(): void {
    const services = this.registry.getAllServices();

    for (const service of services) {
      this.startServiceMonitoring(service);
    }

    this.logger.info('Health monitoring started', undefined, {
      services_count: services.length,
    });
  }

  stopMonitoring(): void {
    for (const [serviceName, interval] of this.monitoringIntervals.entries()) {
      clearInterval(interval);
      this.logger.debug(`Stopped monitoring: ${serviceName}`);
    }

    this.monitoringIntervals.clear();
    this.logger.info('Health monitoring stopped');
  }

  private startServiceMonitoring(service: ServiceConfig): void {
    const interval = service.checkInterval || 30000; // Default 30 seconds

    const intervalId = setInterval(async () => {
      try {
        await this.checkService(service.name);
      } catch (error) {
        this.logger.error(
          `Health check failed for ${service.name}`,
          error as Error,
          { service_name: service.name }
        );
      }
    }, interval);

    this.monitoringIntervals.set(service.name, intervalId);
    this.logger.debug(`Started monitoring: ${service.name}`, {
      interval_ms: interval,
    });
  }

  private stopServiceMonitoring(serviceName: string): void {
    const interval = this.monitoringIntervals.get(serviceName);

    if (interval) {
      clearInterval(interval);
      this.monitoringIntervals.delete(serviceName);
    }
  }

  private calculateStatus(
    result: HealthCheckResult,
    latencyMs: number
  ): ServiceStatus {
    if (!result.success) {
      return 'unhealthy';
    }

    // Consider degraded if latency is high
    if (latencyMs > 1000) {
      return 'degraded';
    }

    if (latencyMs > 500) {
      return 'degraded';
    }

    return 'healthy';
  }

  private async defaultHealthCheck(
    url: string,
    timeout: number = 5000
  ): Promise<HealthCheckResult> {
    const startTime = Date.now();

    try {
      const response = await Promise.race([
        fetch(url),
        this.timeout(timeout),
      ]);

      const latencyMs = Date.now() - startTime;

      return {
        success: response.ok,
        latency_ms: latencyMs,
        metadata: {
          status: response.status,
          url,
        },
      };
    } catch (error) {
      return {
        success: false,
        latency_ms: Date.now() - startTime,
        error: (error as Error).message,
      };
    }
  }

  private timeout(ms: number): Promise<never> {
    return new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Health check timeout')), ms)
    );
  }

  private getMemoryUsage(): number {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      return process.memoryUsage().heapUsed / (1024 * 1024);
    }
    return 0;
  }
}

// Export singleton instance
let healthCheckerInstance: HealthChecker | null = null;

export const getHealthChecker = (): HealthChecker => {
  if (!healthCheckerInstance) {
    healthCheckerInstance = new HealthChecker();
  }
  return healthCheckerInstance;
};
