// Service Registry for USDCOP Trading System

import type { IServiceRegistry, ServiceConfig, ServiceHealth } from './types';
import { getLogger } from '../logging';

/**
 * Service registry for tracking all system services
 * Maintains service configurations and current health status
 */
export class ServiceRegistry implements IServiceRegistry {
  private services: Map<string, ServiceConfig> = new Map();
  private statuses: Map<string, ServiceHealth> = new Map();
  private logger = getLogger({ service: 'ServiceRegistry' });

  register(service: ServiceConfig): void {
    if (this.services.has(service.name)) {
      this.logger.warn(`Service already registered: ${service.name}`, {
        service_name: service.name,
      });
      return;
    }

    this.services.set(service.name, service);

    this.logger.info(`Service registered in registry: ${service.name}`, {
      service_name: service.name,
      has_url: !!service.url,
      has_custom_check: !!service.healthCheck,
    });
  }

  unregister(serviceName: string): void {
    this.services.delete(serviceName);
    this.statuses.delete(serviceName);

    this.logger.info(`Service unregistered from registry: ${serviceName}`, {
      service_name: serviceName,
    });
  }

  getService(serviceName: string): ServiceConfig | null {
    return this.services.get(serviceName) || null;
  }

  getAllServices(): ServiceConfig[] {
    return Array.from(this.services.values());
  }

  updateServiceStatus(serviceName: string, health: ServiceHealth): void {
    this.statuses.set(serviceName, health);

    // Log status changes
    const previousStatus = this.statuses.get(serviceName);
    if (previousStatus && previousStatus.status !== health.status) {
      this.logger.warn(
        `Service status changed: ${serviceName}`,
        { service_name: serviceName },
        {
          previous_status: previousStatus.status,
          new_status: health.status,
          latency_ms: health.latency_ms,
        }
      );
    }
  }

  getServiceStatus(serviceName: string): ServiceHealth | null {
    return this.statuses.get(serviceName) || null;
  }

  getAllStatuses(): ServiceHealth[] {
    return Array.from(this.statuses.values());
  }

  /**
   * Get services by status
   */
  getServicesByStatus(status: ServiceHealth['status']): ServiceHealth[] {
    return Array.from(this.statuses.values()).filter((s) => s.status === status);
  }

  /**
   * Get service count
   */
  getServiceCount(): {
    total: number;
    registered: number;
    with_status: number;
  } {
    return {
      total: this.services.size,
      registered: this.services.size,
      with_status: this.statuses.size,
    };
  }

  /**
   * Check if service is registered
   */
  isRegistered(serviceName: string): boolean {
    return this.services.has(serviceName);
  }

  /**
   * Get services with high latency
   */
  getHighLatencyServices(thresholdMs: number = 1000): ServiceHealth[] {
    return Array.from(this.statuses.values()).filter(
      (s) => s.latency_ms > thresholdMs
    );
  }

  /**
   * Get unhealthy services
   */
  getUnhealthyServices(): ServiceHealth[] {
    return this.getServicesByStatus('unhealthy');
  }

  /**
   * Get degraded services
   */
  getDegradedServices(): ServiceHealth[] {
    return this.getServicesByStatus('degraded');
  }

  /**
   * Clear all statuses
   */
  clearStatuses(): void {
    this.logger.info('Clearing all service statuses', undefined, {
      statuses_cleared: this.statuses.size,
    });

    this.statuses.clear();
  }

  /**
   * Export registry data
   */
  export(): {
    services: ServiceConfig[];
    statuses: ServiceHealth[];
  } {
    return {
      services: this.getAllServices(),
      statuses: this.getAllStatuses(),
    };
  }

  /**
   * Get registry summary
   */
  getSummary(): {
    total_services: number;
    services_with_status: number;
    healthy: number;
    degraded: number;
    unhealthy: number;
    avg_latency_ms: number;
  } {
    const statuses = this.getAllStatuses();

    const summary = {
      total_services: this.services.size,
      services_with_status: statuses.length,
      healthy: 0,
      degraded: 0,
      unhealthy: 0,
      avg_latency_ms: 0,
    };

    if (statuses.length === 0) return summary;

    let totalLatency = 0;

    for (const status of statuses) {
      switch (status.status) {
        case 'healthy':
          summary.healthy++;
          break;
        case 'degraded':
          summary.degraded++;
          break;
        case 'unhealthy':
          summary.unhealthy++;
          break;
      }

      totalLatency += status.latency_ms;
    }

    summary.avg_latency_ms = totalLatency / statuses.length;

    return summary;
  }
}

// Export singleton instance
let serviceRegistryInstance: ServiceRegistry | null = null;

export const getServiceRegistry = (): ServiceRegistry => {
  if (!serviceRegistryInstance) {
    serviceRegistryInstance = new ServiceRegistry();
  }
  return serviceRegistryInstance;
};
