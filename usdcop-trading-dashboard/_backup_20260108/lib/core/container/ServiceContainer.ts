/**
 * ServiceContainer
 * ================
 *
 * Simple dependency injection container using registry pattern.
 */

/**
 * Service factory function
 */
export type ServiceFactory<T> = () => T;

/**
 * Service lifecycle
 */
export enum ServiceLifetime {
  /**
   * Single instance shared across all consumers
   */
  SINGLETON = 'singleton',

  /**
   * New instance created each time
   */
  TRANSIENT = 'transient',

  /**
   * Scoped to a specific context (not implemented yet)
   */
  SCOPED = 'scoped',
}

/**
 * Service registration entry
 */
interface ServiceRegistration<T = unknown> {
  factory: ServiceFactory<T>;
  lifetime: ServiceLifetime;
  instance?: T;
}

/**
 * Simple service container for dependency injection
 */
export class ServiceContainer {
  private services = new Map<string, ServiceRegistration>();
  private parent?: ServiceContainer;

  constructor(parent?: ServiceContainer) {
    this.parent = parent;
  }

  /**
   * Register a service with a factory function
   */
  register<T>(
    key: string,
    factory: ServiceFactory<T>,
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
  ): void {
    if (this.services.has(key)) {
      console.warn(`Service '${key}' is already registered. Overwriting.`);
    }

    this.services.set(key, {
      factory: factory as ServiceFactory<unknown>,
      lifetime,
    });
  }

  /**
   * Register a service instance directly (singleton)
   */
  registerInstance<T>(key: string, instance: T): void {
    this.services.set(key, {
      factory: () => instance,
      lifetime: ServiceLifetime.SINGLETON,
      instance: instance as unknown,
    });
  }

  /**
   * Register a singleton service
   */
  registerSingleton<T>(key: string, factory: ServiceFactory<T>): void {
    this.register(key, factory, ServiceLifetime.SINGLETON);
  }

  /**
   * Register a transient service
   */
  registerTransient<T>(key: string, factory: ServiceFactory<T>): void {
    this.register(key, factory, ServiceLifetime.TRANSIENT);
  }

  /**
   * Get a service by key
   */
  get<T>(key: string): T {
    const registration = this.services.get(key);

    if (!registration) {
      // Try parent container if available
      if (this.parent) {
        return this.parent.get<T>(key);
      }
      throw new Error(`Service '${key}' not found in container`);
    }

    // Return existing singleton instance
    if (
      registration.lifetime === ServiceLifetime.SINGLETON &&
      registration.instance !== undefined
    ) {
      return registration.instance as T;
    }

    // Create new instance
    const instance = registration.factory();

    // Store singleton instance for future use
    if (registration.lifetime === ServiceLifetime.SINGLETON) {
      registration.instance = instance;
    }

    return instance as T;
  }

  /**
   * Try to get a service, returns undefined if not found
   */
  tryGet<T>(key: string): T | undefined {
    try {
      return this.get<T>(key);
    } catch {
      return undefined;
    }
  }

  /**
   * Check if a service is registered
   */
  has(key: string): boolean {
    return this.services.has(key) || (this.parent?.has(key) ?? false);
  }

  /**
   * Remove a service registration
   */
  unregister(key: string): boolean {
    return this.services.delete(key);
  }

  /**
   * Clear all service registrations
   */
  clear(): void {
    this.services.clear();
  }

  /**
   * Get all registered service keys
   */
  getKeys(): string[] {
    const keys = Array.from(this.services.keys());
    if (this.parent) {
      return [...keys, ...this.parent.getKeys()];
    }
    return keys;
  }

  /**
   * Create a child container
   */
  createChild(): ServiceContainer {
    return new ServiceContainer(this);
  }

  /**
   * Get service registration info
   */
  getRegistration(key: string): { lifetime: ServiceLifetime; hasInstance: boolean } | undefined {
    const registration = this.services.get(key);
    if (!registration) {
      return this.parent?.getRegistration(key);
    }

    return {
      lifetime: registration.lifetime,
      hasInstance: registration.instance !== undefined,
    };
  }

  /**
   * Dispose all singleton instances that have a dispose method
   */
  dispose(): void {
    for (const [key, registration] of this.services.entries()) {
      if (
        registration.lifetime === ServiceLifetime.SINGLETON &&
        registration.instance
      ) {
        const instance = registration.instance as any;
        if (typeof instance.dispose === 'function') {
          try {
            instance.dispose();
          } catch (error) {
            console.error(`Error disposing service '${key}':`, error);
          }
        }
      }
    }
    this.clear();
  }
}

/**
 * Global service container instance
 */
let globalContainer: ServiceContainer | null = null;

/**
 * Get or create the global service container
 */
export function getGlobalContainer(): ServiceContainer {
  if (!globalContainer) {
    globalContainer = new ServiceContainer();
  }
  return globalContainer;
}

/**
 * Reset the global container (useful for testing)
 */
export function resetGlobalContainer(): void {
  if (globalContainer) {
    globalContainer.dispose();
  }
  globalContainer = new ServiceContainer();
}

/**
 * Service key constants for type-safe registration
 */
export const ServiceKeys = {
  // Data providers
  DATA_PROVIDER: 'dataProvider',
  EXTENDED_DATA_PROVIDER: 'extendedDataProvider',

  // WebSocket providers
  WEBSOCKET_PROVIDER: 'websocketProvider',
  EXTENDED_WEBSOCKET_PROVIDER: 'extendedWebsocketProvider',

  // Risk calculators
  RISK_CALCULATOR: 'riskCalculator',
  ADVANCED_RISK_CALCULATOR: 'advancedRiskCalculator',

  // Utilities
  LOGGER: 'logger',
  HTTP_CLIENT: 'httpClient',
  CACHE: 'cache',
  EVENT_BUS: 'eventBus',
} as const;

/**
 * Type-safe service key type
 */
export type ServiceKey = typeof ServiceKeys[keyof typeof ServiceKeys];
