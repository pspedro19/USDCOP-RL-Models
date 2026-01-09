/**
 * ServiceContainer Tests
 * =======================
 *
 * Unit tests for the dependency injection container.
 */

import {
  ServiceContainer,
  ServiceLifetime,
  ServiceKeys,
  getGlobalContainer,
  resetGlobalContainer,
} from '../container';

describe('ServiceContainer', () => {
  let container: ServiceContainer;

  beforeEach(() => {
    container = new ServiceContainer();
    resetGlobalContainer();
  });

  describe('Basic Registration and Resolution', () => {
    it('should register and resolve a service', () => {
      const service = { name: 'test' };
      container.registerInstance('test', service);

      const resolved = container.get<typeof service>('test');
      expect(resolved).toBe(service);
    });

    it('should throw error when service not found', () => {
      expect(() => container.get('nonexistent')).toThrow(
        "Service 'nonexistent' not found"
      );
    });

    it('should return undefined for tryGet when service not found', () => {
      const result = container.tryGet('nonexistent');
      expect(result).toBeUndefined();
    });

    it('should check if service exists', () => {
      container.registerInstance('test', { name: 'test' });

      expect(container.has('test')).toBe(true);
      expect(container.has('nonexistent')).toBe(false);
    });
  });

  describe('Singleton Lifetime', () => {
    it('should return same instance for singleton', () => {
      let counter = 0;
      container.registerSingleton('counter', () => ({
        id: ++counter,
      }));

      const instance1 = container.get<{ id: number }>('counter');
      const instance2 = container.get<{ id: number }>('counter');

      expect(instance1).toBe(instance2);
      expect(instance1.id).toBe(1);
      expect(counter).toBe(1); // Factory called only once
    });

    it('should create singleton lazily', () => {
      let created = false;
      container.registerSingleton('lazy', () => {
        created = true;
        return { value: 'lazy' };
      });

      expect(created).toBe(false);

      container.get('lazy');
      expect(created).toBe(true);
    });
  });

  describe('Transient Lifetime', () => {
    it('should return new instance for transient', () => {
      let counter = 0;
      container.registerTransient('counter', () => ({
        id: ++counter,
      }));

      const instance1 = container.get<{ id: number }>('counter');
      const instance2 = container.get<{ id: number }>('counter');

      expect(instance1).not.toBe(instance2);
      expect(instance1.id).toBe(1);
      expect(instance2.id).toBe(2);
      expect(counter).toBe(2); // Factory called twice
    });
  });

  describe('Service Overwriting', () => {
    it('should warn when overwriting service', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation();

      container.registerInstance('test', { version: 1 });
      container.registerInstance('test', { version: 2 });

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("Service 'test' is already registered")
      );

      const resolved = container.get<{ version: number }>('test');
      expect(resolved.version).toBe(2);

      warnSpy.mockRestore();
    });
  });

  describe('Service Keys', () => {
    it('should use predefined service keys', () => {
      const dataProvider = { type: 'data-provider' };
      container.registerInstance(ServiceKeys.DATA_PROVIDER, dataProvider);

      const resolved = container.get(ServiceKeys.DATA_PROVIDER);
      expect(resolved).toBe(dataProvider);
    });

    it('should have all expected service keys', () => {
      expect(ServiceKeys.DATA_PROVIDER).toBe('dataProvider');
      expect(ServiceKeys.WEBSOCKET_PROVIDER).toBe('websocketProvider');
      expect(ServiceKeys.RISK_CALCULATOR).toBe('riskCalculator');
      expect(ServiceKeys.LOGGER).toBe('logger');
    });
  });

  describe('Child Containers', () => {
    it('should create child container', () => {
      const child = container.createChild();
      expect(child).toBeInstanceOf(ServiceContainer);
    });

    it('should resolve services from parent', () => {
      const service = { name: 'parent' };
      container.registerInstance('parentService', service);

      const child = container.createChild();
      const resolved = child.get<typeof service>('parentService');

      expect(resolved).toBe(service);
    });

    it('should allow child to override parent service', () => {
      container.registerInstance('service', { source: 'parent' });

      const child = container.createChild();
      child.registerInstance('service', { source: 'child' });

      const parentResolved = container.get<{ source: string }>('service');
      const childResolved = child.get<{ source: string }>('service');

      expect(parentResolved.source).toBe('parent');
      expect(childResolved.source).toBe('child');
    });

    it('should not affect parent when registering in child', () => {
      const child = container.createChild();
      child.registerInstance('childOnly', { value: 'child' });

      expect(child.has('childOnly')).toBe(true);
      expect(container.has('childOnly')).toBe(false);
    });
  });

  describe('Service Disposal', () => {
    it('should dispose services with dispose method', () => {
      const disposeMock = jest.fn();
      const service = {
        name: 'disposable',
        dispose: disposeMock,
      };

      container.registerInstance('disposable', service);
      container.dispose();

      expect(disposeMock).toHaveBeenCalledTimes(1);
    });

    it('should handle errors during disposal', () => {
      const errorSpy = jest.spyOn(console, 'error').mockImplementation();
      const service = {
        dispose: () => {
          throw new Error('Disposal failed');
        },
      };

      container.registerInstance('faulty', service);
      expect(() => container.dispose()).not.toThrow();

      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining("Error disposing service 'faulty'"),
        expect.any(Error)
      );

      errorSpy.mockRestore();
    });

    it('should clear all services after disposal', () => {
      container.registerInstance('service1', { name: 'one' });
      container.registerInstance('service2', { name: 'two' });

      expect(container.has('service1')).toBe(true);
      expect(container.has('service2')).toBe(true);

      container.dispose();

      expect(container.has('service1')).toBe(false);
      expect(container.has('service2')).toBe(false);
    });

    it('should only dispose singleton instances', () => {
      const singletonDispose = jest.fn();
      const transientDispose = jest.fn();

      container.registerSingleton('singleton', () => ({
        dispose: singletonDispose,
      }));
      container.registerTransient('transient', () => ({
        dispose: transientDispose,
      }));

      // Create singleton instance
      container.get('singleton');

      // Don't create transient instance
      container.dispose();

      expect(singletonDispose).toHaveBeenCalledTimes(1);
      expect(transientDispose).not.toHaveBeenCalled();
    });
  });

  describe('Service Registration Info', () => {
    it('should return registration info', () => {
      container.registerSingleton('singleton', () => ({ value: 'test' }));

      const info = container.getRegistration('singleton');
      expect(info).toEqual({
        lifetime: ServiceLifetime.SINGLETON,
        hasInstance: false,
      });

      container.get('singleton');

      const infoAfter = container.getRegistration('singleton');
      expect(infoAfter).toEqual({
        lifetime: ServiceLifetime.SINGLETON,
        hasInstance: true,
      });
    });

    it('should return undefined for non-existent service', () => {
      const info = container.getRegistration('nonexistent');
      expect(info).toBeUndefined();
    });
  });

  describe('Service Unregistration', () => {
    it('should unregister a service', () => {
      container.registerInstance('test', { name: 'test' });
      expect(container.has('test')).toBe(true);

      const removed = container.unregister('test');
      expect(removed).toBe(true);
      expect(container.has('test')).toBe(false);
    });

    it('should return false when unregistering non-existent service', () => {
      const removed = container.unregister('nonexistent');
      expect(removed).toBe(false);
    });
  });

  describe('Clear All Services', () => {
    it('should clear all registered services', () => {
      container.registerInstance('service1', { name: 'one' });
      container.registerInstance('service2', { name: 'two' });
      container.registerInstance('service3', { name: 'three' });

      expect(container.getKeys().length).toBe(3);

      container.clear();

      expect(container.getKeys().length).toBe(0);
      expect(container.has('service1')).toBe(false);
      expect(container.has('service2')).toBe(false);
    });
  });

  describe('Get All Keys', () => {
    it('should return all registered service keys', () => {
      container.registerInstance('service1', { name: 'one' });
      container.registerInstance('service2', { name: 'two' });

      const keys = container.getKeys();
      expect(keys).toContain('service1');
      expect(keys).toContain('service2');
      expect(keys.length).toBe(2);
    });

    it('should include parent keys in child container', () => {
      container.registerInstance('parent', { name: 'parent' });

      const child = container.createChild();
      child.registerInstance('child', { name: 'child' });

      const keys = child.getKeys();
      expect(keys).toContain('parent');
      expect(keys).toContain('child');
    });
  });

  describe('Global Container', () => {
    it('should return same global container instance', () => {
      const container1 = getGlobalContainer();
      const container2 = getGlobalContainer();

      expect(container1).toBe(container2);
    });

    it('should reset global container', () => {
      const container1 = getGlobalContainer();
      container1.registerInstance('test', { value: 'test' });

      resetGlobalContainer();

      const container2 = getGlobalContainer();
      expect(container2).not.toBe(container1);
      expect(container2.has('test')).toBe(false);
    });

    it('should dispose old container when resetting', () => {
      const disposeMock = jest.fn();
      const container1 = getGlobalContainer();
      container1.registerInstance('disposable', { dispose: disposeMock });
      container1.get('disposable');

      resetGlobalContainer();

      expect(disposeMock).toHaveBeenCalledTimes(1);
    });
  });

  describe('Type Safety', () => {
    it('should provide type-safe service resolution', () => {
      interface TestService {
        getValue(): string;
      }

      const service: TestService = {
        getValue: () => 'test',
      };

      container.registerInstance('testService', service);

      const resolved = container.get<TestService>('testService');
      expect(resolved.getValue()).toBe('test');
      expect(typeof resolved.getValue).toBe('function');
    });
  });

  describe('Factory Dependencies', () => {
    it('should resolve dependencies in factory', () => {
      interface Logger {
        log(msg: string): void;
      }

      interface Service {
        doWork(): void;
      }

      const logger: Logger = {
        log: jest.fn(),
      };

      container.registerInstance('logger', logger);

      container.registerSingleton('service', () => {
        const log = container.get<Logger>('logger');
        return {
          doWork: () => log.log('working'),
        };
      });

      const service = container.get<Service>('service');
      service.doWork();

      expect(logger.log).toHaveBeenCalledWith('working');
    });
  });
});
