/**
 * Service Container
 * =================
 *
 * Exports for the dependency injection container.
 */

export {
  ServiceContainer,
  ServiceLifetime,
  getGlobalContainer,
  resetGlobalContainer,
  ServiceKeys,
} from './ServiceContainer';

export type { ServiceFactory, ServiceKey } from './ServiceContainer';
