/**
 * Type Declarations
 * =================
 *
 * Global type declarations for better IDE support.
 */

import type { ServiceContainer } from './container';

declare global {
  /**
   * Global container singleton (available in browser console for debugging)
   */
  var __CONTAINER__: ServiceContainer | undefined;
}

export {};
