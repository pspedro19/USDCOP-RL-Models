/**
 * ISubscribable Interface
 * ========================
 *
 * Generic subscription interface for observable patterns and event handling.
 */

/**
 * Subscription handler function
 */
export type SubscriptionHandler<T = unknown> = (data: T) => void;

/**
 * Error handler function
 */
export type ErrorHandler = (error: Error | string) => void;

/**
 * Unsubscribe function
 */
export type UnsubscribeFn = () => void;

/**
 * Generic subscribable interface
 */
export interface ISubscribable<T = unknown> {
  /**
   * Subscribe to updates
   * @returns Unsubscribe function
   */
  subscribe(handler: SubscriptionHandler<T>): UnsubscribeFn;

  /**
   * Unsubscribe a specific handler
   */
  unsubscribe(handler: SubscriptionHandler<T>): void;

  /**
   * Unsubscribe all handlers
   */
  unsubscribeAll(): void;

  /**
   * Get current subscriber count
   */
  getSubscriberCount(): number;
}

/**
 * Extended subscribable with error handling
 */
export interface ISubscribableWithErrors<T = unknown> extends ISubscribable<T> {
  /**
   * Subscribe to errors
   * @returns Unsubscribe function
   */
  onError(handler: ErrorHandler): UnsubscribeFn;

  /**
   * Remove error handler
   */
  offError(handler: ErrorHandler): void;
}

/**
 * Observable value that can be subscribed to
 */
export interface IObservable<T> extends ISubscribable<T> {
  /**
   * Get current value
   */
  getValue(): T;

  /**
   * Set value and notify subscribers
   */
  setValue(value: T): void;

  /**
   * Update value with a function
   */
  update(updater: (current: T) => T): void;
}

/**
 * Observable with async updates
 */
export interface IAsyncObservable<T> extends IObservable<T> {
  /**
   * Set value asynchronously
   */
  setValueAsync(value: Promise<T>): Promise<void>;

  /**
   * Check if loading
   */
  isLoading(): boolean;

  /**
   * Subscribe to loading state changes
   */
  onLoadingChange(handler: (loading: boolean) => void): UnsubscribeFn;
}

/**
 * Event emitter interface
 */
export interface IEventEmitter<TEvents extends Record<string, unknown> = Record<string, unknown>> {
  /**
   * Register event handler
   */
  on<K extends keyof TEvents>(event: K, handler: SubscriptionHandler<TEvents[K]>): UnsubscribeFn;

  /**
   * Register one-time event handler
   */
  once<K extends keyof TEvents>(event: K, handler: SubscriptionHandler<TEvents[K]>): UnsubscribeFn;

  /**
   * Unregister event handler
   */
  off<K extends keyof TEvents>(event: K, handler: SubscriptionHandler<TEvents[K]>): void;

  /**
   * Emit event to all handlers
   */
  emit<K extends keyof TEvents>(event: K, data: TEvents[K]): void;

  /**
   * Remove all handlers for an event
   */
  removeAllListeners<K extends keyof TEvents>(event?: K): void;

  /**
   * Get listener count for an event
   */
  listenerCount<K extends keyof TEvents>(event: K): number;
}

/**
 * Subject interface (can both emit and subscribe)
 */
export interface ISubject<T> extends ISubscribable<T> {
  /**
   * Emit value to all subscribers
   */
  next(value: T): void;

  /**
   * Emit error to all subscribers
   */
  error(error: Error | string): void;

  /**
   * Complete the subject (no more values)
   */
  complete(): void;

  /**
   * Check if subject is completed
   */
  isCompleted(): boolean;
}

/**
 * Behavior subject (subject with current value)
 */
export interface IBehaviorSubject<T> extends ISubject<T>, IObservable<T> {
  /**
   * Get current value (same as getValue)
   */
  value: T;
}

/**
 * Replay subject (buffers last N values)
 */
export interface IReplaySubject<T> extends ISubject<T> {
  /**
   * Get buffered values
   */
  getBufferedValues(): T[];

  /**
   * Clear buffer
   */
  clearBuffer(): void;

  /**
   * Get buffer size
   */
  getBufferSize(): number;
}

/**
 * Subscription options
 */
export interface SubscriptionOptions {
  /**
   * Immediately invoke handler with current value
   */
  immediate?: boolean;

  /**
   * Filter function to skip certain updates
   */
  filter?: (value: unknown) => boolean;

  /**
   * Throttle updates (milliseconds)
   */
  throttle?: number;

  /**
   * Debounce updates (milliseconds)
   */
  debounce?: number;

  /**
   * Only emit distinct values
   */
  distinct?: boolean;
}
