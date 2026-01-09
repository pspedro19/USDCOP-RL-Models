/**
 * Shared Validation Utilities for Elite Trading Platform
 * Professional validation and sanitization functions
 */

/**
 * Validate trading symbol format
 */
export function isValidSymbol(symbol: string): boolean {
  // Basic symbol validation - letters only, 3-10 characters
  const symbolRegex = /^[A-Z]{3,10}$/;
  return symbolRegex.test(symbol.toUpperCase());
}

/**
 * Validate forex pair format
 */
export function isValidForexPair(pair: string): boolean {
  // Forex pairs are typically 6 characters (USDEUR, GBPJPY, etc.)
  const forexRegex = /^[A-Z]{6}$/;
  return forexRegex.test(pair.toUpperCase());
}

/**
 * Validate price value
 */
export function isValidPrice(price: number): boolean {
  return typeof price === 'number' &&
         !isNaN(price) &&
         isFinite(price) &&
         price > 0;
}

/**
 * Validate volume/size value
 */
export function isValidVolume(volume: number): boolean {
  return typeof volume === 'number' &&
         !isNaN(volume) &&
         isFinite(volume) &&
         volume >= 0;
}

/**
 * Validate order side
 */
export function isValidOrderSide(side: string): boolean {
  return ['buy', 'sell'].includes(side.toLowerCase());
}

/**
 * Validate order type
 */
export function isValidOrderType(type: string): boolean {
  const validTypes = ['market', 'limit', 'stop_market', 'stop_limit', 'take_profit', 'take_profit_limit'];
  return validTypes.includes(type.toLowerCase());
}

/**
 * Validate time in force
 */
export function isValidTimeInForce(tif: string): boolean {
  const validTIF = ['GTC', 'IOC', 'FOK', 'GTD', 'GTT'];
  return validTIF.includes(tif.toUpperCase());
}

/**
 * Validate timestamp
 */
export function isValidTimestamp(timestamp: number): boolean {
  return typeof timestamp === 'number' &&
         !isNaN(timestamp) &&
         isFinite(timestamp) &&
         timestamp > 0 &&
         timestamp <= Date.now() + 86400000; // Not more than 1 day in future
}

/**
 * Validate percentage value
 */
export function isValidPercentage(percentage: number): boolean {
  return typeof percentage === 'number' &&
         !isNaN(percentage) &&
         isFinite(percentage) &&
         percentage >= -100 &&
         percentage <= 100;
}

/**
 * Validate leverage value
 */
export function isValidLeverage(leverage: number): boolean {
  return typeof leverage === 'number' &&
         !isNaN(leverage) &&
         isFinite(leverage) &&
         leverage >= 1 &&
         leverage <= 1000;
}

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Validate API key format
 */
export function isValidApiKey(apiKey: string): boolean {
  // Basic API key validation - alphanumeric and special chars, 20-128 characters
  const apiKeyRegex = /^[a-zA-Z0-9_\-=+/]{20,128}$/;
  return apiKeyRegex.test(apiKey);
}

/**
 * Validate WebSocket URL
 */
export function isValidWebSocketUrl(url: string): boolean {
  try {
    const urlObj = new URL(url);
    return urlObj.protocol === 'ws:' || urlObj.protocol === 'wss:';
  } catch {
    return false;
  }
}

/**
 * Validate HTTP URL
 */
export function isValidHttpUrl(url: string): boolean {
  try {
    const urlObj = new URL(url);
    return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
  } catch {
    return false;
  }
}

/**
 * Sanitize string input
 */
export function sanitizeString(input: string): string {
  return input
    .trim()
    .replace(/[<>]/g, '') // Remove potential HTML tags
    .replace(/[^\w\s\-_.@]/g, ''); // Keep only safe characters
}

/**
 * Sanitize number input
 */
export function sanitizeNumber(input: any): number | null {
  const num = parseFloat(input);
  return isNaN(num) ? null : num;
}

/**
 * Validate order data
 */
export interface OrderValidation {
  isValid: boolean;
  errors: string[];
}

export function validateOrderData(order: {
  symbol?: string;
  side?: string;
  type?: string;
  quantity?: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: string;
}): OrderValidation {
  const errors: string[] = [];

  // Required fields
  if (!order.symbol) {
    errors.push('Symbol is required');
  } else if (!isValidSymbol(order.symbol)) {
    errors.push('Invalid symbol format');
  }

  if (!order.side) {
    errors.push('Order side is required');
  } else if (!isValidOrderSide(order.side)) {
    errors.push('Invalid order side');
  }

  if (!order.type) {
    errors.push('Order type is required');
  } else if (!isValidOrderType(order.type)) {
    errors.push('Invalid order type');
  }

  if (order.quantity === undefined || order.quantity === null) {
    errors.push('Quantity is required');
  } else if (!isValidVolume(order.quantity)) {
    errors.push('Invalid quantity');
  }

  // Price validation for limit orders
  if (['limit', 'stop_limit', 'take_profit_limit'].includes(order.type || '')) {
    if (order.price === undefined || order.price === null) {
      errors.push('Price is required for limit orders');
    } else if (!isValidPrice(order.price)) {
      errors.push('Invalid price');
    }
  }

  // Stop price validation for stop orders
  if (['stop_market', 'stop_limit'].includes(order.type || '')) {
    if (order.stopPrice === undefined || order.stopPrice === null) {
      errors.push('Stop price is required for stop orders');
    } else if (!isValidPrice(order.stopPrice)) {
      errors.push('Invalid stop price');
    }
  }

  // Time in force validation
  if (order.timeInForce && !isValidTimeInForce(order.timeInForce)) {
    errors.push('Invalid time in force');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
}

/**
 * Validate market data
 */
export function validateMarketTick(tick: {
  symbol?: string;
  timestamp?: number;
  bid?: number;
  ask?: number;
  last?: number;
  volume?: number;
}): OrderValidation {
  const errors: string[] = [];

  if (!tick.symbol || !isValidSymbol(tick.symbol)) {
    errors.push('Invalid or missing symbol');
  }

  if (!tick.timestamp || !isValidTimestamp(tick.timestamp)) {
    errors.push('Invalid or missing timestamp');
  }

  if (tick.bid !== undefined && !isValidPrice(tick.bid)) {
    errors.push('Invalid bid price');
  }

  if (tick.ask !== undefined && !isValidPrice(tick.ask)) {
    errors.push('Invalid ask price');
  }

  if (tick.last !== undefined && !isValidPrice(tick.last)) {
    errors.push('Invalid last price');
  }

  if (tick.volume !== undefined && !isValidVolume(tick.volume)) {
    errors.push('Invalid volume');
  }

  // Cross validation
  if (tick.bid && tick.ask && tick.bid >= tick.ask) {
    errors.push('Bid price must be less than ask price');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
}

/**
 * Validate configuration object
 */
export function validateConfig(config: Record<string, any>): OrderValidation {
  const errors: string[] = [];

  // Validate API URLs
  if (config.apiUrl && !isValidHttpUrl(config.apiUrl)) {
    errors.push('Invalid API URL format');
  }

  if (config.wsUrl && !isValidWebSocketUrl(config.wsUrl)) {
    errors.push('Invalid WebSocket URL format');
  }

  // Validate numeric configs
  if (config.maxRetries !== undefined && (config.maxRetries < 0 || config.maxRetries > 100)) {
    errors.push('Max retries must be between 0 and 100');
  }

  if (config.timeout !== undefined && (config.timeout < 0 || config.timeout > 300000)) {
    errors.push('Timeout must be between 0 and 300000ms');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
}

/**
 * Validate range (min <= value <= max)
 */
export function isInRange(value: number, min: number, max: number): boolean {
  return value >= min && value <= max;
}

/**
 * Validate array is not empty
 */
export function isNonEmptyArray<T>(arr: T[]): boolean {
  return Array.isArray(arr) && arr.length > 0;
}

/**
 * Validate object has required properties
 */
export function hasRequiredProperties(obj: any, requiredProps: string[]): boolean {
  if (!obj || typeof obj !== 'object') return false;

  return requiredProps.every(prop => obj.hasOwnProperty(prop) && obj[prop] !== undefined);
}