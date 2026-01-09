/**
 * Portfolio Tracker
 * ==================
 *
 * Single Responsibility: Track and manage portfolio positions.
 * This class is responsible ONLY for:
 * - Adding positions to the portfolio
 * - Updating existing positions
 * - Removing positions from the portfolio
 * - Retrieving position information
 */

import { createLogger } from '@/lib/utils/logger';
import type { Position } from './types';

const logger = createLogger('PortfolioTracker');

export class PortfolioTracker {
  private positions: Map<string, Position> = new Map();

  /**
   * Add or update a position in the portfolio
   */
  updatePosition(position: Position): void {
    const isNew = !this.positions.has(position.symbol);
    this.positions.set(position.symbol, position);

    logger.debug(
      `Position ${isNew ? 'added' : 'updated'}: ${position.symbol}`,
      {
        quantity: position.quantity,
        marketValue: position.marketValue,
        pnl: position.pnl,
      }
    );
  }

  /**
   * Remove a position from the portfolio
   */
  removePosition(symbol: string): boolean {
    const existed = this.positions.has(symbol);
    this.positions.delete(symbol);

    if (existed) {
      logger.debug(`Position removed: ${symbol}`);
    } else {
      logger.warn(`Attempted to remove non-existent position: ${symbol}`);
    }

    return existed;
  }

  /**
   * Get a specific position by symbol
   */
  getPosition(symbol: string): Position | undefined {
    return this.positions.get(symbol);
  }

  /**
   * Get all positions as an array
   */
  getAllPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get all position symbols
   */
  getSymbols(): string[] {
    return Array.from(this.positions.keys());
  }

  /**
   * Get the number of positions
   */
  getPositionCount(): number {
    return this.positions.size;
  }

  /**
   * Check if a position exists
   */
  hasPosition(symbol: string): boolean {
    return this.positions.has(symbol);
  }

  /**
   * Clear all positions
   */
  clearAllPositions(): void {
    const count = this.positions.size;
    this.positions.clear();
    logger.info(`Cleared ${count} positions from portfolio`);
  }

  /**
   * Get portfolio summary statistics
   */
  getPortfolioSummary(): {
    totalValue: number;
    positionCount: number;
    symbols: string[];
  } {
    const positions = this.getAllPositions();
    const totalValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);

    return {
      totalValue,
      positionCount: this.positions.size,
      symbols: this.getSymbols(),
    };
  }
}
