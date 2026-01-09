/**
 * VirtualOrderBook - High-Performance Order Book Virtualization
 *
 * Elite trading platform order book optimized for:
 * - Real-time bid/ask updates
 * - Thousands of price levels
 * - Smooth scrolling at 60 FPS
 * - Market depth visualization
 * - Volume aggregation
 *
 * Features:
 * - Adaptive level aggregation
 * - Smart price level grouping
 * - Real-time diff updates
 * - Memory-efficient rendering
 * - Spread highlighting
 * - Volume heatmap
 */

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
  memo
} from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FixedSizeList as List } from 'react-window';
import { getPerformanceOptimizer } from '../../libs/core/performance/PerformanceOptimizer';

export interface OrderBookLevel {
  readonly price: number;
  readonly volume: number;
  readonly count: number;
  readonly side: 'bid' | 'ask';
  readonly aggregated?: boolean;
  readonly change?: 'new' | 'updated' | 'removed';
}

export interface OrderBookData {
  readonly bids: OrderBookLevel[];
  readonly asks: OrderBookLevel[];
  readonly spread: number;
  readonly spreadPercent: number;
  readonly midPrice: number;
  readonly bestBid: number;
  readonly bestAsk: number;
  readonly totalBidVolume: number;
  readonly totalAskVolume: number;
  readonly timestamp: number;
}

export interface VirtualOrderBookProps {
  data: OrderBookData;
  height: number;
  maxLevels?: number;
  priceDecimals?: number;
  volumeDecimals?: number;
  enableAggregation?: boolean;
  aggregationStep?: number;
  enableHeatmap?: boolean;
  enableAnimation?: boolean;
  enableVirtualization?: boolean;
  onLevelClick?: (level: OrderBookLevel) => void;
  onLevelHover?: (level: OrderBookLevel | null) => void;
  className?: string;
}

interface AggregatedLevel {
  readonly price: number;
  readonly volume: number;
  readonly count: number;
  readonly side: 'bid' | 'ask';
  readonly levels: OrderBookLevel[];
  readonly avgPrice: number;
  readonly minPrice: number;
  readonly maxPrice: number;
}

const VirtualOrderBook: React.FC<VirtualOrderBookProps> = ({
  data,
  height,
  maxLevels = 50,
  priceDecimals = 2,
  volumeDecimals = 0,
  enableAggregation = true,
  aggregationStep = 0.1,
  enableHeatmap = true,
  enableAnimation = true,
  enableVirtualization = true,
  onLevelClick,
  onLevelHover,
  className = ''
}) => {
  const optimizer = getPerformanceOptimizer();
  const bidsListRef = useRef<List>(null);
  const asksListRef = useRef<List>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [hoveredLevel, setHoveredLevel] = useState<OrderBookLevel | null>(null);
  const [selectedPrice, setSelectedPrice] = useState<number | null>(null);
  const [aggregationEnabled, setAggregationEnabled] = useState(enableAggregation);
  const [currentAggregationStep, setCurrentAggregationStep] = useState(aggregationStep);

  // Process and aggregate order book data
  const processedData = useMemo(() => {
    if (!data.bids.length && !data.asks.length) {
      return {
        bids: [],
        asks: [],
        maxBidVolume: 0,
        maxAskVolume: 0,
        cumulativeBids: [],
        cumulativeAsks: []
      };
    }

    let processedBids = [...data.bids];
    let processedAsks = [...data.asks];

    // Apply aggregation if enabled
    if (aggregationEnabled && currentAggregationStep > 0) {
      processedBids = aggregateLevels(processedBids, 'bid', currentAggregationStep);
      processedAsks = aggregateLevels(processedAsks, 'ask', currentAggregationStep);
    }

    // Limit levels for performance
    processedBids = processedBids.slice(0, maxLevels);
    processedAsks = processedAsks.slice(0, maxLevels);

    // Calculate cumulative volumes for depth visualization
    const cumulativeBids = calculateCumulativeVolume(processedBids);
    const cumulativeAsks = calculateCumulativeVolume(processedAsks);

    const maxBidVolume = Math.max(...processedBids.map(l => l.volume), 0);
    const maxAskVolume = Math.max(...processedAsks.map(l => l.volume), 0);

    return {
      bids: processedBids,
      asks: processedAsks,
      maxBidVolume,
      maxAskVolume,
      cumulativeBids,
      cumulativeAsks
    };
  }, [data, aggregationEnabled, currentAggregationStep, maxLevels]);

  // Aggregate price levels for better performance and readability
  const aggregateLevels = useCallback((
    levels: OrderBookLevel[],
    side: 'bid' | 'ask',
    step: number
  ): OrderBookLevel[] => {
    const aggregated = new Map<number, AggregatedLevel>();

    levels.forEach(level => {
      const bucketPrice = side === 'bid'
        ? Math.floor(level.price / step) * step
        : Math.ceil(level.price / step) * step;

      if (!aggregated.has(bucketPrice)) {
        aggregated.set(bucketPrice, {
          price: bucketPrice,
          volume: 0,
          count: 0,
          side,
          levels: [],
          avgPrice: 0,
          minPrice: level.price,
          maxPrice: level.price
        });
      }

      const bucket = aggregated.get(bucketPrice)!;
      bucket.volume += level.volume;
      bucket.count += level.count;
      bucket.levels.push(level);
      bucket.minPrice = Math.min(bucket.minPrice, level.price);
      bucket.maxPrice = Math.max(bucket.maxPrice, level.price);
      bucket.avgPrice = bucket.levels.reduce((sum, l) => sum + l.price * l.volume, 0) / bucket.volume;
    });

    return Array.from(aggregated.values())
      .map(bucket => ({
        price: bucket.avgPrice,
        volume: bucket.volume,
        count: bucket.count,
        side: bucket.side,
        aggregated: true
      }))
      .sort((a, b) => side === 'bid' ? b.price - a.price : a.price - b.price);
  }, []);

  // Calculate cumulative volume for depth chart
  const calculateCumulativeVolume = useCallback((levels: OrderBookLevel[]): number[] => {
    const cumulative: number[] = [];
    let total = 0;

    levels.forEach(level => {
      total += level.volume;
      cumulative.push(total);
    });

    return cumulative;
  }, []);

  // Handle level interactions
  const handleLevelClick = useCallback((level: OrderBookLevel) => {
    setSelectedPrice(level.price);
    onLevelClick?.(level);
  }, [onLevelClick]);

  const handleLevelHover = useCallback((level: OrderBookLevel | null) => {
    setHoveredLevel(level);
    onLevelHover?.(level);
  }, [onLevelHover]);

  // Format price with appropriate precision
  const formatPrice = useCallback((price: number): string => {
    return price.toFixed(priceDecimals);
  }, [priceDecimals]);

  // Format volume with appropriate precision
  const formatVolume = useCallback((volume: number): string => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(1)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(1)}K`;
    }
    return volume.toFixed(volumeDecimals);
  }, [volumeDecimals]);

  // Calculate volume bar width for visualization
  const calculateVolumeBarWidth = useCallback((
    volume: number,
    maxVolume: number,
    side: 'bid' | 'ask'
  ): number => {
    if (maxVolume === 0) return 0;
    return (volume / maxVolume) * 100;
  }, []);

  // Get volume-based color intensity for heatmap
  const getVolumeIntensity = useCallback((
    volume: number,
    maxVolume: number
  ): number => {
    if (maxVolume === 0) return 0;
    return Math.min(1, volume / maxVolume);
  }, []);

  // Order book level row component
  const OrderBookRow = memo<{
    index: number;
    style: React.CSSProperties;
    side: 'bid' | 'ask';
  }>(({ index, style, side }) => {
    const levels = side === 'bid' ? processedData.bids : processedData.asks;
    const maxVolume = side === 'bid' ? processedData.maxBidVolume : processedData.maxAskVolume;
    const cumulativeVolumes = side === 'bid' ? processedData.cumulativeBids : processedData.cumulativeAsks;

    if (index >= levels.length) {
      return <div style={style} className="empty-row" />;
    }

    const level = levels[index];
    const cumulativeVolume = cumulativeVolumes[index] || 0;
    const volumeBarWidth = calculateVolumeBarWidth(level.volume, maxVolume, side);
    const volumeIntensity = getVolumeIntensity(level.volume, maxVolume);
    const cumulativeBarWidth = calculateVolumeBarWidth(
      cumulativeVolume,
      Math.max(...cumulativeVolumes),
      side
    );

    const isHovered = hoveredLevel?.price === level.price;
    const isSelected = selectedPrice === level.price;
    const isBestLevel = index === 0;

    const priceChangeClass = level.change ? `price-${level.change}` : '';
    const sideClass = side === 'bid' ? 'bid-side' : 'ask-side';

    return (
      <motion.div
        style={style}
        className={`order-book-row ${sideClass} ${priceChangeClass} ${
          isHovered ? 'hovered' : ''
        } ${isSelected ? 'selected' : ''} ${isBestLevel ? 'best-level' : ''}`}
        onClick={() => handleLevelClick(level)}
        onMouseEnter={() => handleLevelHover(level)}
        onMouseLeave={() => handleLevelHover(null)}
        initial={enableAnimation ? { opacity: 0, x: side === 'bid' ? -20 : 20 } : false}
        animate={enableAnimation ? { opacity: 1, x: 0 } : false}
        exit={enableAnimation ? { opacity: 0, x: side === 'bid' ? -20 : 20 } : false}
        transition={{ duration: 0.15 }}
      >
        {/* Volume background bar */}
        <div
          className={`volume-bar ${side}`}
          style={{
            width: `${volumeBarWidth}%`,
            opacity: enableHeatmap ? 0.3 + (volumeIntensity * 0.4) : 0.3
          }}
        />

        {/* Cumulative depth bar */}
        <div
          className={`depth-bar ${side}`}
          style={{
            width: `${cumulativeBarWidth}%`,
            opacity: 0.1
          }}
        />

        {/* Row content */}
        <div className="row-content">
          {side === 'bid' ? (
            <>
              <div className="volume-cell">
                {formatVolume(level.volume)}
                {level.aggregated && <span className="aggregated-indicator">*</span>}
              </div>
              <div className="price-cell">
                {formatPrice(level.price)}
              </div>
            </>
          ) : (
            <>
              <div className="price-cell">
                {formatPrice(level.price)}
              </div>
              <div className="volume-cell">
                {formatVolume(level.volume)}
                {level.aggregated && <span className="aggregated-indicator">*</span>}
              </div>
            </>
          )}
        </div>

        {/* Hover details */}
        <AnimatePresence>
          {isHovered && (
            <motion.div
              className="hover-details"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.1 }}
            >
              <div className="detail-item">
                <span>Orders: {level.count}</span>
              </div>
              <div className="detail-item">
                <span>Cumulative: {formatVolume(cumulativeVolume)}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    );
  });

  OrderBookRow.displayName = 'OrderBookRow';

  // Spread indicator component
  const SpreadIndicator = memo(() => (
    <motion.div
      className="spread-indicator"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="spread-content">
        <div className="spread-value">
          Spread: {formatPrice(data.spread)}
        </div>
        <div className="spread-percent">
          ({data.spreadPercent.toFixed(3)}%)
        </div>
      </div>
      <div className="mid-price">
        Mid: {formatPrice(data.midPrice)}
      </div>
    </motion.div>
  ));

  SpreadIndicator.displayName = 'SpreadIndicator';

  // Controls component
  const OrderBookControls = memo(() => (
    <div className="order-book-controls">
      <div className="control-group">
        <label className="control-label">
          <input
            type="checkbox"
            checked={aggregationEnabled}
            onChange={(e) => setAggregationEnabled(e.target.checked)}
          />
          Aggregate
        </label>
        {aggregationEnabled && (
          <select
            value={currentAggregationStep}
            onChange={(e) => setCurrentAggregationStep(Number(e.target.value))}
            className="aggregation-select"
          >
            <option value={0.01}>0.01</option>
            <option value={0.1}>0.1</option>
            <option value={1}>1.0</option>
            <option value={10}>10.0</option>
          </select>
        )}
      </div>
    </div>
  ));

  OrderBookControls.displayName = 'OrderBookControls';

  return (
    <motion.div
      ref={containerRef}
      className={`virtual-order-book ${className}`}
      style={{ height }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
    >
      <OrderBookControls />

      <div className="order-book-container">
        {/* Asks (top) */}
        <div className="asks-container">
          {enableVirtualization ? (
            <List
              ref={asksListRef}
              height={(height - 120) / 2}
              itemCount={Math.min(processedData.asks.length, maxLevels)}
              itemSize={28}
              className="asks-list"
              itemData={{ side: 'ask' }}
            >
              {({ index, style }) => (
                <OrderBookRow index={index} style={style} side="ask" />
              )}
            </List>
          ) : (
            <div className="static-list asks-list" style={{ height: (height - 120) / 2 }}>
              {processedData.asks.slice(0, maxLevels).map((_, index) => (
                <OrderBookRow
                  key={index}
                  index={index}
                  style={{ height: 28 }}
                  side="ask"
                />
              ))}
            </div>
          )}
        </div>

        <SpreadIndicator />

        {/* Bids (bottom) */}
        <div className="bids-container">
          {enableVirtualization ? (
            <List
              ref={bidsListRef}
              height={(height - 120) / 2}
              itemCount={Math.min(processedData.bids.length, maxLevels)}
              itemSize={28}
              className="bids-list"
              itemData={{ side: 'bid' }}
            >
              {({ index, style }) => (
                <OrderBookRow index={index} style={style} side="bid" />
              )}
            </List>
          ) : (
            <div className="static-list bids-list" style={{ height: (height - 120) / 2 }}>
              {processedData.bids.slice(0, maxLevels).map((_, index) => (
                <OrderBookRow
                  key={index}
                  index={index}
                  style={{ height: 28 }}
                  side="bid"
                />
              ))}
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        .virtual-order-book {
          display: flex;
          flex-direction: column;
          background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
          border-radius: 12px;
          border: 1px solid #334155;
          overflow: hidden;
          position: relative;
          font-family: 'SF Mono', 'Monaco', 'Roboto Mono', monospace;
        }

        .order-book-controls {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 12px;
          background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
          border-bottom: 1px solid #475569;
        }

        .control-group {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .control-label {
          display: flex;
          align-items: center;
          gap: 4px;
          color: #e2e8f0;
          font-size: 12px;
          cursor: pointer;
        }

        .aggregation-select {
          padding: 2px 6px;
          background: #1e293b;
          border: 1px solid #475569;
          border-radius: 4px;
          color: #e2e8f0;
          font-size: 11px;
        }

        .order-book-container {
          flex: 1;
          display: flex;
          flex-direction: column;
        }

        .asks-container,
        .bids-container {
          position: relative;
        }

        .asks-list,
        .bids-list,
        .static-list {
          overflow: auto;
        }

        .order-book-row {
          display: flex;
          align-items: center;
          position: relative;
          cursor: pointer;
          transition: all 0.15s ease;
          font-size: 11px;
          border-bottom: 1px solid rgba(51, 65, 85, 0.3);
        }

        .order-book-row:hover {
          background-color: rgba(59, 130, 246, 0.1);
        }

        .order-book-row.selected {
          background-color: rgba(59, 130, 246, 0.2);
          border-left: 3px solid #3b82f6;
        }

        .order-book-row.best-level {
          border-left: 2px solid;
        }

        .order-book-row.bid-side.best-level {
          border-left-color: #10b981;
        }

        .order-book-row.ask-side.best-level {
          border-left-color: #ef4444;
        }

        .volume-bar {
          position: absolute;
          top: 0;
          bottom: 0;
          z-index: 1;
          transition: all 0.2s ease;
        }

        .volume-bar.bid {
          right: 0;
          background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.4));
        }

        .volume-bar.ask {
          left: 0;
          background: linear-gradient(90deg, rgba(239, 68, 68, 0.4), transparent);
        }

        .depth-bar {
          position: absolute;
          top: 0;
          bottom: 0;
          z-index: 0;
          background: rgba(100, 116, 139, 0.2);
        }

        .depth-bar.bid {
          right: 0;
        }

        .depth-bar.ask {
          left: 0;
        }

        .row-content {
          display: flex;
          width: 100%;
          padding: 6px 12px;
          position: relative;
          z-index: 2;
        }

        .bid-side .row-content {
          justify-content: space-between;
        }

        .ask-side .row-content {
          justify-content: space-between;
        }

        .price-cell {
          font-weight: 600;
          min-width: 80px;
          text-align: right;
        }

        .bid-side .price-cell {
          color: #10b981;
        }

        .ask-side .price-cell {
          color: #ef4444;
        }

        .volume-cell {
          color: #e2e8f0;
          min-width: 60px;
          text-align: right;
        }

        .aggregated-indicator {
          color: #fbbf24;
          margin-left: 2px;
          font-size: 9px;
        }

        .price-new {
          animation: flash-new 0.5s ease-out;
        }

        .price-updated {
          animation: flash-updated 0.3s ease-out;
        }

        .price-removed {
          animation: flash-removed 0.3s ease-out;
        }

        @keyframes flash-new {
          0% { background-color: rgba(16, 185, 129, 0.5); }
          100% { background-color: transparent; }
        }

        @keyframes flash-updated {
          0% { background-color: rgba(59, 130, 246, 0.3); }
          100% { background-color: transparent; }
        }

        @keyframes flash-removed {
          0% { background-color: rgba(239, 68, 68, 0.3); }
          100% { background-color: transparent; }
        }

        .spread-indicator {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 12px;
          background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
          border-top: 1px solid #6b7280;
          border-bottom: 1px solid #6b7280;
          color: #e2e8f0;
          font-size: 11px;
          font-weight: 600;
        }

        .spread-content {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .spread-value {
          color: #fbbf24;
        }

        .spread-percent {
          color: #94a3b8;
        }

        .mid-price {
          color: #3b82f6;
        }

        .hover-details {
          position: absolute;
          right: 100%;
          top: 50%;
          transform: translateY(-50%);
          background: rgba(15, 23, 42, 0.95);
          backdrop-filter: blur(8px);
          border: 1px solid #334155;
          border-radius: 6px;
          padding: 6px 8px;
          margin-right: 8px;
          z-index: 100;
          white-space: nowrap;
          font-size: 10px;
          color: #e2e8f0;
        }

        .detail-item {
          margin-bottom: 2px;
        }

        .detail-item:last-child {
          margin-bottom: 0;
        }

        .empty-row {
          background: transparent;
        }
      `}</style>
    </motion.div>
  );
};

VirtualOrderBook.displayName = 'VirtualOrderBook';

export default VirtualOrderBook;