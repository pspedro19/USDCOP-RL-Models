"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Copy, TrendingUp, TrendingDown, Clock, DollarSign, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import { toast } from "react-hot-toast";

interface PriceData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface CrosshairPosition {
  x: number;
  y: number;
  price: number;
  time: number;
  visible: boolean;
}

interface SmartCrosshairProps {
  position: CrosshairPosition;
  data?: PriceData[];
  priceScale?: {
    min: number;
    max: number;
  };
  snapToPrice?: boolean;
  snapTolerance?: number;
  showTooltip?: boolean;
  showDistanceMeasurement?: boolean;
  referencePrice?: number;
  onPriceCopy?: (price: number) => void;
  className?: string;
}

const SmartCrosshair = React.forwardRef<
  HTMLDivElement,
  SmartCrosshairProps
>(({
  position,
  data = [],
  priceScale,
  snapToPrice = true,
  snapTolerance = 5,
  showTooltip = true,
  showDistanceMeasurement = true,
  referencePrice,
  onPriceCopy,
  className,
}, ref) => {
  const [snappedPosition, setSnappedPosition] = React.useState(position);
  const [tooltipData, setTooltipData] = React.useState<PriceData | null>(null);
  const [isSnapped, setIsSnapped] = React.useState(false);

  // Find nearest data point for snapping
  const findNearestDataPoint = React.useCallback((targetTime: number): PriceData | null => {
    if (!data.length) return null;

    let closest = data[0];
    let minDiff = Math.abs(data[0].time - targetTime);

    for (const point of data) {
      const diff = Math.abs(point.time - targetTime);
      if (diff < minDiff) {
        minDiff = diff;
        closest = point;
      }
    }

    return closest;
  }, [data]);

  // Find nearest OHLC price for snapping
  const findNearestPrice = React.useCallback((targetPrice: number, candleData: PriceData): number => {
    const prices = [candleData.open, candleData.high, candleData.low, candleData.close];
    let closest = prices[0];
    let minDiff = Math.abs(prices[0] - targetPrice);

    for (const price of prices) {
      const diff = Math.abs(price - targetPrice);
      if (diff < minDiff) {
        minDiff = diff;
        closest = price;
      }
    }

    return closest;
  }, []);

  // Update snapped position and tooltip data
  React.useEffect(() => {
    if (!position.visible) {
      setSnappedPosition(position);
      setTooltipData(null);
      setIsSnapped(false);
      return;
    }

    const nearestData = findNearestDataPoint(position.time);

    if (snapToPrice && nearestData && priceScale) {
      const nearestPrice = findNearestPrice(position.price, nearestData);
      const priceRange = priceScale.max - priceScale.min;
      const tolerance = (priceRange * snapTolerance) / 100;

      if (Math.abs(position.price - nearestPrice) <= tolerance) {
        setSnappedPosition({
          ...position,
          price: nearestPrice,
        });
        setIsSnapped(true);
      } else {
        setSnappedPosition(position);
        setIsSnapped(false);
      }
    } else {
      setSnappedPosition(position);
      setIsSnapped(false);
    }

    setTooltipData(nearestData);
  }, [position, findNearestDataPoint, findNearestPrice, snapToPrice, priceScale, snapTolerance]);

  // Copy price to clipboard
  const copyPrice = React.useCallback(async () => {
    try {
      await navigator.clipboard.writeText(snappedPosition.price.toFixed(4));
      onPriceCopy?.(snappedPosition.price);
      toast.success(`Price copied: ${snappedPosition.price.toFixed(4)}`, {
        duration: 2000,
        position: "bottom-right",
      });
    } catch (error) {
      toast.error("Failed to copy price");
    }
  }, [snappedPosition.price, onPriceCopy]);

  // Calculate percentage change from reference price
  const calculatePercentageChange = React.useCallback(() => {
    if (!referencePrice) return null;
    const change = snappedPosition.price - referencePrice;
    const percentage = (change / referencePrice) * 100;
    return {
      absolute: change,
      percentage,
      isPositive: change >= 0,
    };
  }, [snappedPosition.price, referencePrice]);

  const percentageChange = calculatePercentageChange();

  if (!position.visible) return null;

  return (
    <div
      ref={ref}
      className={cn("absolute inset-0 pointer-events-none z-40", className)}
    >
      {/* Vertical crosshair line */}
      <motion.div
        className={cn(
          "absolute w-px bg-blue-400 opacity-60",
          isSnapped && "bg-yellow-400 opacity-80"
        )}
        style={{
          left: snappedPosition.x,
          top: 0,
          height: "100%",
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.6 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.1 }}
      />

      {/* Horizontal crosshair line */}
      <motion.div
        className={cn(
          "absolute h-px bg-blue-400 opacity-60",
          isSnapped && "bg-yellow-400 opacity-80"
        )}
        style={{
          top: snappedPosition.y,
          left: 0,
          width: "100%",
        }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.6 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.1 }}
      />

      {/* Center point */}
      <motion.div
        className={cn(
          "absolute w-2 h-2 bg-blue-400 rounded-full transform -translate-x-1 -translate-y-1",
          isSnapped && "bg-yellow-400 ring-2 ring-yellow-400 ring-opacity-30"
        )}
        style={{
          left: snappedPosition.x,
          top: snappedPosition.y,
        }}
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0 }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
      />

      {/* Price label */}
      <motion.div
        className="absolute pointer-events-auto"
        style={{
          right: 10,
          top: snappedPosition.y - 12,
        }}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 20 }}
      >
        <button
          onClick={copyPrice}
          className={cn(
            "px-2 py-1 text-xs font-mono bg-gray-900 text-white rounded border",
            isSnapped && "bg-yellow-900 border-yellow-400",
            "hover:bg-gray-800 transition-colors duration-150"
          )}
        >
          {snappedPosition.price.toFixed(4)}
        </button>
      </motion.div>

      {/* Time label */}
      <motion.div
        className="absolute"
        style={{
          left: snappedPosition.x - 40,
          bottom: 10,
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 20 }}
      >
        <div className="px-2 py-1 text-xs font-mono bg-gray-900 text-white rounded border">
          {format(new Date(snappedPosition.time), "HH:mm:ss")}
        </div>
      </motion.div>

      {/* Tooltip with comprehensive information */}
      <AnimatePresence>
        {showTooltip && tooltipData && (
          <motion.div
            className="absolute pointer-events-auto z-50"
            style={{
              left: Math.min(snappedPosition.x + 20, window.innerWidth - 320),
              top: Math.max(snappedPosition.y - 160, 20),
            }}
            initial={{ opacity: 0, scale: 0.9, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 10 }}
            transition={{ type: "spring", stiffness: 400, damping: 25 }}
          >
            <div className="bg-gray-900/95 backdrop-blur-sm border border-gray-700 rounded-lg shadow-2xl p-4 min-w-[280px]">
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="h-4 w-4 text-blue-400" />
                  <span className="text-sm font-medium text-gray-200">Price Info</span>
                </div>
                <button
                  onClick={copyPrice}
                  className="p-1 hover:bg-gray-700 rounded transition-colors"
                >
                  <Copy className="h-3 w-3 text-gray-400" />
                </button>
              </div>

              {/* OHLC Data */}
              <div className="space-y-2 mb-3">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Open:</span>
                    <span className="text-white font-mono">{tooltipData.open.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">High:</span>
                    <span className="text-green-400 font-mono">{tooltipData.high.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Low:</span>
                    <span className="text-red-400 font-mono">{tooltipData.low.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Close:</span>
                    <span className="text-white font-mono">{tooltipData.close.toFixed(4)}</span>
                  </div>
                </div>
              </div>

              {/* Volume */}
              {tooltipData.volume && (
                <div className="flex justify-between text-xs mb-3">
                  <span className="text-gray-400">Volume:</span>
                  <span className="text-white font-mono">
                    {tooltipData.volume.toLocaleString()}
                  </span>
                </div>
              )}

              {/* Time */}
              <div className="flex items-center space-x-2 text-xs mb-3">
                <Clock className="h-3 w-3 text-gray-400" />
                <span className="text-gray-400">
                  {format(new Date(tooltipData.time), "MMM dd, yyyy HH:mm:ss")}
                </span>
              </div>

              {/* Distance measurement */}
              {showDistanceMeasurement && percentageChange && (
                <div className="border-t border-gray-700 pt-3">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center space-x-1">
                      {percentageChange.isPositive ? (
                        <TrendingUp className="h-3 w-3 text-green-400" />
                      ) : (
                        <TrendingDown className="h-3 w-3 text-red-400" />
                      )}
                      <span className="text-gray-400">Change:</span>
                    </div>
                    <div className="text-right">
                      <div className={cn(
                        "font-mono",
                        percentageChange.isPositive ? "text-green-400" : "text-red-400"
                      )}>
                        {percentageChange.isPositive ? "+" : ""}{percentageChange.absolute.toFixed(4)}
                      </div>
                      <div className={cn(
                        "font-mono text-xs",
                        percentageChange.isPositive ? "text-green-400" : "text-red-400"
                      )}>
                        {percentageChange.isPositive ? "+" : ""}{percentageChange.percentage.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Snap indicator */}
              {isSnapped && (
                <div className="border-t border-gray-700 pt-2 mt-2">
                  <div className="flex items-center space-x-1 text-xs text-yellow-400">
                    <div className="w-1.5 h-1.5 bg-yellow-400 rounded-full" />
                    <span>Snapped to OHLC</span>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});

SmartCrosshair.displayName = "SmartCrosshair";

export { SmartCrosshair, type CrosshairPosition, type PriceData };