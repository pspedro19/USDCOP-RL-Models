/**
 * Replay Animation Hook
 *
 * Handles the animation loop for replay playback using requestAnimationFrame.
 * Provides smooth frame timing, performance monitoring, and adaptive quality.
 */

import { useEffect, useRef, useCallback, useState, useMemo } from 'react';
import { ReplaySpeed, MODEL_CONFIG } from '@/types/replay';
import {
  replayPerfMonitor,
  getAdaptiveQuality,
  QualitySettings,
  QUALITY_PRESETS,
} from '@/utils/replayPerformance';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface UseReplayAnimationOptions {
  isPlaying: boolean;
  speed: ReplaySpeed;
  currentDate: Date;
  startDate: Date;
  endDate: Date;
  onTick: (deltaMs: number) => void;
  onComplete?: () => void;
  adaptiveQuality?: boolean;
  targetFps?: number;
}

export interface UseReplayAnimationReturn {
  fps: number;
  frameCount: number;
  quality: QualitySettings;
  qualityLevel: 'high' | 'medium' | 'low';
  isAnimating: boolean;
  resetAnimation: () => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

const DEFAULT_TARGET_FPS = 60;
const TICK_INTERVAL_BASE_MS = 100; // Base tick interval at 1x speed
const FPS_UPDATE_INTERVAL_MS = 500; // How often to update FPS display

// ═══════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════

export function useReplayAnimation(
  options: UseReplayAnimationOptions
): UseReplayAnimationReturn {
  const {
    isPlaying,
    speed,
    currentDate,
    // startDate is available for future use (e.g., progress calculation)
    startDate: _startDate,
    endDate,
    onTick,
    onComplete,
    adaptiveQuality = true,
    targetFps = DEFAULT_TARGET_FPS,
  } = options;
  void _startDate; // Suppress unused warning - available for future use

  // Animation state
  const [fps, setFps] = useState(targetFps);
  const [frameCount, setFrameCount] = useState(0);
  const [quality, setQuality] = useState<QualitySettings>(QUALITY_PRESETS.high);
  const [isAnimating, setIsAnimating] = useState(false);

  // Refs for animation loop
  const frameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const lastTickTimeRef = useRef<number>(0);
  const lastFpsUpdateRef = useRef<number>(0);
  const frameCountRef = useRef(0);
  const fpsFramesRef = useRef(0);

  // Calculate tick interval based on speed
  const tickInterval = useMemo(() => {
    return Math.max(
      MODEL_CONFIG.LIMITS.MIN_TICK_INTERVAL_MS,
      TICK_INTERVAL_BASE_MS / speed
    );
  }, [speed]);

  // Animation loop
  const animate = useCallback(
    (timestamp: number) => {
      // Start frame timing
      const endFrame = replayPerfMonitor.startFrame();

      // Track frame timing (delta available for future use like animation smoothing)
      lastFrameTimeRef.current = timestamp;

      // Increment frame count
      frameCountRef.current++;
      fpsFramesRef.current++;

      // Update FPS display periodically
      if (timestamp - lastFpsUpdateRef.current >= FPS_UPDATE_INTERVAL_MS) {
        const elapsed = timestamp - lastFpsUpdateRef.current;
        const currentFps = (fpsFramesRef.current / elapsed) * 1000;
        setFps(Math.round(currentFps));
        fpsFramesRef.current = 0;
        lastFpsUpdateRef.current = timestamp;

        // Update quality if adaptive
        if (adaptiveQuality) {
          const stats = replayPerfMonitor.getStats();
          const newQuality = getAdaptiveQuality(stats.avgFrameTime);
          setQuality(newQuality);
        }
      }

      // Check if enough time has passed for a tick
      const timeSinceLastTick = timestamp - lastTickTimeRef.current;
      if (timeSinceLastTick >= tickInterval) {
        lastTickTimeRef.current = timestamp;
        onTick(timeSinceLastTick);
      }

      // Check if replay should complete
      if (currentDate >= endDate) {
        endFrame();
        onComplete?.();
        return;
      }

      // End frame timing
      endFrame();

      // Schedule next frame if still playing
      if (isPlaying) {
        frameRef.current = requestAnimationFrame(animate);
      }
    },
    [isPlaying, tickInterval, currentDate, endDate, onTick, onComplete, adaptiveQuality]
  );

  // Start/stop animation based on isPlaying
  useEffect(() => {
    if (isPlaying) {
      setIsAnimating(true);
      lastFrameTimeRef.current = performance.now();
      lastTickTimeRef.current = performance.now();
      lastFpsUpdateRef.current = performance.now();
      frameRef.current = requestAnimationFrame(animate);
    } else {
      setIsAnimating(false);
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    }

    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [isPlaying, animate]);

  // Update frame count state periodically (not every frame to avoid re-renders)
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setFrameCount(frameCountRef.current);
    }, 1000);

    return () => clearInterval(interval);
  }, [isPlaying]);

  // Reset animation
  const resetAnimation = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    lastFrameTimeRef.current = 0;
    lastTickTimeRef.current = 0;
    frameCountRef.current = 0;
    fpsFramesRef.current = 0;
    replayPerfMonitor.reset();
    setFrameCount(0);
    setFps(targetFps);
    setQuality(QUALITY_PRESETS.high);
    setIsAnimating(false);
  }, [targetFps]);

  // Determine quality level
  const qualityLevel = useMemo(() => {
    if (quality.maxVisibleCandles >= QUALITY_PRESETS.high.maxVisibleCandles) return 'high';
    if (quality.maxVisibleCandles >= QUALITY_PRESETS.medium.maxVisibleCandles) return 'medium';
    return 'low';
  }, [quality]);

  return {
    fps,
    frameCount,
    quality,
    qualityLevel,
    isAnimating,
    resetAnimation,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// TICK SCHEDULER HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Alternative tick-based scheduler (non-RAF) for simpler use cases
 */
export function useReplayTicker(
  isPlaying: boolean,
  speed: ReplaySpeed,
  onTick: () => void,
  baseIntervalMs: number = 100
): void {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (isPlaying) {
      const interval = Math.max(
        MODEL_CONFIG.LIMITS.MIN_TICK_INTERVAL_MS,
        baseIntervalMs / speed
      );
      intervalRef.current = setInterval(onTick, interval);
    } else {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isPlaying, speed, onTick, baseIntervalMs]);
}

// ═══════════════════════════════════════════════════════════════════════════
// FRAME SKIP LOGIC
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Determine if a frame should be skipped based on quality settings
 */
export function useFrameSkip(
  quality: QualitySettings,
  frameCount: number
): { shouldSkipChart: boolean; shouldSkipMetrics: boolean } {
  return useMemo(() => ({
    shouldSkipChart: frameCount % quality.chartUpdateFrequency !== 0,
    shouldSkipMetrics: frameCount % quality.metricsUpdateFrequency !== 0,
  }), [quality, frameCount]);
}

// ═══════════════════════════════════════════════════════════════════════════
// TRADE HIGHLIGHT HOOK
// ═══════════════════════════════════════════════════════════════════════════

interface HighlightedTrade {
  tradeId: string;
  expiresAt: number;
}

/**
 * Manage highlighted trades with automatic expiration
 */
export function useTradeHighlight(
  highlightDuration: number = 2000
): {
  highlightedTradeIds: Set<string>;
  highlightTrade: (tradeId: string) => void;
  clearHighlights: () => void;
} {
  const [highlights, setHighlights] = useState<HighlightedTrade[]>([]);

  // Clean up expired highlights
  useEffect(() => {
    if (highlights.length === 0) return;

    const interval = setInterval(() => {
      const now = Date.now();
      setHighlights(prev => prev.filter(h => h.expiresAt > now));
    }, 500);

    return () => clearInterval(interval);
  }, [highlights.length]);

  const highlightTrade = useCallback((tradeId: string) => {
    setHighlights(prev => {
      // Remove existing highlight for this trade
      const filtered = prev.filter(h => h.tradeId !== tradeId);
      // Add new highlight
      return [...filtered, { tradeId, expiresAt: Date.now() + highlightDuration }];
    });
  }, [highlightDuration]);

  const clearHighlights = useCallback(() => {
    setHighlights([]);
  }, []);

  const highlightedTradeIds = useMemo(
    () => new Set(highlights.map(h => h.tradeId)),
    [highlights]
  );

  return {
    highlightedTradeIds,
    highlightTrade,
    clearHighlights,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// VISIBILITY-BASED PAUSE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Pause replay when tab is not visible
 */
export function useVisibilityPause(
  isPlaying: boolean,
  onPause: () => void,
  onResume: () => void
): void {
  const wasPlayingRef = useRef(false);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        if (isPlaying) {
          wasPlayingRef.current = true;
          onPause();
        }
      } else {
        if (wasPlayingRef.current) {
          wasPlayingRef.current = false;
          onResume();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [isPlaying, onPause, onResume]);
}

// ═══════════════════════════════════════════════════════════════════════════
// HYBRID REPLAY - TIMELINE-BASED ANIMATION
// ═══════════════════════════════════════════════════════════════════════════

import {
  GeneratedTimeline,
  SpeedConfig,
  SPEED_CONFIGS,
} from '@/types/replay';

export interface UseTimelineAnimationOptions {
  isPlaying: boolean;
  speed: ReplaySpeed;
  timeline: GeneratedTimeline | null;
  progress: number;
  onProgressChange: (progress: number) => void;
  onComplete?: () => void;
  adaptiveQuality?: boolean;
}

export interface UseTimelineAnimationReturn {
  fps: number;
  isAnimating: boolean;
  isPausedOnTrade: boolean;
  resetAnimation: () => void;
}

/**
 * Hook for timeline-based replay animation
 * Advances progress based on timeline durations, not wall clock time
 */
export function useTimelineAnimation(
  options: UseTimelineAnimationOptions
): UseTimelineAnimationReturn {
  const {
    isPlaying,
    speed,
    timeline,
    progress,
    onProgressChange,
    onComplete,
    adaptiveQuality = true,
  } = options;

  const [fps, setFps] = useState(60);
  const [isAnimating, setIsAnimating] = useState(false);
  const [isPausedOnTrade, setIsPausedOnTrade] = useState(false);

  const frameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const lastFpsUpdateRef = useRef<number>(0);
  const fpsFramesRef = useRef(0);

  // Get speed config
  const speedConfig = useMemo(() => SPEED_CONFIGS[speed], [speed]);

  // Calculate progress increment per millisecond
  const progressPerMs = useMemo(() => {
    if (!timeline || timeline.totalDurationMs === 0) return 0;
    // 100% progress over totalDurationMs, adjusted by speed
    return (100 / timeline.totalDurationMs) * speed;
  }, [timeline, speed]);

  // Animation loop
  const animate = useCallback(
    (timestamp: number) => {
      if (!timeline) return;

      const deltaMs = lastFrameTimeRef.current > 0
        ? timestamp - lastFrameTimeRef.current
        : 16; // Default to ~60fps for first frame
      lastFrameTimeRef.current = timestamp;

      // Update FPS
      fpsFramesRef.current++;
      if (timestamp - lastFpsUpdateRef.current >= 500) {
        const elapsed = timestamp - lastFpsUpdateRef.current;
        const currentFps = (fpsFramesRef.current / elapsed) * 1000;
        setFps(Math.round(currentFps));
        fpsFramesRef.current = 0;
        lastFpsUpdateRef.current = timestamp;
      }

      // Calculate new progress
      const progressIncrement = progressPerMs * deltaMs;
      const newProgress = Math.min(100, progress + progressIncrement);

      // Check if we're paused on a trade
      // This logic can be extended to actually pause based on tick type
      const currentTimeMs = (progress / 100) * timeline.totalDurationMs;
      let accumulatedTime = 0;
      for (const tick of timeline.ticks) {
        if (accumulatedTime + tick.duration >= currentTimeMs) {
          const isPause = tick.type === 'TRADE_PAUSE' || tick.type === 'GROUP_PAUSE';
          setIsPausedOnTrade(isPause);
          break;
        }
        accumulatedTime += tick.duration;
      }

      // Update progress
      onProgressChange(newProgress);

      // Check completion
      if (newProgress >= 100) {
        onComplete?.();
        return;
      }

      // Schedule next frame
      if (isPlaying) {
        frameRef.current = requestAnimationFrame(animate);
      }
    },
    [isPlaying, timeline, progress, progressPerMs, onProgressChange, onComplete]
  );

  // Start/stop animation
  useEffect(() => {
    if (isPlaying && timeline) {
      setIsAnimating(true);
      lastFrameTimeRef.current = performance.now();
      lastFpsUpdateRef.current = performance.now();
      frameRef.current = requestAnimationFrame(animate);
    } else {
      setIsAnimating(false);
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    }

    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [isPlaying, timeline, animate]);

  // Reset
  const resetAnimation = useCallback(() => {
    if (frameRef.current !== null) {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = null;
    }
    lastFrameTimeRef.current = 0;
    lastFpsUpdateRef.current = 0;
    fpsFramesRef.current = 0;
    setFps(60);
    setIsAnimating(false);
    setIsPausedOnTrade(false);
  }, []);

  return {
    fps,
    isAnimating,
    isPausedOnTrade,
    resetAnimation,
  };
}
