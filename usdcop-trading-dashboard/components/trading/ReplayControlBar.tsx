/**
 * Replay Control Bar Component
 *
 * Provides the main UI for controlling the replay system:
 * - Play/Pause/Stop buttons
 * - Progress slider
 * - Speed selector
 * - Mode selector (validation/test/both)
 * - Date range display
 * - Performance indicators
 */

'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  ReplayState,
  ReplaySpeed,
  ReplayMode,
  formatReplayDate,
} from '@/types/replay';
import { getFormattedShortcuts, getModeDisplayName } from '@/hooks/useReplayKeyboard';

// ═══════════════════════════════════════════════════════════════════════════
// ICONS (inline SVG for minimal dependencies)
// ═══════════════════════════════════════════════════════════════════════════

const PlayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M8 5v14l11-7z" />
  </svg>
);

const PauseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
  </svg>
);

const StopIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M6 6h12v12H6z" />
  </svg>
);

const ReplayIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z" />
  </svg>
);

const KeyboardIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <path d="M20 5H4c-1.1 0-1.99.9-1.99 2L2 17c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm-9 3h2v2h-2V8zm0 3h2v2h-2v-2zM8 8h2v2H8V8zm0 3h2v2H8v-2zm-1 2H5v-2h2v2zm0-3H5V8h2v2zm9 7H8v-2h8v2zm0-4h-2v-2h2v2zm0-3h-2V8h2v2zm3 3h-2v-2h2v2zm0-3h-2V8h2v2z" />
  </svg>
);

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface ReplayControlBarProps {
  state: ReplayState;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onSeek: (progress: number) => void;
  onSetSpeed: (speed: ReplaySpeed) => void;
  onSetMode: (mode: ReplayMode) => void;
  fps?: number;
  qualityLevel?: 'high' | 'medium' | 'low';
  tradeCount?: number;
  isLoading?: boolean;
  className?: string;
  // Hybrid replay props
  estimatedDuration?: string;
  groupCount?: number;
  isPausedOnTrade?: boolean;
  currentTradeId?: string | null;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════

interface SpeedButtonProps {
  speed: ReplaySpeed;
  currentSpeed: ReplaySpeed;
  onClick: (speed: ReplaySpeed) => void;
}

function SpeedButton({ speed, currentSpeed, onClick }: SpeedButtonProps) {
  const isActive = speed === currentSpeed;
  return (
    <button
      onClick={() => onClick(speed)}
      className={cn(
        'px-2 py-1 text-xs font-mono rounded transition-all duration-200',
        isActive
          ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400/50'
          : 'bg-slate-800/50 text-slate-400 border border-slate-700/50 hover:text-slate-200 hover:border-slate-600'
      )}
    >
      {speed}x
    </button>
  );
}

interface ModeButtonProps {
  mode: ReplayMode;
  currentMode: ReplayMode;
  onClick: (mode: ReplayMode) => void;
}

function ModeButton({ mode, currentMode, onClick }: ModeButtonProps) {
  const isActive = mode === currentMode;
  return (
    <button
      onClick={() => onClick(mode)}
      className={cn(
        'px-2 py-1 text-xs rounded transition-all duration-200',
        isActive
          ? 'bg-emerald-500/30 text-emerald-300 border border-emerald-400/50'
          : 'bg-slate-800/50 text-slate-400 border border-slate-700/50 hover:text-slate-200 hover:border-slate-600'
      )}
    >
      {getModeDisplayName(mode)}
    </button>
  );
}

interface ProgressSliderProps {
  progress: number;
  onChange: (progress: number) => void;
  disabled?: boolean;
}

function ProgressSlider({ progress, onChange, disabled }: ProgressSliderProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseFloat(e.target.value));
  };

  return (
    <div className="relative flex-1 h-2 group">
      {/* Track background */}
      <div className="absolute inset-0 bg-slate-700/50 rounded-full" />
      {/* Progress fill */}
      <div
        className="absolute inset-y-0 left-0 bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-full transition-all"
        style={{ width: `${progress}%` }}
      />
      {/* Input */}
      <input
        type="range"
        min="0"
        max="100"
        step="0.1"
        value={progress}
        onChange={handleChange}
        disabled={disabled}
        className={cn(
          'absolute inset-0 w-full h-full opacity-0 cursor-pointer',
          disabled && 'cursor-not-allowed'
        )}
      />
      {/* Thumb indicator */}
      <div
        className={cn(
          'absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-lg transition-all',
          'group-hover:scale-125 group-hover:shadow-cyan-500/50',
          disabled && 'opacity-50'
        )}
        style={{ left: `calc(${progress}% - 6px)` }}
      />
    </div>
  );
}

function KeyboardShortcutsTooltip() {
  const [isOpen, setIsOpen] = useState(false);
  const shortcuts = getFormattedShortcuts();

  return (
    <div className="relative">
      <button
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        className="p-1.5 text-slate-400 hover:text-slate-200 rounded transition-colors"
        title="Atajos de teclado"
      >
        <KeyboardIcon />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            className="absolute bottom-full right-0 mb-2 w-56 p-3 bg-slate-900/95 border border-slate-700/50 rounded-lg shadow-xl z-50"
          >
            <div className="text-xs font-medium text-slate-300 mb-2">Atajos de Teclado</div>
            <div className="space-y-1">
              {shortcuts.slice(0, 8).map((shortcut, i) => (
                <div key={i} className="flex justify-between text-xs">
                  <span className="text-slate-400">{shortcut.description}</span>
                  <kbd className="px-1.5 py-0.5 bg-slate-800 rounded text-slate-300 font-mono">
                    {shortcut.key}
                  </kbd>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

export function ReplayControlBar({
  state,
  onPlay,
  onPause,
  onStop,
  onSeek,
  onSetSpeed,
  onSetMode,
  fps,
  qualityLevel,
  tradeCount,
  isLoading = false,
  className,
  // Hybrid replay props
  estimatedDuration,
  groupCount,
  isPausedOnTrade,
  currentTradeId,
}: ReplayControlBarProps) {
  const { status, isPlaying, speed, mode, progress, currentDate, startDate, endDate } = state;

  // Determine button states (include 'idle' to allow initial load+play)
  const canPlay = ['idle', 'ready', 'paused', 'completed'].includes(status);
  const canPause = status === 'playing';
  const canStop = ['playing', 'paused', 'completed'].includes(status);
  const canSeek = ['ready', 'playing', 'paused', 'completed'].includes(status);

  // Handle play/pause toggle
  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      onPause();
    } else {
      onPlay();
    }
  }, [isPlaying, onPlay, onPause]);

  // Format dates for display
  const formattedStart = useMemo(() => formatReplayDate(startDate), [startDate]);
  const formattedEnd = useMemo(() => formatReplayDate(endDate), [endDate]);
  const formattedCurrent = useMemo(() => formatReplayDate(currentDate), [currentDate]);

  // Quality indicator color
  const qualityColor = useMemo(() => {
    switch (qualityLevel) {
      case 'high': return 'text-emerald-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-red-400';
      default: return 'text-slate-400';
    }
  }, [qualityLevel]);

  return (
    <div
      className={cn(
        'flex flex-col gap-2 p-3',
        'bg-slate-900/95 backdrop-blur-xl',
        'border border-slate-700/50 rounded-xl',
        'shadow-xl shadow-cyan-500/5',
        className
      )}
    >
      {/* Compact single row layout */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Playback controls */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={handlePlayPause}
            disabled={!canPlay && !canPause}
            className={cn(
              'p-1.5 rounded-lg transition-all',
              isPlaying
                ? 'bg-cyan-500/30 text-cyan-400 hover:bg-cyan-500/40'
                : 'bg-slate-700/50 text-slate-300 hover:bg-slate-600'
            )}
          >
            {isPlaying ? <PauseIcon /> : <PlayIcon />}
          </button>
          <button
            onClick={onStop}
            disabled={!canStop}
            className="p-1.5 rounded-lg bg-slate-700/50 text-slate-400 hover:bg-slate-600 transition-colors disabled:opacity-40"
          >
            <StopIcon />
          </button>
        </div>

        {/* Divider */}
        <div className="w-px h-5 bg-slate-700" />

        {/* Mode selector */}
        <div className="flex items-center gap-1">
          {(['validation', 'test', 'both'] as ReplayMode[]).map((m) => (
            <ModeButton key={m} mode={m} currentMode={mode} onClick={onSetMode} />
          ))}
        </div>

        {/* Divider */}
        <div className="w-px h-5 bg-slate-700" />

        {/* Progress */}
        <div className="flex items-center gap-2 flex-1 min-w-[120px] max-w-[200px]">
          <span className="text-[10px] font-mono text-cyan-400">{formattedCurrent}</span>
          <ProgressSlider
            progress={progress}
            onChange={onSeek}
            disabled={!canSeek || isLoading}
          />
          <span className="text-[10px] font-mono text-slate-400">{progress.toFixed(0)}%</span>
        </div>

        {/* Divider */}
        <div className="w-px h-5 bg-slate-700" />

        {/* Speed */}
        <div className="flex items-center gap-0.5">
          {([1, 2, 4, 8] as ReplaySpeed[]).map((s) => (
            <SpeedButton key={s} speed={s} currentSpeed={speed} onClick={onSetSpeed} />
          ))}
        </div>

        {/* Divider */}
        <div className="w-px h-5 bg-slate-700" />

        {/* Status & info */}
        <div className="flex items-center gap-2">
          {tradeCount !== undefined && (
            <span className="text-[10px] text-slate-400">
              <span className="text-slate-300">{tradeCount}</span> trades
            </span>
          )}

          {/* Status badge */}
          <div
            className={cn(
              'px-1.5 py-0.5 text-[10px] rounded-full font-medium',
              status === 'playing' && 'bg-emerald-500/20 text-emerald-400',
              status === 'paused' && 'bg-yellow-500/20 text-yellow-400',
              status === 'loading' && 'bg-cyan-500/20 text-cyan-400 animate-pulse',
              status === 'ready' && 'bg-slate-500/20 text-slate-400',
              status === 'completed' && 'bg-purple-500/20 text-purple-400',
              status === 'error' && 'bg-red-500/20 text-red-400',
              status === 'idle' && 'bg-slate-600/20 text-slate-500'
            )}
          >
            {status === 'playing' && 'Playing'}
            {status === 'paused' && 'Paused'}
            {status === 'loading' && 'Loading'}
            {status === 'ready' && 'Ready'}
            {status === 'completed' && 'Done'}
            {status === 'error' && 'Error'}
            {status === 'idle' && 'Idle'}
          </div>

          <KeyboardShortcutsTooltip />
        </div>
      </div>

      {/* Error message */}
      <AnimatePresence>
        {state.error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2"
          >
            {state.error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPACT VARIANT
// ═══════════════════════════════════════════════════════════════════════════

export interface ReplayControlBarCompactProps {
  state: ReplayState;
  onPlayPause: () => void;
  onStop: () => void;
  onCycleSpeed: () => void;
  className?: string;
}

export function ReplayControlBarCompact({
  state,
  onPlayPause,
  onStop,
  onCycleSpeed,
  className,
}: ReplayControlBarCompactProps) {
  const { status, isPlaying, speed, progress, currentDate } = state;
  const canPlay = ['idle', 'ready', 'paused', 'completed'].includes(status);

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-3 py-2',
        'bg-slate-900/70 backdrop-blur-md',
        'border border-slate-700/50 rounded-lg',
        className
      )}
    >
      {/* Play/Pause */}
      <button
        onClick={onPlayPause}
        disabled={!canPlay && !isPlaying}
        className={cn(
          'p-1.5 rounded-lg transition-colors',
          isPlaying
            ? 'bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30'
            : 'bg-slate-700/50 text-slate-300 hover:bg-slate-700'
        )}
      >
        {isPlaying ? <PauseIcon /> : <PlayIcon />}
      </button>

      {/* Stop */}
      <button
        onClick={onStop}
        className="p-1.5 rounded-lg bg-slate-700/50 text-slate-300 hover:bg-slate-700 transition-colors"
      >
        <StopIcon />
      </button>

      {/* Progress */}
      <div className="flex-1 h-1 bg-slate-700/50 rounded-full">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-full"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Speed */}
      <button
        onClick={onCycleSpeed}
        className="px-2 py-0.5 text-xs font-mono bg-slate-700/50 text-slate-300 rounded hover:bg-slate-700 transition-colors"
      >
        {speed}x
      </button>

      {/* Current date */}
      <span className="text-xs font-mono text-slate-400">
        {formatReplayDate(currentDate)}
      </span>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// FLOATING MINI VARIANT - Ultra-compact floating pill
// ═══════════════════════════════════════════════════════════════════════════

const ChevronUpIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M18 15l-6-6-6 6" />
  </svg>
);

const ChevronDownIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M6 9l6 6 6-6" />
  </svg>
);

export interface ReplayControlBarFloatingProps {
  state: ReplayState;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onSeek: (progress: number) => void;
  onSetSpeed: (speed: ReplaySpeed) => void;
  onSetMode: (mode: ReplayMode) => void;
  onExpand: () => void;
  tradeCount?: number;
  className?: string;
}

export function ReplayControlBarFloating({
  state,
  onPlay,
  onPause,
  onStop,
  onSeek,
  onSetSpeed,
  onSetMode,
  onExpand,
  tradeCount,
  className,
}: ReplayControlBarFloatingProps) {
  const { status, isPlaying, speed, progress, currentDate, mode } = state;
  const canPlay = ['idle', 'ready', 'paused', 'completed'].includes(status);
  const canStop = ['playing', 'paused', 'completed'].includes(status);

  const handlePlayPause = useCallback(() => {
    if (isPlaying) {
      onPause();
    } else {
      onPlay();
    }
  }, [isPlaying, onPlay, onPause]);

  const cycleSpeed = useCallback(() => {
    const speeds: ReplaySpeed[] = [0.5, 1, 2, 4, 8, 16];
    const currentIndex = speeds.indexOf(speed);
    const nextIndex = (currentIndex + 1) % speeds.length;
    onSetSpeed(speeds[nextIndex]);
  }, [speed, onSetSpeed]);

  const cycleMode = useCallback(() => {
    const modes: ReplayMode[] = ['validation', 'test', 'both'];
    const currentIndex = modes.indexOf(mode);
    const nextIndex = (currentIndex + 1) % modes.length;
    onSetMode(modes[nextIndex]);
  }, [mode, onSetMode]);

  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn(
        'inline-flex items-center gap-2 px-3 py-2',
        'bg-slate-900/95 backdrop-blur-xl',
        'border border-cyan-500/30 rounded-full',
        'shadow-lg shadow-cyan-500/10',
        className
      )}
    >
      {/* Status dot */}
      <div className={cn(
        'w-2 h-2 rounded-full',
        isPlaying && 'bg-emerald-400 animate-pulse',
        status === 'paused' && 'bg-yellow-400',
        status === 'completed' && 'bg-purple-400',
        status === 'loading' && 'bg-cyan-400 animate-pulse',
        ['idle', 'ready'].includes(status) && 'bg-slate-500'
      )} />

      {/* Mode badge */}
      <button
        onClick={cycleMode}
        className="px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide rounded bg-slate-800/80 text-slate-300 hover:bg-slate-700 transition-colors"
        title="Click to change mode"
      >
        {mode === 'validation' ? 'Val' : mode === 'test' ? 'Test' : 'All'}
      </button>

      {/* Divider */}
      <div className="w-px h-4 bg-slate-700" />

      {/* Play/Pause */}
      <button
        onClick={handlePlayPause}
        disabled={!canPlay && !isPlaying}
        className={cn(
          'p-1 rounded-full transition-all duration-200',
          isPlaying
            ? 'bg-cyan-500/30 text-cyan-400 hover:bg-cyan-500/40'
            : 'bg-slate-700/50 text-slate-300 hover:bg-slate-600'
        )}
      >
        {isPlaying ? <PauseIcon /> : <PlayIcon />}
      </button>

      {/* Stop */}
      <button
        onClick={onStop}
        disabled={!canStop}
        className="p-1 rounded-full bg-slate-700/50 text-slate-400 hover:bg-slate-600 hover:text-slate-200 transition-colors disabled:opacity-40"
      >
        <StopIcon />
      </button>

      {/* Divider */}
      <div className="w-px h-4 bg-slate-700" />

      {/* Mini progress */}
      <div className="w-16 h-1.5 bg-slate-700/50 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all duration-100"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Progress % */}
      <span className="text-[10px] font-mono text-cyan-400 w-8">
        {progress.toFixed(0)}%
      </span>

      {/* Divider */}
      <div className="w-px h-4 bg-slate-700" />

      {/* Speed */}
      <button
        onClick={cycleSpeed}
        className="px-1.5 py-0.5 text-[10px] font-mono bg-slate-800/80 text-slate-300 rounded hover:bg-slate-700 transition-colors"
        title="Click to change speed"
      >
        {speed}x
      </button>

      {/* Trade count */}
      {tradeCount !== undefined && tradeCount > 0 && (
        <>
          <div className="w-px h-4 bg-slate-700" />
          <span className="text-[10px] text-slate-400">
            <span className="text-slate-300">{tradeCount}</span> trades
          </span>
        </>
      )}

      {/* Divider */}
      <div className="w-px h-4 bg-slate-700" />

      {/* Current date - compact */}
      <span className="text-[10px] font-mono text-slate-400">
        {formatReplayDate(currentDate)}
      </span>

      {/* Expand button */}
      <button
        onClick={onExpand}
        className="p-1 rounded-full hover:bg-slate-700/50 text-slate-400 hover:text-slate-200 transition-colors"
        title="Expand controls"
      >
        <ChevronDownIcon />
      </button>
    </motion.div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// COLLAPSIBLE WRAPPER - Manages expanded/collapsed state
// ═══════════════════════════════════════════════════════════════════════════

export interface ReplayControlBarCollapsibleProps extends Omit<ReplayControlBarProps, 'className'> {
  defaultExpanded?: boolean;
  className?: string;
}

export function ReplayControlBarCollapsible({
  defaultExpanded = false,
  className,
  ...props
}: ReplayControlBarCollapsibleProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className={cn('relative inline-block', className)}>
      <AnimatePresence mode="wait">
        {isExpanded ? (
          <motion.div
            key="expanded"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <div className="relative max-w-xl">
              <ReplayControlBar {...props} className="text-xs" />
              {/* Collapse button */}
              <button
                onClick={() => setIsExpanded(false)}
                className="absolute -top-2 right-2 p-1 rounded-full bg-slate-800 border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-700 transition-colors shadow-lg"
                title="Collapse to mini mode"
              >
                <ChevronUpIcon />
              </button>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="collapsed"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            <ReplayControlBarFloating
              state={props.state}
              onPlay={props.onPlay}
              onPause={props.onPause}
              onStop={props.onStop}
              onSeek={props.onSeek}
              onSetSpeed={props.onSetSpeed}
              onSetMode={props.onSetMode}
              onExpand={() => setIsExpanded(true)}
              tradeCount={props.tradeCount}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default ReplayControlBar;
