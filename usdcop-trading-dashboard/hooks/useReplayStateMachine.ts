/**
 * Replay State Machine Hook
 *
 * Provides type-safe state management for the replay system using
 * a reducer pattern with explicit state transitions.
 */

import { useReducer, useCallback, useMemo } from 'react';
import {
  ReplayState,
  ReplayStatus,
  ReplaySpeed,
  ReplayMode,
  MODEL_CONFIG,
  getModeRange,
  calculateProgress,
} from '@/types/replay';

// ═══════════════════════════════════════════════════════════════════════════
// ACTION TYPES
// ═══════════════════════════════════════════════════════════════════════════

export type ReplayAction =
  | { type: 'START_LOADING' }
  | { type: 'LOAD_SUCCESS' }
  | { type: 'LOAD_ERROR'; error: string }
  | { type: 'PLAY' }
  | { type: 'PAUSE' }
  | { type: 'STOP' }
  | { type: 'COMPLETE' }
  | { type: 'RESET' }
  | { type: 'SET_SPEED'; speed: ReplaySpeed }
  | { type: 'SET_MODE'; mode: ReplayMode }
  | { type: 'SET_DATE_RANGE'; startDate: Date; endDate: Date }
  | { type: 'SET_CURRENT_DATE'; date: Date }
  | { type: 'TICK'; deltaMs: number }
  | { type: 'SEEK'; progress: number }
  | { type: 'JUMP_TO_DATE'; date: Date };

// ═══════════════════════════════════════════════════════════════════════════
// STATE TRANSITION MATRIX
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Defines valid state transitions.
 * If a transition is not listed, it's not allowed.
 */
const VALID_TRANSITIONS: Record<ReplayStatus, ReplayStatus[]> = {
  idle: ['loading'],
  loading: ['ready', 'error'],
  ready: ['playing', 'loading', 'idle'],
  playing: ['paused', 'completed', 'error', 'ready'],
  paused: ['playing', 'ready', 'idle'],
  completed: ['ready', 'playing', 'idle'],
  error: ['idle', 'loading'],
};

/**
 * Check if a transition is valid
 */
function isValidTransition(from: ReplayStatus, to: ReplayStatus): boolean {
  return VALID_TRANSITIONS[from]?.includes(to) ?? false;
}

// ═══════════════════════════════════════════════════════════════════════════
// INITIAL STATE
// ═══════════════════════════════════════════════════════════════════════════

function createInitialState(mode: ReplayMode = 'validation'): ReplayState {
  const range = getModeRange(mode);
  return {
    status: 'idle',
    startDate: range.start,
    endDate: range.end,
    currentDate: range.start,
    isPlaying: false,
    speed: 1,
    mode,
    progress: 0,
    error: null,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// REDUCER
// ═══════════════════════════════════════════════════════════════════════════

function replayReducer(state: ReplayState, action: ReplayAction): ReplayState {
  switch (action.type) {
    case 'START_LOADING': {
      if (!isValidTransition(state.status, 'loading')) {
        console.warn(`[Replay] Invalid transition: ${state.status} -> loading`);
        return state;
      }
      return {
        ...state,
        status: 'loading',
        error: null,
      };
    }

    case 'LOAD_SUCCESS': {
      if (!isValidTransition(state.status, 'ready')) {
        console.warn(`[Replay] Invalid transition: ${state.status} -> ready`);
        return state;
      }
      return {
        ...state,
        status: 'ready',
        error: null,
      };
    }

    case 'LOAD_ERROR': {
      if (!isValidTransition(state.status, 'error')) {
        console.warn(`[Replay] Invalid transition: ${state.status} -> error`);
        return state;
      }
      return {
        ...state,
        status: 'error',
        isPlaying: false,
        error: action.error,
      };
    }

    case 'PLAY': {
      if (!isValidTransition(state.status, 'playing')) {
        console.warn(`[Replay] Invalid transition: ${state.status} -> playing`);
        return state;
      }
      return {
        ...state,
        status: 'playing',
        isPlaying: true,
      };
    }

    case 'PAUSE': {
      if (!isValidTransition(state.status, 'paused')) {
        console.warn(`[Replay] Invalid transition: ${state.status} -> paused`);
        return state;
      }
      return {
        ...state,
        status: 'paused',
        isPlaying: false,
      };
    }

    case 'STOP': {
      return {
        ...state,
        status: 'ready',
        isPlaying: false,
        currentDate: state.startDate,
        progress: 0,
      };
    }

    case 'COMPLETE': {
      if (!isValidTransition(state.status, 'completed')) {
        console.warn(`[Replay] Invalid transition: ${state.status} -> completed`);
        return state;
      }
      return {
        ...state,
        status: 'completed',
        isPlaying: false,
        currentDate: state.endDate,
        progress: 100,
      };
    }

    case 'RESET': {
      return createInitialState(state.mode);
    }

    case 'SET_SPEED': {
      return {
        ...state,
        speed: action.speed,
      };
    }

    case 'SET_MODE': {
      const range = getModeRange(action.mode);
      return {
        ...state,
        mode: action.mode,
        startDate: range.start,
        endDate: range.end,
        currentDate: range.start,
        progress: 0,
        status: state.status === 'completed' ? 'ready' : state.status,
      };
    }

    case 'SET_DATE_RANGE': {
      // Validate dates are within model bounds
      const minDate = new Date(MODEL_CONFIG.DATES.VALIDATION_START);
      const maxDate = new Date(MODEL_CONFIG.DATES.TEST_END);

      const startDate = action.startDate < minDate ? minDate : action.startDate;
      const endDate = action.endDate > maxDate ? maxDate : action.endDate;

      // Ensure start <= end
      if (startDate > endDate) {
        console.warn('[Replay] Invalid date range: start > end');
        return state;
      }

      // Adjust currentDate if outside new range
      let currentDate = state.currentDate;
      if (currentDate < startDate) currentDate = startDate;
      if (currentDate > endDate) currentDate = endDate;

      return {
        ...state,
        startDate,
        endDate,
        currentDate,
        progress: calculateProgress(currentDate, startDate, endDate),
        status: state.status === 'completed' ? 'ready' : state.status,
      };
    }

    case 'SET_CURRENT_DATE': {
      const clampedDate = new Date(
        Math.max(
          state.startDate.getTime(),
          Math.min(state.endDate.getTime(), action.date.getTime())
        )
      );
      const progress = calculateProgress(clampedDate, state.startDate, state.endDate);

      // Auto-complete if reached end
      if (progress >= 100 && state.isPlaying) {
        return {
          ...state,
          currentDate: state.endDate,
          progress: 100,
          status: 'completed',
          isPlaying: false,
        };
      }

      return {
        ...state,
        currentDate: clampedDate,
        progress,
      };
    }

    case 'TICK': {
      if (!state.isPlaying) return state;

      // Calculate time to add based on speed
      // 1x = real time, 2x = 2x real time, etc.
      // But we also accelerate time: 1 second of replay = several hours/days of market time
      const MARKET_TIME_PER_TICK_MS = 5 * 60 * 1000; // 5 minutes of market time per tick
      const marketTimeDelta = MARKET_TIME_PER_TICK_MS * state.speed;

      const newTime = state.currentDate.getTime() + marketTimeDelta;
      const newDate = new Date(Math.min(newTime, state.endDate.getTime()));
      const progress = calculateProgress(newDate, state.startDate, state.endDate);

      // Check completion
      if (progress >= 100) {
        return {
          ...state,
          currentDate: state.endDate,
          progress: 100,
          status: 'completed',
          isPlaying: false,
        };
      }

      return {
        ...state,
        currentDate: newDate,
        progress,
      };
    }

    case 'SEEK': {
      const clampedProgress = Math.max(0, Math.min(100, action.progress));
      const totalMs = state.endDate.getTime() - state.startDate.getTime();
      const targetMs = state.startDate.getTime() + (totalMs * clampedProgress) / 100;
      const targetDate = new Date(targetMs);

      return {
        ...state,
        currentDate: targetDate,
        progress: clampedProgress,
        status: clampedProgress >= 100 ? 'completed' : state.status === 'completed' ? 'ready' : state.status,
        isPlaying: clampedProgress >= 100 ? false : state.isPlaying,
      };
    }

    case 'JUMP_TO_DATE': {
      const clampedDate = new Date(
        Math.max(
          state.startDate.getTime(),
          Math.min(state.endDate.getTime(), action.date.getTime())
        )
      );
      const progress = calculateProgress(clampedDate, state.startDate, state.endDate);

      return {
        ...state,
        currentDate: clampedDate,
        progress,
        status: progress >= 100 ? 'completed' : state.status === 'completed' ? 'ready' : state.status,
        isPlaying: progress >= 100 ? false : state.isPlaying,
      };
    }

    default:
      return state;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════

export interface UseReplayStateMachineReturn {
  state: ReplayState;
  dispatch: React.Dispatch<ReplayAction>;

  // Convenience actions
  startLoading: () => void;
  loadSuccess: () => void;
  loadError: (error: string) => void;
  play: () => void;
  pause: () => void;
  togglePlayPause: () => void;
  stop: () => void;
  reset: () => void;
  setSpeed: (speed: ReplaySpeed) => void;
  cycleSpeed: () => void;
  setMode: (mode: ReplayMode) => void;
  setDateRange: (startDate: Date, endDate: Date) => void;
  setCurrentDate: (date: Date) => void;
  tick: (deltaMs: number) => void;
  seek: (progress: number) => void;
  jumpToDate: (date: Date) => void;

  // Computed values
  canPlay: boolean;
  canPause: boolean;
  canSeek: boolean;
  isLoading: boolean;
  hasError: boolean;
  isComplete: boolean;
  formattedProgress: string;
}

export function useReplayStateMachine(
  initialMode: ReplayMode = 'validation'
): UseReplayStateMachineReturn {
  const [state, dispatch] = useReducer(replayReducer, initialMode, createInitialState);

  // Convenience action creators
  const startLoading = useCallback(() => dispatch({ type: 'START_LOADING' }), []);
  const loadSuccess = useCallback(() => dispatch({ type: 'LOAD_SUCCESS' }), []);
  const loadError = useCallback((error: string) => dispatch({ type: 'LOAD_ERROR', error }), []);
  const play = useCallback(() => dispatch({ type: 'PLAY' }), []);
  const pause = useCallback(() => dispatch({ type: 'PAUSE' }), []);
  const stop = useCallback(() => dispatch({ type: 'STOP' }), []);
  const reset = useCallback(() => dispatch({ type: 'RESET' }), []);
  const setSpeed = useCallback((speed: ReplaySpeed) => dispatch({ type: 'SET_SPEED', speed }), []);
  const setMode = useCallback((mode: ReplayMode) => dispatch({ type: 'SET_MODE', mode }), []);
  const setDateRange = useCallback(
    (startDate: Date, endDate: Date) => dispatch({ type: 'SET_DATE_RANGE', startDate, endDate }),
    []
  );
  const setCurrentDate = useCallback(
    (date: Date) => dispatch({ type: 'SET_CURRENT_DATE', date }),
    []
  );
  const tick = useCallback((deltaMs: number) => dispatch({ type: 'TICK', deltaMs }), []);
  const seek = useCallback((progress: number) => dispatch({ type: 'SEEK', progress }), []);
  const jumpToDate = useCallback((date: Date) => dispatch({ type: 'JUMP_TO_DATE', date }), []);

  const togglePlayPause = useCallback(() => {
    if (state.isPlaying) {
      pause();
    } else if (state.status === 'completed') {
      // Restart from beginning
      dispatch({ type: 'SEEK', progress: 0 });
      setTimeout(play, 0);
    } else {
      play();
    }
  }, [state.isPlaying, state.status, pause, play]);

  const cycleSpeed = useCallback(() => {
    const speeds: ReplaySpeed[] = [1, 2, 4, 8];
    const currentIndex = speeds.indexOf(state.speed);
    const nextIndex = (currentIndex + 1) % speeds.length;
    setSpeed(speeds[nextIndex]);
  }, [state.speed, setSpeed]);

  // Computed values
  const computed = useMemo(() => ({
    canPlay: ['ready', 'paused', 'completed'].includes(state.status),
    canPause: state.status === 'playing',
    canSeek: ['ready', 'playing', 'paused', 'completed'].includes(state.status),
    isLoading: state.status === 'loading',
    hasError: state.status === 'error',
    isComplete: state.status === 'completed',
    formattedProgress: `${state.progress.toFixed(1)}%`,
  }), [state.status, state.progress]);

  return {
    state,
    dispatch,
    startLoading,
    loadSuccess,
    loadError,
    play,
    pause,
    togglePlayPause,
    stop,
    reset,
    setSpeed,
    cycleSpeed,
    setMode,
    setDateRange,
    setCurrentDate,
    tick,
    seek,
    jumpToDate,
    ...computed,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// SELECTORS (for external use)
// ═══════════════════════════════════════════════════════════════════════════

export function selectProgress(state: ReplayState): number {
  return state.progress;
}

export function selectIsPlaying(state: ReplayState): boolean {
  return state.isPlaying;
}

export function selectCurrentDate(state: ReplayState): Date {
  return state.currentDate;
}

export function selectDateRange(state: ReplayState): { start: Date; end: Date } {
  return { start: state.startDate, end: state.endDate };
}

export function selectSpeed(state: ReplayState): ReplaySpeed {
  return state.speed;
}

export function selectMode(state: ReplayState): ReplayMode {
  return state.mode;
}
