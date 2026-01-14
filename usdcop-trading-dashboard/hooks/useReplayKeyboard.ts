/**
 * Replay Keyboard Shortcuts Hook
 *
 * Provides keyboard navigation for the replay system:
 * - Space: Play/Pause toggle
 * - Escape: Stop replay
 * - Arrow keys: Seek forward/backward
 * - Number keys 1-4: Set speed
 * - M: Cycle through modes
 */

import { useEffect, useCallback, useRef } from 'react';
import { ReplaySpeed, ReplayMode } from '@/types/replay';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

export interface ReplayKeyboardHandlers {
  onPlayPause: () => void;
  onStop: () => void;
  onSeekForward: (amount: number) => void;
  onSeekBackward: (amount: number) => void;
  onSetSpeed: (speed: ReplaySpeed) => void;
  onCycleMode: () => void;
  onReset: () => void;
  // Hybrid replay navigation
  onNextTrade?: () => void;
  onPrevTrade?: () => void;
}

export interface UseReplayKeyboardOptions {
  enabled?: boolean;
  seekAmount?: number; // Percentage to seek (default 5%)
  seekAmountLarge?: number; // Percentage for Shift+Arrow (default 15%)
}

// ═══════════════════════════════════════════════════════════════════════════
// SHORTCUT DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════

export const KEYBOARD_SHORTCUTS = [
  { key: 'Space', description: 'Play / Pause', action: 'playPause' },
  { key: 'Escape', description: 'Stop replay', action: 'stop' },
  { key: '←', description: 'Seek backward 5%', action: 'seekBackward' },
  { key: '→', description: 'Seek forward 5%', action: 'seekForward' },
  { key: 'Shift + ←', description: 'Seek backward 15%', action: 'seekBackwardLarge' },
  { key: 'Shift + →', description: 'Seek forward 15%', action: 'seekForwardLarge' },
  { key: '`', description: 'Speed 0.5x (slow)', action: 'speed0.5' },
  { key: '1', description: 'Speed 1x', action: 'speed1' },
  { key: '2', description: 'Speed 2x', action: 'speed2' },
  { key: '3', description: 'Speed 4x', action: 'speed4' },
  { key: '4', description: 'Speed 8x', action: 'speed8' },
  { key: '5', description: 'Speed 16x (fast)', action: 'speed16' },
  { key: '[', description: 'Previous trade', action: 'prevTrade' },
  { key: ']', description: 'Next trade', action: 'nextTrade' },
  { key: 'M', description: 'Cycle mode', action: 'cycleMode' },
  { key: 'R', description: 'Reset replay', action: 'reset' },
] as const;

// ═══════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════

export function useReplayKeyboard(
  handlers: ReplayKeyboardHandlers,
  options: UseReplayKeyboardOptions = {}
): void {
  const { enabled = true, seekAmount = 5, seekAmountLarge = 15 } = options;

  // Use ref to avoid recreating the event handler
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      // Ignore if typing in input/textarea
      const target = event.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      const { onPlayPause, onStop, onSeekForward, onSeekBackward, onSetSpeed, onCycleMode, onReset, onNextTrade, onPrevTrade } =
        handlersRef.current;

      switch (event.code) {
        case 'Space':
          event.preventDefault();
          onPlayPause();
          break;

        case 'Escape':
          event.preventDefault();
          onStop();
          break;

        case 'ArrowRight':
          event.preventDefault();
          if (event.shiftKey) {
            onSeekForward(seekAmountLarge);
          } else {
            onSeekForward(seekAmount);
          }
          break;

        case 'ArrowLeft':
          event.preventDefault();
          if (event.shiftKey) {
            onSeekBackward(seekAmountLarge);
          } else {
            onSeekBackward(seekAmount);
          }
          break;

        // Speed: 0.5x with backtick (`)
        case 'Backquote':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onSetSpeed(0.5);
          }
          break;

        case 'Digit1':
        case 'Numpad1':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onSetSpeed(1);
          }
          break;

        case 'Digit2':
        case 'Numpad2':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onSetSpeed(2);
          }
          break;

        case 'Digit3':
        case 'Numpad3':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onSetSpeed(4);
          }
          break;

        case 'Digit4':
        case 'Numpad4':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onSetSpeed(8);
          }
          break;

        // Speed: 16x with 5
        case 'Digit5':
        case 'Numpad5':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onSetSpeed(16);
          }
          break;

        // Trade navigation with [ and ]
        case 'BracketLeft':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onPrevTrade?.();
          }
          break;

        case 'BracketRight':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onNextTrade?.();
          }
          break;

        case 'KeyM':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onCycleMode();
          }
          break;

        case 'KeyR':
          if (!event.ctrlKey && !event.metaKey) {
            event.preventDefault();
            onReset();
          }
          break;
      }
    },
    [seekAmount, seekAmountLarge]
  );

  useEffect(() => {
    if (!enabled) return;

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [enabled, handleKeyDown]);
}

// ═══════════════════════════════════════════════════════════════════════════
// SHORTCUT DISPLAY COMPONENT HELPER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Format keyboard shortcut for display (e.g., show ⌘ on Mac, Ctrl on Windows)
 */
export function formatShortcutKey(key: string): string {
  const isMac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform);

  return key
    .replace('Ctrl', isMac ? '⌘' : 'Ctrl')
    .replace('Shift', isMac ? '⇧' : 'Shift')
    .replace('Alt', isMac ? '⌥' : 'Alt')
    .replace('Space', '␣')
    .replace('Escape', 'Esc')
    .replace('←', '◄')
    .replace('→', '►');
}

/**
 * Get all shortcuts formatted for display
 */
export function getFormattedShortcuts(): Array<{ key: string; description: string }> {
  return KEYBOARD_SHORTCUTS.map(shortcut => ({
    key: formatShortcutKey(shortcut.key),
    description: shortcut.description,
  }));
}

// ═══════════════════════════════════════════════════════════════════════════
// MODE CYCLING HELPER
// ═══════════════════════════════════════════════════════════════════════════

const MODE_CYCLE: ReplayMode[] = ['validation', 'test', 'both'];

/**
 * Get next mode in cycle
 */
export function getNextMode(currentMode: ReplayMode): ReplayMode {
  const currentIndex = MODE_CYCLE.indexOf(currentMode);
  const nextIndex = (currentIndex + 1) % MODE_CYCLE.length;
  return MODE_CYCLE[nextIndex];
}

/**
 * Get mode display name
 */
export function getModeDisplayName(mode: ReplayMode): string {
  switch (mode) {
    case 'validation':
      return 'Validación';
    case 'test':
      return 'Test';
    case 'both':
      return 'Ambos';
  }
}
