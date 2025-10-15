"use client";

import { useGesture } from "@use-gesture/react";
import { useSpring, config } from "framer-motion";
import { useCallback, useRef, useState } from "react";
import { toast } from "react-hot-toast";

interface TouchGestureState {
  scale: number;
  x: number;
  y: number;
  rotation: number;
}

interface UseTouchGesturesProps {
  onZoom?: (scale: number, center: { x: number; y: number }) => void;
  onPan?: (delta: { x: number; y: number }) => void;
  onLongPress?: (position: { x: number; y: number }) => void;
  onDoubleClick?: (position: { x: number; y: number }) => void;
  onThreeFingerTap?: (position: { x: number; y: number }) => void;
  onTwoFingerRotate?: (angle: number) => void;
  minScale?: number;
  maxScale?: number;
  enabled?: boolean;
  containerRef?: React.RefObject<HTMLElement>;
}

export const useTouchGestures = ({
  onZoom,
  onPan,
  onLongPress,
  onDoubleClick,
  onThreeFingerTap,
  onTwoFingerRotate,
  minScale = 0.5,
  maxScale = 10,
  enabled = true,
  containerRef,
}: UseTouchGesturesProps = {}) => {
  const [gestureState, setGestureState] = useState<TouchGestureState>({
    scale: 1,
    x: 0,
    y: 0,
    rotation: 0,
  });

  const longPressTimeoutRef = useRef<NodeJS.Timeout>();
  const lastTapTimeRef = useRef<number>(0);
  const tapCountRef = useRef<number>(0);
  const isLongPressingRef = useRef<boolean>(false);
  const momentumRef = useRef<{ vx: number; vy: number }>({ vx: 0, vy: 0 });

  // Spring animations for smooth interactions
  const scaleSpring = useSpring(1, { stiffness: 400, damping: 40 });
  const xSpring = useSpring(0, { stiffness: 300, damping: 30 });
  const ySpring = useSpring(0, { stiffness: 300, damping: 30 });

  // Clear long press timeout
  const clearLongPressTimeout = useCallback(() => {
    if (longPressTimeoutRef.current) {
      clearTimeout(longPressTimeoutRef.current);
      longPressTimeoutRef.current = undefined;
    }
  }, []);

  // Handle long press
  const startLongPress = useCallback((position: { x: number; y: number }) => {
    if (!enabled) return;

    clearLongPressTimeout();
    isLongPressingRef.current = false;

    longPressTimeoutRef.current = setTimeout(() => {
      if (!isLongPressingRef.current) {
        isLongPressingRef.current = true;
        onLongPress?.(position);

        // Haptic feedback on supported devices
        if ("vibrate" in navigator) {
          navigator.vibrate(50);
        }

        toast.success("Long press detected", {
          duration: 1000,
          position: "bottom-center",
        });
      }
    }, 500);
  }, [enabled, onLongPress, clearLongPressTimeout]);

  // Handle tap detection for double/triple taps
  const handleTap = useCallback((position: { x: number; y: number }, fingerCount: number) => {
    if (!enabled) return;

    const now = Date.now();
    const timeDiff = now - lastTapTimeRef.current;

    if (timeDiff < 300) {
      tapCountRef.current += 1;
    } else {
      tapCountRef.current = 1;
    }

    lastTapTimeRef.current = now;

    // Handle different tap types based on finger count and tap count
    if (fingerCount === 3 && tapCountRef.current === 1) {
      onThreeFingerTap?.(position);
      toast.success("Three finger tap", {
        duration: 1000,
        position: "bottom-center",
      });
    } else if (fingerCount === 1 && tapCountRef.current === 2) {
      onDoubleClick?.(position);
      toast.success("Double tap", {
        duration: 1000,
        position: "bottom-center",
      });
    }
  }, [enabled, onDoubleClick, onThreeFingerTap]);

  // Apply momentum after pan ends
  const applyMomentum = useCallback(() => {
    if (!enabled) return;

    const { vx, vy } = momentumRef.current;
    const threshold = 0.1;

    if (Math.abs(vx) > threshold || Math.abs(vy) > threshold) {
      const decay = 0.95;
      const step = () => {
        momentumRef.current.vx *= decay;
        momentumRef.current.vy *= decay;

        if (Math.abs(momentumRef.current.vx) > 0.01 || Math.abs(momentumRef.current.vy) > 0.01) {
          onPan?.({ x: momentumRef.current.vx, y: momentumRef.current.vy });
          requestAnimationFrame(step);
        }
      };
      requestAnimationFrame(step);
    }
  }, [enabled, onPan]);

  // Gesture handlers using @use-gesture/react
  const gestures = useGesture(
    {
      onDrag: ({ offset: [x, y], velocity: [vx, vy], active, event, touches }) => {
        if (!enabled || touches > 1) return;

        event?.preventDefault?.();
        clearLongPressTimeout();
        isLongPressingRef.current = false;

        // Update momentum for physics-based scrolling
        momentumRef.current = { vx: vx * 10, vy: vy * 10 };

        setGestureState(prev => ({ ...prev, x, y }));
        xSpring.set(x);
        ySpring.set(y);

        onPan?.({ x: vx * 10, y: vy * 10 });

        // Apply momentum when drag ends
        if (!active) {
          applyMomentum();
        }
      },

      onPinch: ({ offset: [scale], origin: [ox, oy], active, event }) => {
        if (!enabled) return;

        event?.preventDefault?.();
        clearLongPressTimeout();

        // Clamp scale within bounds
        const clampedScale = Math.min(Math.max(scale, minScale), maxScale);

        setGestureState(prev => ({ ...prev, scale: clampedScale }));
        scaleSpring.set(clampedScale);

        // Get center point relative to container
        const containerRect = containerRef?.current?.getBoundingClientRect();
        const center = containerRect ? {
          x: ox - containerRect.left,
          y: oy - containerRect.top,
        } : { x: ox, y: oy };

        onZoom?.(clampedScale, center);

        // Smooth spring animation when pinch ends
        if (!active) {
          scaleSpring.set(clampedScale, { type: "spring", ...config.gentle });
        }
      },

      onWheel: ({ delta: [, dy], event }) => {
        if (!enabled) return;

        event?.preventDefault?.();

        // Convert wheel delta to scale change
        const scaleFactor = 1 - dy * 0.001;
        const newScale = Math.min(Math.max(gestureState.scale * scaleFactor, minScale), maxScale);

        setGestureState(prev => ({ ...prev, scale: newScale }));
        scaleSpring.set(newScale);

        // Get mouse position for zoom center
        const rect = (event.target as HTMLElement)?.getBoundingClientRect();
        const center = rect ? {
          x: (event as WheelEvent).clientX - rect.left,
          y: (event as WheelEvent).clientY - rect.top,
        } : { x: 0, y: 0 };

        onZoom?.(newScale, center);
      },

      onPointerDown: ({ event, touches }) => {
        if (!enabled) return;

        const rect = (event.target as HTMLElement)?.getBoundingClientRect();
        const position = rect ? {
          x: (event as PointerEvent).clientX - rect.left,
          y: (event as PointerEvent).clientY - rect.top,
        } : { x: 0, y: 0 };

        // Start long press detection for single finger
        if (touches === 1) {
          startLongPress(position);
        }
      },

      onPointerUp: ({ event, touches }) => {
        if (!enabled) return;

        clearLongPressTimeout();

        // Handle tap detection if not long pressing
        if (!isLongPressingRef.current) {
          const rect = (event.target as HTMLElement)?.getBoundingClientRect();
          const position = rect ? {
            x: (event as PointerEvent).clientX - rect.left,
            y: (event as PointerEvent).clientY - rect.top,
          } : { x: 0, y: 0 };

          handleTap(position, touches);
        }

        isLongPressingRef.current = false;
      },

      // Handle rotation gesture (two fingers)
      onMove: ({ touches, event }) => {
        if (!enabled || touches !== 2) return;

        const pointerEvent = event as PointerEvent;
        if (pointerEvent.type === "pointermove") {
          // Calculate rotation between two touch points
          const touches_list = (event.target as HTMLElement).getPointerCapture ? [] : [];
          if (touches_list.length === 2) {
            const angle = Math.atan2(
              touches_list[1].clientY - touches_list[0].clientY,
              touches_list[1].clientX - touches_list[0].clientX
            );
            const degrees = (angle * 180) / Math.PI;

            setGestureState(prev => ({ ...prev, rotation: degrees }));
            onTwoFingerRotate?.(degrees);
          }
        }
      },
    },
    {
      target: containerRef?.current,
      eventOptions: { passive: false },
      drag: {
        filterTaps: true,
        rubberband: true,
      },
      pinch: {
        scaleBounds: { min: minScale, max: maxScale },
        rubberband: true,
      },
    }
  );

  // Reset gestures to initial state
  const resetGestures = useCallback(() => {
    const initialState = { scale: 1, x: 0, y: 0, rotation: 0 };
    setGestureState(initialState);
    scaleSpring.set(1);
    xSpring.set(0);
    ySpring.set(0);
    clearLongPressTimeout();
    isLongPressingRef.current = false;
  }, [scaleSpring, xSpring, ySpring, clearLongPressTimeout]);

  // Get current spring values for smooth animations
  const getSpringValues = useCallback(() => ({
    scale: scaleSpring.get(),
    x: xSpring.get(),
    y: ySpring.get(),
  }), [scaleSpring, xSpring, ySpring]);

  return {
    gestures,
    gestureState,
    resetGestures,
    getSpringValues,
    springs: {
      scale: scaleSpring,
      x: xSpring,
      y: ySpring,
    },
  };
};