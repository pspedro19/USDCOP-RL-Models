'use client';

import React, { useState, useEffect } from 'react';
import { motion, useMotionValue, useSpring } from 'framer-motion';

interface TrailPoint {
  x: number;
  y: number;
  id: number;
}

export const CustomCursor: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const [trails, setTrails] = useState<TrailPoint[]>([]);
  const [trailId, setTrailId] = useState(0);
  
  const cursorX = useMotionValue(0);
  const cursorY = useMotionValue(0);
  
  const springConfig = { damping: 25, stiffness: 400, mass: 0.5 };
  const cursorXSpring = useSpring(cursorX, springConfig);
  const cursorYSpring = useSpring(cursorY, springConfig);

  useEffect(() => {
    let animationFrame: number;
    
    const updateCursor = (e: MouseEvent) => {
      cursorX.set(e.clientX);
      cursorY.set(e.clientY);
      
      // Add trail point
      setTrails(prev => {
        const newTrail = { x: e.clientX, y: e.clientY, id: trailId };
        setTrailId(id => id + 1);
        
        // Keep only last 8 trail points
        const updatedTrails = [newTrail, ...prev].slice(0, 8);
        return updatedTrails;
      });
    };

    const handleMouseEnter = () => setIsVisible(true);
    const handleMouseLeave = () => setIsVisible(false);
    
    const handleMouseOverInteractive = (e: Event) => {
      const target = e.target as HTMLElement;
      if (target.matches('button, a, input, [role="button"], [data-cursor="pointer"]')) {
        setIsHovering(true);
      }
    };
    
    const handleMouseOutInteractive = (e: Event) => {
      const target = e.target as HTMLElement;
      if (target.matches('button, a, input, [role="button"], [data-cursor="pointer"]')) {
        setIsHovering(false);
      }
    };

    document.addEventListener('mousemove', updateCursor);
    document.addEventListener('mouseenter', handleMouseEnter);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseover', handleMouseOverInteractive);
    document.addEventListener('mouseout', handleMouseOutInteractive);

    // Clean up old trails
    const cleanupInterval = setInterval(() => {
      setTrails(prev => prev.slice(0, 6));
    }, 100);

    return () => {
      document.removeEventListener('mousemove', updateCursor);
      document.removeEventListener('mouseenter', handleMouseEnter);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseover', handleMouseOverInteractive);
      document.removeEventListener('mouseout', handleMouseOutInteractive);
      clearInterval(cleanupInterval);
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [cursorX, cursorY, trailId]);

  // Hide default cursor
  useEffect(() => {
    document.body.style.cursor = 'none';
    return () => {
      document.body.style.cursor = 'auto';
    };
  }, []);

  if (!isVisible) return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-[9999]">
      {/* Trail particles */}
      {trails.map((trail, index) => (
        <motion.div
          key={trail.id}
          initial={{ opacity: 0.8, scale: 1 }}
          animate={{ 
            opacity: 0,
            scale: 0.3,
            x: trail.x - 6,
            y: trail.y - 6
          }}
          transition={{
            duration: 0.6,
            delay: index * 0.05,
            ease: "easeOut"
          }}
          className="absolute w-3 h-3 rounded-full pointer-events-none"
          style={{
            background: `radial-gradient(circle, rgba(6, 182, 212, ${0.8 - index * 0.1}) 0%, rgba(59, 130, 246, ${0.6 - index * 0.1}) 50%, transparent 100%)`,
            boxShadow: `0 0 ${8 - index}px rgba(6, 182, 212, ${0.4 - index * 0.05})`
          }}
        />
      ))}
      
      {/* Main cursor */}
      <motion.div
        style={{
          x: cursorXSpring,
          y: cursorYSpring,
        }}
        className="absolute pointer-events-none"
      >
        <motion.div
          animate={{
            scale: isHovering ? 1.5 : 1,
            rotate: isHovering ? 180 : 0,
          }}
          transition={{
            type: "spring",
            stiffness: 300,
            damping: 20
          }}
          className="relative -translate-x-1/2 -translate-y-1/2"
        >
          {/* Outer glow ring */}
          <div className="absolute inset-0 w-8 h-8 rounded-full border-2 border-cyan-400/30 animate-pulse" />
          
          {/* Inner core */}
          <motion.div
            animate={{
              scale: isHovering ? 0.3 : 1,
              opacity: isHovering ? 0.8 : 1
            }}
            className="w-2 h-2 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full mx-auto my-3 shadow-[0_0_12px_rgba(6,182,212,0.8)]"
          />
          
          {/* Crosshair when hovering */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: isHovering ? 1 : 0 }}
            className="absolute inset-0 flex items-center justify-center"
          >
            <div className="w-6 h-[1px] bg-cyan-400/60 absolute" />
            <div className="h-6 w-[1px] bg-cyan-400/60 absolute" />
          </motion.div>
        </motion.div>
      </motion.div>
      
      {/* Magnetic field effect when hovering */}
      {isHovering && (
        <motion.div
          style={{
            x: cursorXSpring,
            y: cursorYSpring,
          }}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 0.3 }}
          exit={{ scale: 0, opacity: 0 }}
          className="absolute pointer-events-none -translate-x-1/2 -translate-y-1/2"
        >
          <div className="w-16 h-16 rounded-full border border-cyan-400/20 animate-ping" />
        </motion.div>
      )}
    </div>
  );
};

export default CustomCursor;