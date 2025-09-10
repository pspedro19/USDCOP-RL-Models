'use client';

import React, { useState, useEffect } from 'react';
import CustomCursor from '@/components/ui/CustomCursor';
import SplashScreen from '@/components/ui/SplashScreen';
import { ComponentErrorBoundary } from '@/components/common/ErrorBoundary';
import { NotificationProvider, NetworkStatusIndicator } from '@/components/common/ErrorNotifications';

interface ClientLayoutProps {
  children: React.ReactNode;
}

const ClientLayout: React.FC<ClientLayoutProps> = ({ children }) => {
  const [showSplash, setShowSplash] = useState(true);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    
    // Check if we should show splash screen (only on first visit or if forced)
    const hasSeenSplash = localStorage.getItem('dashboard-splash-seen');
    const shouldShowSplash = !hasSeenSplash || sessionStorage.getItem('force-splash');
    
    if (!shouldShowSplash) {
      setShowSplash(false);
    }
  }, []);

  const handleSplashComplete = () => {
    setShowSplash(false);
    localStorage.setItem('dashboard-splash-seen', 'true');
    sessionStorage.removeItem('force-splash');
  };

  if (!isMounted) {
    return <div className="min-h-screen bg-slate-950" />;
  }

  return (
    <NotificationProvider>
      <ComponentErrorBoundary>
        {/* Custom cursor - only on desktop */}
        {typeof window !== 'undefined' && window.innerWidth > 768 && (
          <ComponentErrorBoundary>
            <CustomCursor />
          </ComponentErrorBoundary>
        )}

        {/* Network status indicator */}
        <div className="fixed top-4 left-4 z-40">
          <ComponentErrorBoundary>
            <NetworkStatusIndicator />
          </ComponentErrorBoundary>
        </div>

        {/* Splash screen */}
        {showSplash && (
          <ComponentErrorBoundary>
            <SplashScreen 
              onComplete={handleSplashComplete} 
              duration={3500}
            />
          </ComponentErrorBoundary>
        )}

        {/* Main content */}
        <div className={`transition-opacity duration-500 ${showSplash ? 'opacity-0' : 'opacity-100'}`}>
          <ComponentErrorBoundary showDetails={false}>
            {children}
          </ComponentErrorBoundary>
        </div>
      </ComponentErrorBoundary>
    </NotificationProvider>
  );
};

export default ClientLayout;