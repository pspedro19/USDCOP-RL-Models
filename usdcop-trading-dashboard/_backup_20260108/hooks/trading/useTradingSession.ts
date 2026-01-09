import { useState, useEffect } from 'react';
import { MarketDataService } from '@/lib/services/market-data-service';

export const useTradingSession = () => {
  const [session, setSession] = useState({
    isActive: false,
    coverage: 0,
    hoursRemaining: '--',
    sessionType: 'Market Closed',
    nextEvent: 'Unknown',
    latency: {
      inference: 0,
      e2e: 0
    }
  });

  useEffect(() => {
    const updateSession = async () => {
      try {
        const health = await MarketDataService.checkAPIHealth();
        if (health.market_status) {
          let sessionCoverage = 0;

          if (health.market_status.is_open && health.session_progress) {
            sessionCoverage = health.session_progress.coverage || 0;
          } else if (health.market_status.is_open) {
            const now = new Date();
            const sessionStart = new Date(now);
            sessionStart.setHours(8, 0, 0, 0);
            const minutesElapsed = Math.max(0, (now.getTime() - sessionStart.getTime()) / (1000 * 60));
            const barsCollected = Math.min(60, Math.floor(minutesElapsed / 5));
            sessionCoverage = (barsCollected / 60) * 100;
          }

          setSession(prev => ({
            ...prev,
            isActive: health.market_status.is_open,
            sessionType: health.market_status.is_open ? 'Market Open' : 'Market Closed',
            nextEvent: health.market_status.next_event_type === 'market_open' ? 'Market Open' : 'Market Close',
            hoursRemaining: health.market_status.time_to_next_event || '--',
            coverage: sessionCoverage,
            latency: {
              inference: health.latency?.inference || 0,
              e2e: health.latency?.e2e || 0
            }
          }));
        }
      } catch (error) {
        console.error('Failed to update trading session:', error);
      }
    };

    updateSession();
    const interval = setInterval(updateSession, 60000);

    return () => clearInterval(interval);
  }, []);

  return session;
};
