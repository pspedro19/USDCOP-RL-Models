import { useEffect, useState, useCallback, useRef } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';

interface MarketData {
  datetime: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  source: string;
}

interface WebSocketMessage {
  type: 'connection' | 'price_update' | 'market_status' | 'pong';
  data?: MarketData;
  status?: string;
  marketStatus?: string;
  timestamp: string;
}

export function useRealtimeData(initialData: MarketData[] = []) {
  const [data, setData] = useState<MarketData[]>(initialData);
  const [isConnected, setIsConnected] = useState(false);
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed'>('closed');
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;

  // WebSocket URL
  const socketUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001';

  // WebSocket connection
  const { sendMessage, lastMessage, readyState } = useWebSocket(socketUrl, {
    onOpen: () => {
      console.log('[WebSocket] Connected');
      setIsConnected(true);
      reconnectAttempts.current = 0;
      
      // Subscribe to updates
      sendMessage(JSON.stringify({ type: 'subscribe' }));
    },
    onClose: () => {
      console.log('[WebSocket] Disconnected');
      setIsConnected(false);
    },
    onError: (error) => {
      console.error('[WebSocket] Error:', error);
      setIsConnected(false);
    },
    shouldReconnect: (closeEvent) => {
      // Reconnect logic
      if (reconnectAttempts.current < maxReconnectAttempts) {
        reconnectAttempts.current++;
        const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        console.log(`[WebSocket] Reconnecting in ${timeout}ms (attempt ${reconnectAttempts.current})`);
        return true;
      }
      return false;
    },
    reconnectInterval: 3000,
    reconnectAttempts: maxReconnectAttempts
  });

  // Handle incoming messages
  useEffect(() => {
    if (lastMessage !== null) {
      try {
        const message: WebSocketMessage = JSON.parse(lastMessage.data);
        
        switch (message.type) {
          case 'connection':
            console.log('[WebSocket] Connection established:', message);
            if (message.marketStatus) {
              setMarketStatus(message.marketStatus as 'open' | 'closed');
            }
            break;
            
          case 'price_update':
            if (message.data) {
              console.log('[WebSocket] Price update received:', message.data);
              setData(prevData => {
                // Check if this is a new candle or update to existing
                const exists = prevData.some(d => d.datetime === message.data!.datetime);
                if (exists) {
                  // Update existing candle
                  return prevData.map(d => 
                    d.datetime === message.data!.datetime ? message.data! : d
                  );
                } else {
                  // Add new candle
                  return [...prevData, message.data!];
                }
              });
              setLastUpdate(new Date(message.timestamp));
            }
            break;
            
          case 'market_status':
            if (message.status) {
              setMarketStatus(message.status as 'open' | 'closed');
              console.log(`[WebSocket] Market status: ${message.status}`);
            }
            break;
        }
      } catch (error) {
        console.error('[WebSocket] Error parsing message:', error);
      }
    }
  }, [lastMessage]);

  // Connection state
  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Connected',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Closed',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  // Request latest data
  const requestLatest = useCallback(() => {
    if (readyState === ReadyState.OPEN) {
      sendMessage(JSON.stringify({ type: 'request_latest' }));
    }
  }, [readyState, sendMessage]);

  // Send ping to keep connection alive
  useEffect(() => {
    const pingInterval = setInterval(() => {
      if (readyState === ReadyState.OPEN) {
        sendMessage(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Ping every 30 seconds

    return () => clearInterval(pingInterval);
  }, [readyState, sendMessage]);

  return {
    data,
    isConnected,
    marketStatus,
    lastUpdate,
    connectionStatus,
    requestLatest,
    setData // Allow manual data updates
  };
}