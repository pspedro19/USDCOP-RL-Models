/**
 * WebSocket Server for Real-time USD/COP Updates
 * Broadcasts price updates every 5 minutes during market hours
 */

const { WebSocketServer } = require('ws');
const http = require('http');

// Create HTTP server
const server = http.createServer();
const wss = new WebSocketServer({ server });

// Store connected clients
const clients = new Set();

// Market data cache
let latestData = null;
let marketStatus = 'closed';

// Check if market is open (Monday-Friday, 8:00 AM - 12:55 PM Colombia time)
function isMarketOpen() {
  const now = new Date();
  const day = now.getDay();
  const hours = now.getHours();
  const minutes = now.getMinutes();
  const totalMinutes = hours * 60 + minutes;
  
  // Monday = 1, Friday = 5
  // 8:00 AM = 480 minutes, 12:55 PM = 775 minutes
  return day >= 1 && day <= 5 && totalMinutes >= 480 && totalMinutes <= 775;
}

// Broadcast to all connected clients
function broadcast(data) {
  const message = JSON.stringify(data);
  clients.forEach(client => {
    if (client.readyState === 1) { // WebSocket.OPEN
      client.send(message);
    }
  });
}

// Fetch latest data from API
async function fetchLatestData() {
  try {
    const response = await fetch('http://localhost:3010/api/market/realtime?action=fetch');
    if (response.ok) {
      const result = await response.json();
      if (result.data && result.data.length > 0) {
        latestData = result.data[0];
        return latestData;
      }
    }
  } catch (error) {
    console.error('[WebSocket] Error fetching data:', error);
  }
  return null;
}

// Connection handler
wss.on('connection', (ws, req) => {
  console.log('[WebSocket] New client connected');
  clients.add(ws);
  
  // Send initial data
  ws.send(JSON.stringify({
    type: 'connection',
    status: 'connected',
    marketStatus: isMarketOpen() ? 'open' : 'closed',
    timestamp: new Date().toISOString()
  }));
  
  // Send latest data if available
  if (latestData) {
    ws.send(JSON.stringify({
      type: 'price_update',
      data: latestData,
      timestamp: new Date().toISOString()
    }));
  }
  
  // Handle client messages
  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message);
      
      switch (data.type) {
        case 'subscribe':
          console.log('[WebSocket] Client subscribed to updates');
          break;
          
        case 'request_latest':
          const latest = await fetchLatestData();
          if (latest) {
            ws.send(JSON.stringify({
              type: 'price_update',
              data: latest,
              timestamp: new Date().toISOString()
            }));
          }
          break;
          
        case 'ping':
          ws.send(JSON.stringify({ type: 'pong' }));
          break;
      }
    } catch (error) {
      console.error('[WebSocket] Error handling message:', error);
    }
  });
  
  // Handle disconnection
  ws.on('close', () => {
    console.log('[WebSocket] Client disconnected');
    clients.delete(ws);
  });
  
  ws.on('error', (error) => {
    console.error('[WebSocket] Client error:', error);
    clients.delete(ws);
  });
});

// Update loop - runs every minute but only broadcasts at 5-minute intervals
setInterval(async () => {
  const now = new Date();
  const minutes = now.getMinutes();
  
  // Check if market is open
  const marketOpen = isMarketOpen();
  const newStatus = marketOpen ? 'open' : 'closed';
  
  // Broadcast market status change
  if (newStatus !== marketStatus) {
    marketStatus = newStatus;
    broadcast({
      type: 'market_status',
      status: marketStatus,
      timestamp: now.toISOString()
    });
  }
  
  // Only fetch and broadcast at 5-minute intervals during market hours
  if (marketOpen && minutes % 5 === 0) {
    console.log(`[WebSocket] Fetching update at ${now.toLocaleTimeString()}`);
    
    const data = await fetchLatestData();
    if (data) {
      broadcast({
        type: 'price_update',
        data: data,
        timestamp: now.toISOString()
      });
      
      console.log(`[WebSocket] Broadcasted to ${clients.size} clients`);
    }
  }
}, 60000); // Check every minute

// Start server
const PORT = process.env.WS_PORT || 3001;
server.listen(PORT, () => {
  console.log(`[WebSocket] Server running on ws://localhost:${PORT}`);
  console.log(`[WebSocket] Market status: ${isMarketOpen() ? 'OPEN' : 'CLOSED'}`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('[WebSocket] Shutting down...');
  wss.close(() => {
    server.close(() => {
      process.exit(0);
    });
  });
});