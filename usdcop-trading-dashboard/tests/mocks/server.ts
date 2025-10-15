import { setupServer } from 'msw/node'
import { handlers } from './handlers'

// Setup the mock service worker server for Node.js testing environment
export const server = setupServer(...handlers)

// Setup and teardown utilities for tests
export function setupMSW() {
  // Start server before all tests
  beforeAll(() => {
    server.listen({
      onUnhandledRequest: 'warn'
    })
  })

  // Reset handlers after each test
  afterEach(() => {
    server.resetHandlers()
  })

  // Clean up after all tests
  afterAll(() => {
    server.close()
  })
}

// Additional server configuration functions
export function mockError(endpoint: string, status: number = 500, message: string = 'Server Error') {
  server.use(
    http.get(endpoint, () => {
      return HttpResponse.json({
        success: false,
        error: {
          code: 'MOCK_ERROR',
          message
        }
      }, { status })
    })
  )
}

export function mockDelay(endpoint: string, delayMs: number = 5000) {
  server.use(
    http.get(endpoint, async () => {
      await delay(delayMs)
      return HttpResponse.json({
        success: true,
        data: { delayed: true, delayMs }
      })
    })
  )
}

export function mockRateLimit(endpoint: string, maxRequests: number = 5) {
  let requestCount = 0

  server.use(
    http.get(endpoint, () => {
      requestCount++

      if (requestCount > maxRequests) {
        return HttpResponse.json({
          success: false,
          error: {
            code: 'RATE_LIMIT_EXCEEDED',
            message: 'Too many requests'
          }
        }, { status: 429 })
      }

      return HttpResponse.json({
        success: true,
        data: { requestCount }
      })
    })
  )
}

export function resetRequestCount() {
  // This would need to be implemented based on specific endpoint tracking
  server.resetHandlers()
}

// WebSocket mock setup for real-time testing
export class MockWebSocketServer {
  private clients: Set<any> = new Set()
  private intervals: Map<string, NodeJS.Timeout> = new Map()

  constructor() {
    this.setupWebSocketMock()
  }

  private setupWebSocketMock() {
    // Mock WebSocket implementation
    global.WebSocket = class MockWebSocket {
      readyState: number = 1
      onopen?: (event: Event) => void
      onmessage?: (event: MessageEvent) => void
      onclose?: (event: CloseEvent) => void
      onerror?: (event: Event) => void

      constructor(public url: string) {
        setTimeout(() => {
          this.onopen?.(new Event('open'))
        }, 100)
      }

      send(data: string) {
        // Echo back for testing
        setTimeout(() => {
          this.onmessage?.(new MessageEvent('message', { data }))
        }, 50)
      }

      close() {
        this.readyState = 3
        setTimeout(() => {
          this.onclose?.(new CloseEvent('close'))
        }, 50)
      }
    } as any
  }

  startPriceUpdates(intervalMs: number = 1000) {
    const interval = setInterval(() => {
      this.broadcast({
        type: 'price_update',
        data: {
          symbol: 'USDCOP',
          price: 4000 + Math.sin(Date.now() / 100000) * 100 + (Math.random() - 0.5) * 10,
          timestamp: Date.now()
        }
      })
    }, intervalMs)

    this.intervals.set('priceUpdates', interval)
  }

  startCandleUpdates(intervalMs: number = 5000) {
    const interval = setInterval(() => {
      const now = Date.now()
      const basePrice = 4000 + Math.sin(now / 100000) * 100

      this.broadcast({
        type: 'new_candle',
        data: {
          datetime: new Date(now).toISOString(),
          timestamp: Math.floor(now / 1000),
          open: basePrice + (Math.random() - 0.5) * 20,
          high: basePrice + Math.random() * 20,
          low: basePrice - Math.random() * 20,
          close: basePrice + (Math.random() - 0.5) * 20,
          volume: Math.floor(1000000 + Math.random() * 500000)
        }
      })
    }, intervalMs)

    this.intervals.set('candleUpdates', interval)
  }

  broadcast(message: any) {
    const messageString = JSON.stringify(message)
    this.clients.forEach(client => {
      if (client.readyState === 1) {
        client.onmessage?.(new MessageEvent('message', { data: messageString }))
      }
    })
  }

  simulateDisconnection() {
    this.clients.forEach(client => {
      client.readyState = 3
      client.onclose?.(new CloseEvent('close', { code: 1006, reason: 'Connection lost' }))
    })
    this.clients.clear()
  }

  simulateReconnection() {
    // This would need to be handled by the client-side reconnection logic
    setTimeout(() => {
      this.broadcast({
        type: 'connection_restored',
        data: { timestamp: Date.now() }
      })
    }, 1000)
  }

  stop() {
    this.intervals.forEach(interval => clearInterval(interval))
    this.intervals.clear()
    this.clients.clear()
  }
}

// Export a singleton instance
export const mockWebSocketServer = new MockWebSocketServer()

// Testing utilities
export function createMockWebSocket(url: string) {
  return new WebSocket(url)
}

export function waitForWebSocketMessage(ws: WebSocket, timeout: number = 5000): Promise<any> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error('WebSocket message timeout'))
    }, timeout)

    ws.onmessage = (event) => {
      clearTimeout(timer)
      try {
        const data = JSON.parse(event.data)
        resolve(data)
      } catch {
        resolve(event.data)
      }
    }

    ws.onerror = (error) => {
      clearTimeout(timer)
      reject(error)
    }
  })
}

export function simulateNetworkConditions(condition: 'slow' | 'unstable' | 'offline') {
  switch (condition) {
    case 'slow':
      // Add delays to all requests
      server.use(
        http.all('*', async ({ request }) => {
          await delay(2000 + Math.random() * 3000)
          return passthrough()
        })
      )
      break

    case 'unstable':
      // Randomly fail requests
      server.use(
        http.all('*', async ({ request }) => {
          if (Math.random() < 0.3) {
            return HttpResponse.json({
              success: false,
              error: { code: 'NETWORK_ERROR', message: 'Connection unstable' }
            }, { status: 503 })
          }
          return passthrough()
        })
      )
      break

    case 'offline':
      // Fail all requests
      server.use(
        http.all('*', () => {
          return HttpResponse.json({
            success: false,
            error: { code: 'OFFLINE', message: 'No network connection' }
          }, { status: 0 })
        })
      )
      break
  }
}