import { setupWorker } from 'msw/browser'
import { handlers } from './handlers'

// Setup the mock service worker for browser testing environment
export const worker = setupWorker(...handlers)

// Browser-specific MSW setup
export async function startWorker() {
  if (typeof window !== 'undefined') {
    try {
      await worker.start({
        onUnhandledRequest: 'warn',
        serviceWorker: {
          url: '/mockServiceWorker.js'
        }
      })
      console.log('🔶 MSW Worker started successfully')
    } catch (error) {
      console.error('Failed to start MSW worker:', error)
    }
  }
}

export async function stopWorker() {
  if (typeof window !== 'undefined') {
    await worker.stop()
    console.log('🔶 MSW Worker stopped')
  }
}

// Development mode helpers
export function enableMocking() {
  if (process.env.NODE_ENV === 'development') {
    startWorker()
  }
}

export function resetWorker() {
  worker.resetHandlers()
}

// Runtime request interception for specific test scenarios
export function interceptRequests(interceptors: Record<string, any>) {
  const runtimeHandlers = Object.entries(interceptors).map(([url, response]) => {
    return http.get(url, () => {
      return HttpResponse.json(response)
    })
  })

  worker.use(...runtimeHandlers)
}

// Mock WebSocket for browser environment
export class BrowserWebSocketMock {
  private static instance: BrowserWebSocketMock
  private mockServer: any

  private constructor() {
    this.setupBrowserWebSocketMock()
  }

  static getInstance(): BrowserWebSocketMock {
    if (!BrowserWebSocketMock.instance) {
      BrowserWebSocketMock.instance = new BrowserWebSocketMock()
    }
    return BrowserWebSocketMock.instance
  }

  private setupBrowserWebSocketMock() {
    if (typeof window !== 'undefined') {
      const OriginalWebSocket = window.WebSocket

      window.WebSocket = class MockWebSocket {
        readyState: number = 1
        url: string
        onopen?: (event: Event) => void
        onmessage?: (event: MessageEvent) => void
        onclose?: (event: CloseEvent) => void
        onerror?: (event: Event) => void

        constructor(url: string) {
          this.url = url

          // Simulate connection delay
          setTimeout(() => {
            this.readyState = 1
            this.onopen?.(new Event('open'))
          }, 100)
        }

        send(data: string) {
          // Simulate server response
          setTimeout(() => {
            try {
              const parsed = JSON.parse(data)

              // Echo back or generate appropriate response
              let response
              if (parsed.type === 'subscribe') {
                response = {
                  type: 'subscription_confirmed',
                  channel: parsed.channel
                }
              } else {
                response = {
                  type: 'echo',
                  data: parsed
                }
              }

              this.onmessage?.(new MessageEvent('message', {
                data: JSON.stringify(response)
              }))
            } catch {
              // Handle non-JSON data
              this.onmessage?.(new MessageEvent('message', { data }))
            }
          }, 50)
        }

        close(code?: number, reason?: string) {
          this.readyState = 3
          setTimeout(() => {
            this.onclose?.(new CloseEvent('close', { code, reason }))
          }, 50)
        }

        // Mock addEventListener
        addEventListener(type: string, listener: any) {
          if (type === 'open') this.onopen = listener
          if (type === 'message') this.onmessage = listener
          if (type === 'close') this.onclose = listener
          if (type === 'error') this.onerror = listener
        }

        removeEventListener(type: string, listener: any) {
          if (type === 'open' && this.onopen === listener) this.onopen = undefined
          if (type === 'message' && this.onmessage === listener) this.onmessage = undefined
          if (type === 'close' && this.onclose === listener) this.onclose = undefined
          if (type === 'error' && this.onerror === listener) this.onerror = undefined
        }
      } as any

      // Store reference to original for restoration
      ;(window as any).OriginalWebSocket = OriginalWebSocket
    }
  }

  restoreWebSocket() {
    if (typeof window !== 'undefined' && (window as any).OriginalWebSocket) {
      window.WebSocket = (window as any).OriginalWebSocket
    }
  }

  simulateRealtimeData() {
    // This would work with the mocked WebSocket to send periodic updates
    setInterval(() => {
      const event = new CustomEvent('mock-websocket-data', {
        detail: {
          type: 'price_update',
          data: {
            symbol: 'USDCOP',
            price: 4000 + Math.sin(Date.now() / 100000) * 100 + (Math.random() - 0.5) * 10,
            timestamp: Date.now()
          }
        }
      })
      window.dispatchEvent(event)
    }, 1000)
  }
}

// Storage mocks for browser testing
export function mockLocalStorage() {
  const store: Record<string, string> = {}

  Object.defineProperty(window, 'localStorage', {
    value: {
      getItem: (key: string) => store[key] || null,
      setItem: (key: string, value: string) => { store[key] = value },
      removeItem: (key: string) => { delete store[key] },
      clear: () => { Object.keys(store).forEach(key => delete store[key]) },
      length: Object.keys(store).length,
      key: (index: number) => Object.keys(store)[index] || null
    },
    writable: true
  })
}

export function mockSessionStorage() {
  const store: Record<string, string> = {}

  Object.defineProperty(window, 'sessionStorage', {
    value: {
      getItem: (key: string) => store[key] || null,
      setItem: (key: string, value: string) => { store[key] = value },
      removeItem: (key: string) => { delete store[key] },
      clear: () => { Object.keys(store).forEach(key => delete store[key]) },
      length: Object.keys(store).length,
      key: (index: number) => Object.keys(store)[index] || null
    },
    writable: true
  })
}

// Canvas mock for chart testing in browser
export function mockCanvas() {
  const canvasProto = HTMLCanvasElement.prototype
  const getContext = canvasProto.getContext

  canvasProto.getContext = function(contextType: string) {
    if (contextType === '2d') {
      return {
        fillRect: () => {},
        clearRect: () => {},
        getImageData: () => ({ data: new Array(4) }),
        putImageData: () => {},
        createImageData: () => [],
        setTransform: () => {},
        drawImage: () => {},
        save: () => {},
        fillText: () => {},
        restore: () => {},
        beginPath: () => {},
        moveTo: () => {},
        lineTo: () => {},
        closePath: () => {},
        stroke: () => {},
        translate: () => {},
        scale: () => {},
        rotate: () => {},
        arc: () => {},
        fill: () => {},
        measureText: () => ({ width: 0 }),
        transform: () => {},
        rect: () => {},
        clip: () => {},
        canvas: this
      }
    }
    return getContext.call(this, contextType)
  }
}

// Performance monitoring mock
export function mockPerformanceObserver() {
  if (typeof window !== 'undefined') {
    window.PerformanceObserver = class MockPerformanceObserver {
      constructor(private callback: any) {}
      observe() {}
      disconnect() {}
    } as any
  }
}

// Notification API mock
export function mockNotification() {
  if (typeof window !== 'undefined') {
    window.Notification = class MockNotification {
      static permission = 'granted'
      static requestPermission = () => Promise.resolve('granted')

      constructor(public title: string, public options?: any) {}
      close() {}
    } as any
  }
}

// Export browser setup function
export function setupBrowserMocks() {
  mockLocalStorage()
  mockSessionStorage()
  mockCanvas()
  mockPerformanceObserver()
  mockNotification()
  BrowserWebSocketMock.getInstance()
}