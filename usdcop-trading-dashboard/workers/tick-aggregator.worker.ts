/**
 * Web Worker for Tick Aggregation
 * Efficiently aggregates tick data into candles off the main thread
 */

interface Tick {
  t: number;  // timestamp
  p: number;  // price
  v?: number; // volume (optional)
  s?: string; // symbol (optional)
}

interface Candle {
  t: number;  // timestamp (bucket start)
  o: number;  // open
  h: number;  // high
  l: number;  // low
  c: number;  // close
  v: number;  // volume
  n: number;  // number of ticks
}

interface WorkerMessage {
  type: 'tick' | 'flush' | 'config' | 'reset' | 'export';
  data?: any;
}

interface WorkerResponse {
  type: 'candle' | 'batch' | 'stats' | 'export' | 'error';
  data: any;
}

class TickAggregator {
  private candles = new Map<number, Candle>();
  private currentCandle: Candle | null = null;
  private interval: number = 5 * 60 * 1000; // Default 5 minutes
  private queue: Tick[] = [];
  private flushTimer?: number;
  private stats = {
    ticksProcessed: 0,
    candlesGenerated: 0,
    errors: 0,
    lastProcessTime: 0
  };
  
  // Performance optimizations
  private readonly BATCH_SIZE = 100;
  private readonly BATCH_INTERVAL = 250; // ms
  private batchTimer?: number;
  
  constructor() {
    this.startBatchProcessor();
  }
  
  private startBatchProcessor() {
    // Process queued ticks in batches for better performance
    this.batchTimer = self.setInterval(() => {
      if (this.queue.length > 0) {
        this.processBatch();
      }
    }, this.BATCH_INTERVAL);
  }
  
  configure(config: { interval?: number }) {
    if (config.interval && config.interval > 0) {
      this.interval = config.interval;
      console.log(`[Worker] Interval set to ${this.interval}ms`);
    }
  }
  
  private bucket(timestamp: number): number {
    // Round down to nearest interval
    return Math.floor(timestamp / this.interval) * this.interval;
  }
  
  addTick(tick: Tick) {
    // Validate tick
    if (!this.isValidTick(tick)) {
      this.stats.errors++;
      return;
    }
    
    // Add to queue for batch processing
    this.queue.push(tick);
    
    // Process immediately if queue is large
    if (this.queue.length >= this.BATCH_SIZE * 2) {
      this.processBatch();
    }
  }
  
  private isValidTick(tick: Tick): boolean {
    return (
      typeof tick.t === 'number' && tick.t > 0 &&
      typeof tick.p === 'number' && tick.p > 0 &&
      !isNaN(tick.p) && isFinite(tick.p)
    );
  }
  
  private processBatch() {
    const startTime = performance.now();
    
    // Take up to BATCH_SIZE items from queue
    const batch = this.queue.splice(0, this.BATCH_SIZE);
    if (batch.length === 0) return;
    
    // Sort by timestamp for correct processing
    batch.sort((a, b) => a.t - b.t);
    
    const newCandles: Candle[] = [];
    
    for (const tick of batch) {
      const bucketTime = this.bucket(tick.t);
      
      // Check if we need to start a new candle
      if (!this.currentCandle || this.currentCandle.t !== bucketTime) {
        // Save previous candle if exists
        if (this.currentCandle) {
          newCandles.push(this.currentCandle);
          this.candles.set(this.currentCandle.t, this.currentCandle);
          this.stats.candlesGenerated++;
        }
        
        // Create new candle
        this.currentCandle = {
          t: bucketTime,
          o: tick.p,
          h: tick.p,
          l: tick.p,
          c: tick.p,
          v: tick.v || 0,
          n: 1
        };
      } else {
        // Update existing candle
        this.currentCandle.h = Math.max(this.currentCandle.h, tick.p);
        this.currentCandle.l = Math.min(this.currentCandle.l, tick.p);
        this.currentCandle.c = tick.p;
        this.currentCandle.v += tick.v || 0;
        this.currentCandle.n++;
      }
      
      this.stats.ticksProcessed++;
    }
    
    // Send new candles to main thread
    if (newCandles.length > 0) {
      self.postMessage({
        type: newCandles.length === 1 ? 'candle' : 'batch',
        data: newCandles.length === 1 ? newCandles[0] : newCandles
      } as WorkerResponse);
    }
    
    // Update stats
    this.stats.lastProcessTime = performance.now() - startTime;
    
    // Send stats periodically
    if (this.stats.ticksProcessed % 1000 === 0) {
      this.sendStats();
    }
  }
  
  flush() {
    // Process remaining queue
    while (this.queue.length > 0) {
      this.processBatch();
    }
    
    // Send current candle if exists
    if (this.currentCandle) {
      self.postMessage({
        type: 'candle',
        data: this.currentCandle
      } as WorkerResponse);
      
      this.candles.set(this.currentCandle.t, this.currentCandle);
      this.currentCandle = null;
      this.stats.candlesGenerated++;
    }
    
    this.sendStats();
  }
  
  reset() {
    this.queue = [];
    this.currentCandle = null;
    this.candles.clear();
    this.stats = {
      ticksProcessed: 0,
      candlesGenerated: 0,
      errors: 0,
      lastProcessTime: 0
    };
    
    console.log('[Worker] Reset complete');
  }
  
  exportCandles(): Candle[] {
    // Export all candles sorted by timestamp
    const allCandles = Array.from(this.candles.values());
    
    // Include current candle if exists
    if (this.currentCandle) {
      allCandles.push(this.currentCandle);
    }
    
    return allCandles.sort((a, b) => a.t - b.t);
  }
  
  private sendStats() {
    self.postMessage({
      type: 'stats',
      data: {
        ...this.stats,
        queueSize: this.queue.length,
        candleCount: this.candles.size,
        memoryUsage: this.estimateMemoryUsage()
      }
    } as WorkerResponse);
  }
  
  private estimateMemoryUsage(): number {
    // Rough estimate of memory usage in bytes
    const candleSize = 48; // 6 numbers * 8 bytes
    const tickSize = 24;   // 3 numbers * 8 bytes
    
    return (this.candles.size * candleSize) + (this.queue.length * tickSize);
  }
  
  destroy() {
    if (this.batchTimer) {
      clearInterval(this.batchTimer);
    }
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
    }
    this.reset();
  }
}

// Initialize aggregator
const aggregator = new TickAggregator();

// Handle messages from main thread
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  try {
    const { type, data } = event.data;
    
    switch (type) {
      case 'tick':
        aggregator.addTick(data);
        break;
        
      case 'flush':
        aggregator.flush();
        break;
        
      case 'config':
        aggregator.configure(data);
        break;
        
      case 'reset':
        aggregator.reset();
        break;
        
      case 'export':
        const candles = aggregator.exportCandles();
        self.postMessage({
          type: 'export',
          data: candles
        } as WorkerResponse);
        break;
        
      default:
        console.warn('[Worker] Unknown message type:', type);
    }
  } catch (error) {
    console.error('[Worker] Error processing message:', error);
    self.postMessage({
      type: 'error',
      data: {
        message: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined
      }
    } as WorkerResponse);
  }
};

// Log initialization
console.log('[Worker] Tick aggregator initialized');

// Export for TypeScript
export {};