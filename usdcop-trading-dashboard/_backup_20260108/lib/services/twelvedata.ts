/**
 * TwelveData API Integration
 * Real API calls to TwelveData forex endpoint
 * Documentation: https://twelvedata.com/docs
 */

export interface TwelveDataResponse {
  symbol: string;
  price: number;
  timestamp: string;
  open?: number;
  high?: number;
  low?: number;
  close?: number;
  volume?: number;
}

export interface PriceData {
  symbol: string;
  price: number;
  timestamp: string;
  volume?: number;
  bid?: number;
  ask?: number;
}

interface TwelveDataTimeSeriesResponse {
  meta: {
    symbol: string;
    interval: string;
    currency_base: string;
    currency_quote: string;
    type: string;
  };
  values: Array<{
    datetime: string;
    open: string;
    high: string;
    low: string;
    close: string;
    volume?: string;
  }>;
  status: string;
}

interface TwelveDataQuoteResponse {
  symbol: string;
  name: string;
  exchange: string;
  datetime: string;
  timestamp: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume?: string;
  previous_close: string;
  change: string;
  percent_change: string;
  average_volume?: string;
  is_market_open: boolean;
}

// API Configuration
const TWELVEDATA_BASE_URL = 'https://api.twelvedata.com';
const API_KEYS = [
  process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY_1,
  process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY_2,
  process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY_3,
  process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY_4,
  process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY_5,
  process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY_6,
  process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY_7,
  process.env.NEXT_PUBLIC_TWELVEDATA_API_KEY_8,
].filter(Boolean);

let currentKeyIndex = 0;

/**
 * Get next available API key (round-robin)
 */
function getApiKey(): string | undefined {
  if (API_KEYS.length === 0) {
    console.warn('[TwelveData] No API keys configured. Set NEXT_PUBLIC_TWELVEDATA_API_KEY_1, etc.');
    return undefined;
  }

  const key = API_KEYS[currentKeyIndex];
  currentKeyIndex = (currentKeyIndex + 1) % API_KEYS.length;
  return key;
}

/**
 * Make API request with error handling and rate limiting
 */
async function makeApiRequest<T>(endpoint: string, params: Record<string, string>): Promise<T> {
  const apiKey = getApiKey();

  if (!apiKey) {
    throw new Error('TwelveData API key not configured');
  }

  const url = new URL(endpoint, TWELVEDATA_BASE_URL);
  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.append(key, value);
  });
  url.searchParams.append('apikey', apiKey);

  console.log(`[TwelveData] Requesting: ${endpoint} with params:`, params);

  try {
    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[TwelveData] API error (${response.status}):`, errorText);

      if (response.status === 429) {
        throw new Error('Rate limit exceeded. Please try again later.');
      }

      throw new Error(`TwelveData API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    // Check for API error responses
    if (data.status === 'error') {
      console.error('[TwelveData] API returned error:', data.message);
      throw new Error(data.message || 'TwelveData API error');
    }

    return data as T;
  } catch (error) {
    console.error('[TwelveData] Request failed:', error);
    throw error;
  }
}

/**
 * Fetch real-time quote for a symbol
 */
export async function fetchRealTimeQuote(symbol: string = 'USD/COP'): Promise<TwelveDataResponse> {
  try {
    const data = await makeApiRequest<TwelveDataQuoteResponse>('/quote', {
      symbol,
      interval: '5min',
    });

    return {
      symbol: data.symbol,
      price: parseFloat(data.close),
      timestamp: data.datetime,
      open: parseFloat(data.open),
      high: parseFloat(data.high),
      low: parseFloat(data.low),
      close: parseFloat(data.close),
      volume: data.volume ? parseFloat(data.volume) : undefined,
    };
  } catch (error) {
    console.error('[TwelveData] fetchRealTimeQuote failed:', error);
    throw error;
  }
}

/**
 * Fetch time series data (OHLC bars)
 */
export async function fetchTimeSeries(
  symbol: string = 'USD/COP',
  interval: string = '5min',
  outputsize: number = 100
): Promise<TwelveDataResponse[]> {
  try {
    const data = await makeApiRequest<TwelveDataTimeSeriesResponse>('/time_series', {
      symbol,
      interval,
      outputsize: outputsize.toString(),
      format: 'JSON',
    });

    if (!data.values || data.values.length === 0) {
      console.warn('[TwelveData] No time series data returned');
      return [];
    }

    return data.values.map((bar) => ({
      symbol: data.meta.symbol,
      timestamp: bar.datetime,
      price: parseFloat(bar.close),
      open: parseFloat(bar.open),
      high: parseFloat(bar.high),
      low: parseFloat(bar.low),
      close: parseFloat(bar.close),
      volume: bar.volume ? parseFloat(bar.volume) : undefined,
    }));
  } catch (error) {
    console.error('[TwelveData] fetchTimeSeries failed:', error);
    throw error;
  }
}

/**
 * Fetch technical indicators
 */
export async function fetchTechnicalIndicators(symbol: string = 'USD/COP', interval: string = '5min') {
  try {
    // Fetch multiple indicators in parallel
    const [rsi, macd, sma, ema, bbands, stoch] = await Promise.allSettled([
      makeApiRequest('/rsi', { symbol, interval, time_period: '14' }),
      makeApiRequest('/macd', { symbol, interval }),
      makeApiRequest('/sma', { symbol, interval, time_period: '20' }),
      makeApiRequest('/ema', { symbol, interval, time_period: '20' }),
      makeApiRequest('/bbands', { symbol, interval, time_period: '20' }),
      makeApiRequest('/stoch', { symbol, interval }),
    ]);

    return {
      rsi: rsi.status === 'fulfilled' ? rsi.value : null,
      macd: macd.status === 'fulfilled' ? macd.value : null,
      sma: sma.status === 'fulfilled' ? sma.value : null,
      ema: ema.status === 'fulfilled' ? ema.value : null,
      bbands: bbands.status === 'fulfilled' ? bbands.value : null,
      stoch: stoch.status === 'fulfilled' ? stoch.value : null,
    };
  } catch (error) {
    console.error('[TwelveData] fetchTechnicalIndicators failed:', error);
    throw error;
  }
}

/**
 * Legacy function for backward compatibility
 */
export async function fetchTwelveData(symbol: string = 'USD/COP'): Promise<TwelveDataResponse> {
  return fetchRealTimeQuote(symbol);
}

/**
 * WebSocket client for real-time data
 * Note: TwelveData WebSocket requires a paid plan
 * This implementation uses HTTP polling as fallback
 */
export const wsClient = {
  connect: (symbol: string, callback: (data: PriceData) => void, interval: number = 5000) => {
    console.log(`[TwelveData] Starting WebSocket fallback (HTTP polling) for ${symbol}`);

    let isActive = true;

    const poll = async () => {
      if (!isActive) return;

      try {
        const quote = await fetchRealTimeQuote(symbol);
        callback({
          symbol: quote.symbol,
          price: quote.price,
          timestamp: quote.timestamp,
          volume: quote.volume,
        });
      } catch (error) {
        console.error('[TwelveData] Polling error:', error);
        // Continue polling even on error (with exponential backoff handled by caller)
      }

      if (isActive) {
        setTimeout(poll, interval);
      }
    };

    // Start polling
    poll();

    // Return disconnect function
    return () => {
      console.log(`[TwelveData] Stopping WebSocket fallback for ${symbol}`);
      isActive = false;
    };
  },
};

/**
 * Check if API is configured and working
 */
export async function testConnection(): Promise<boolean> {
  try {
    await fetchRealTimeQuote('USD/COP');
    console.log('[TwelveData] Connection test successful');
    return true;
  } catch (error) {
    console.error('[TwelveData] Connection test failed:', error);
    return false;
  }
}

export default {
  fetchTwelveData,
  fetchRealTimeQuote,
  fetchTimeSeries,
  fetchTechnicalIndicators,
  wsClient,
  testConnection,
};
