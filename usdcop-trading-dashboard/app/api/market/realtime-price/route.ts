/**
 * Real-time USD/COP Price API
 * ===========================
 * Hybrid endpoint that provides the most up-to-date USD/COP price:
 *
 * - During market hours (8am-12:55pm COT, Mon-Fri): Uses TwelveData API
 * - Outside market hours: Scrapes investing.com for real-time price
 *
 * GET /api/market/realtime-price
 *
 * Returns:
 * {
 *   price: number,
 *   change: number,
 *   changePct: number,
 *   source: 'twelvedata' | 'investing' | 'cache' | 'fallback',
 *   lastUpdate: string (ISO),
 *   isMarketOpen: boolean,
 *   marketStatus: 'open' | 'closed' | 'pre_market' | 'after_hours',
 *   nextUpdate: string (ISO)
 * }
 */

import { NextRequest, NextResponse } from 'next/server';

// Cache for price data
interface PriceCache {
  price: number;
  previousClose: number;
  change: number;
  changePct: number;
  dayHigh: number | null;
  dayLow: number | null;
  week52High: number | null;
  week52Low: number | null;
  source: 'twelvedata' | 'investing' | 'cache' | 'fallback';
  lastUpdate: Date;
}

let priceCache: PriceCache | null = null;
const CACHE_TTL_MARKET_OPEN = 5 * 60 * 1000;    // 5 minutes during market hours
const CACHE_TTL_MARKET_CLOSED = 30 * 60 * 1000; // 30 minutes outside market hours

// Colombia timezone offset (COT = UTC-5)
const COT_OFFSET = -5;

/**
 * Check if market is currently open
 * Market hours: 8:00 AM - 12:55 PM COT, Monday-Friday
 */
function getMarketStatus(): {
  isOpen: boolean;
  status: 'open' | 'closed' | 'pre_market' | 'after_hours';
  nextOpen: Date | null;
  minutesToNextUpdate: number;
} {
  const now = new Date();

  // Convert to COT
  const cotDate = new Date(now.getTime() + (COT_OFFSET * 60 * 60 * 1000));
  const cotHour = cotDate.getUTCHours();
  const cotMinute = cotDate.getUTCMinutes();
  const dayOfWeek = cotDate.getUTCDay(); // 0=Sun, 1=Mon, ..., 6=Sat

  // Weekend check
  if (dayOfWeek === 0 || dayOfWeek === 6) {
    return {
      isOpen: false,
      status: 'closed',
      nextOpen: getNextMarketOpen(cotDate),
      minutesToNextUpdate: 30
    };
  }

  // Market hours: 8:00 AM - 12:55 PM COT
  const timeInMinutes = cotHour * 60 + cotMinute;
  const marketOpen = 8 * 60;        // 8:00 AM = 480 minutes
  const marketClose = 12 * 60 + 55; // 12:55 PM = 775 minutes

  if (timeInMinutes >= marketOpen && timeInMinutes < marketClose) {
    return {
      isOpen: true,
      status: 'open',
      nextOpen: null,
      minutesToNextUpdate: 5
    };
  } else if (timeInMinutes < marketOpen) {
    return {
      isOpen: false,
      status: 'pre_market',
      nextOpen: getNextMarketOpen(cotDate),
      minutesToNextUpdate: 15
    };
  } else {
    return {
      isOpen: false,
      status: 'after_hours',
      nextOpen: getNextMarketOpen(cotDate),
      minutesToNextUpdate: 30
    };
  }
}

/**
 * Get next market open time
 */
function getNextMarketOpen(cotDate: Date): Date {
  const next = new Date(cotDate);
  next.setUTCHours(8, 0, 0, 0);

  // If today's market already passed, go to next day
  if (cotDate.getUTCHours() >= 13 || (cotDate.getUTCHours() === 12 && cotDate.getUTCMinutes() >= 55)) {
    next.setUTCDate(next.getUTCDate() + 1);
  }

  // Skip weekends
  while (next.getUTCDay() === 0 || next.getUTCDay() === 6) {
    next.setUTCDate(next.getUTCDate() + 1);
  }

  // Convert back to UTC
  return new Date(next.getTime() - (COT_OFFSET * 60 * 60 * 1000));
}

/**
 * Fetch price from TwelveData API (via our proxy)
 */
async function fetchFromTwelveData(): Promise<PriceCache | null> {
  try {
    const baseUrl = process.env.TRADING_API_URL || 'http://usdcop-trading-api:8000';
    const response = await fetch(`${baseUrl}/api/candlesticks/USDCOP?limit=2&timeframe=5m`, {
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(10000)
    });

    if (!response.ok) {
      console.error(`[RealtimePrice] TwelveData API error: ${response.status}`);
      return null;
    }

    const data = await response.json();

    if (data.data && data.data.length > 0) {
      const latest = data.data[data.data.length - 1];
      const previous = data.data.length > 1 ? data.data[data.data.length - 2] : latest;

      const price = latest.close;
      const previousClose = previous.close;
      const change = price - previousClose;
      const changePct = (change / previousClose) * 100;

      return {
        price,
        previousClose,
        change,
        changePct,
        dayHigh: null,
        dayLow: null,
        week52High: null,
        week52Low: null,
        source: 'twelvedata',
        lastUpdate: new Date(latest.time)
      };
    }

    return null;
  } catch (error) {
    console.error('[RealtimePrice] TwelveData fetch error:', error);
    return null;
  }
}

/**
 * Scrape real-time price from Investing.com
 * URL: https://www.investing.com/currencies/usd-cop
 *
 * Extracts:
 * - Current price
 * - Change (absolute and percentage)
 * - Day range (high/low)
 * - 52-week range (high/low)
 */
async function fetchFromInvesting(): Promise<PriceCache | null> {
  try {
    const url = 'https://www.investing.com/currencies/usd-cop';

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'es-CO,es;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
      },
      signal: AbortSignal.timeout(15000)
    });

    if (!response.ok) {
      console.error(`[RealtimePrice] Investing.com error: ${response.status}`);
      return null;
    }

    const html = await response.text();

    // Helper to parse numbers in Spanish/European format (3.715,26 or 3,715.26)
    const parseNumber = (str: string): number | null => {
      if (!str) return null;
      // Remove spaces
      str = str.trim();
      // Check if it's European format (comma as decimal)
      if (str.match(/^\d{1,3}(\.\d{3})*,\d+$/)) {
        // European: 3.715,26 -> 3715.26
        str = str.replace(/\./g, '').replace(',', '.');
      } else {
        // US format: remove commas
        str = str.replace(/,/g, '');
      }
      const num = parseFloat(str);
      return isNaN(num) ? null : num;
    };

    // Extract USD/COP specific data first using the main JSON block
    // Pattern: "last":3715.26,"changePcr":0.14,"change":5.26,...,"high":3715.26,"low":3708.59
    let price: number | null = null;
    let change: number | null = null;
    let changePct: number | null = null;

    // Try to find USD/COP specific JSON block with long_name
    const usdcopJsonMatch = html.match(/"last":\s*([0-9.]+),\s*"changePcr":\s*([0-9.-]+),\s*"change":\s*([0-9.-]+)[^}]*"long_name":\s*"US Dollar Colombian Peso"/);
    if (usdcopJsonMatch) {
      price = parseFloat(usdcopJsonMatch[1]);
      changePct = parseFloat(usdcopJsonMatch[2]);
      change = parseFloat(usdcopJsonMatch[3]);
      console.log(`[RealtimePrice] Found USD/COP JSON: price=${price}, change=${change}, changePct=${changePct}`);
    }

    // Fallback: Extract price from other patterns
    if (!price) {
      const pricePatterns = [
        /data-test="instrument-price-last"[^>]*>([0-9.,]+)</,
        /instrument-price-last"[^>]*>([0-9.,]+)</,
        /class="[^"]*instrument-price_last[^"]*"[^>]*>([0-9.,]+)</,
        /pid-\d+-last[^>]*>([0-9.,]+)</,
      ];

      for (const pattern of pricePatterns) {
        const match = html.match(pattern);
        if (match) {
          price = parseNumber(match[1]);
          if (price && price > 3000 && price < 6000) break;
          price = null;
        }
      }
    }

    // Fallback: Extract change from other patterns
    if (change === null) {
      const changePatterns = [
        /data-test="instrument-price-change"[^>]*>([+-]?[0-9.,]+)</,
        /pid-\d+-pc[^>]*>([+-]?[0-9.,]+)</,
      ];

      for (const pattern of changePatterns) {
        const match = html.match(pattern);
        if (match) {
          change = parseNumber(match[1]);
          if (change !== null && Math.abs(change) < 500) break;
          change = null;
        }
      }
    }

    // Fallback: Extract change percentage from other patterns
    if (changePct === null) {
      const changePctPatterns = [
        /data-test="instrument-price-change-percent"[^>]*>\(?([+-]?[0-9.,]+)%?\)?</i,
        /"changePcr":\s*([0-9.-]+)/,
        /pid-\d+-pcp[^>]*>\(?([+-]?[0-9.,]+)%?\)?</,
        /\(([+-]?[0-9.,]+)\s*%\)/,
      ];

      for (const pattern of changePctPatterns) {
        const match = html.match(pattern);
        if (match) {
          changePct = parseNumber(match[1]);
          if (changePct !== null && Math.abs(changePct) < 20) break;
          changePct = null;
        }
      }
    }

    // If we have change but no changePct, calculate it
    if (changePct === null && change !== null && price !== null) {
      const prevClose = price - change;
      if (prevClose > 0) {
        changePct = (change / prevClose) * 100;
      }
    }

    // Extract day range from FAQ structured data
    // Pattern: "Today's USD/COP range is from 3,708.59 to 3,715.26"
    let dayHigh: number | null = null;
    let dayLow: number | null = null;

    const dayRangePatterns = [
      /(?:Today'?s?\s+)?(?:USD\/COP\s+)?range\s+is\s+(?:from\s+)?([0-9,\.]+)\s+to\s+([0-9,\.]+)/i,
      /(?:Rango d[ií]a|Day['']?s Range)[^\d]*([0-9.,]+)[^\d]+([0-9.,]+)/i,
      /data-test="[^"]*day[^"]*range[^"]*"[^>]*>([0-9.,]+)[^\d]+([0-9.,]+)/i,
    ];

    for (const pattern of dayRangePatterns) {
      const match = html.match(pattern);
      if (match) {
        const val1 = parseNumber(match[1]);
        const val2 = parseNumber(match[2]);
        if (val1 && val2 && val1 > 3000 && val2 > 3000) {
          dayLow = Math.min(val1, val2);
          dayHigh = Math.max(val1, val2);
          break;
        }
      }
    }

    // Extract 52-week range from FAQ structured data
    // Pattern: "The 52-week range for USD/COP is 3,682.93 to 4,463.50"
    let week52High: number | null = null;
    let week52Low: number | null = null;

    const week52Patterns = [
      /52-week\s+range\s+(?:for\s+)?(?:USD\/COP\s+)?is\s+([0-9,\.]+)\s+to\s+([0-9,\.]+)/i,
      /(?:52 semanas|52[- ]?[Ww]eek|52wk)[^\d]*([0-9.,]+)[^\d]+([0-9.,]+)/i,
      /data-test="[^"]*52[^"]*week[^"]*"[^>]*>([0-9.,]+)[^\d]+([0-9.,]+)/i,
    ];

    for (const pattern of week52Patterns) {
      const match = html.match(pattern);
      if (match) {
        const val1 = parseNumber(match[1]);
        const val2 = parseNumber(match[2]);
        if (val1 && val2 && val1 > 3000 && val2 > 3000) {
          week52Low = Math.min(val1, val2);
          week52High = Math.max(val1, val2);
          break;
        }
      }
    }

    // If we have price, return the data
    if (price) {
      // Calculate previousClose from change if available
      let previousClose = price;
      if (change !== null) {
        previousClose = price - change;
      } else if (changePct !== null) {
        previousClose = price / (1 + changePct / 100);
        change = price - previousClose;
      }

      console.log(`[RealtimePrice] Investing.com data: price=${price}, change=${change}, changePct=${changePct}, dayRange=${dayLow}-${dayHigh}, 52wk=${week52Low}-${week52High}`);

      return {
        price,
        previousClose,
        change: change ?? 0,
        changePct: changePct ?? 0,
        dayHigh,
        dayLow,
        week52High,
        week52Low,
        source: 'investing',
        lastUpdate: new Date()
      };
    }

    console.warn('[RealtimePrice] Could not extract price from Investing.com HTML');
    return null;
  } catch (error) {
    console.error('[RealtimePrice] Investing.com fetch error:', error);
    return null;
  }
}

/**
 * Try alternative source: DolarHoy Colombia API
 */
async function fetchFromDolarHoy(): Promise<PriceCache | null> {
  try {
    // This is a backup source
    const response = await fetch('https://api.exchangerate-api.com/v4/latest/USD', {
      signal: AbortSignal.timeout(10000)
    });

    if (!response.ok) return null;

    const data = await response.json();
    const price = data.rates?.COP;

    if (price && price > 3000 && price < 6000) {
      return {
        price,
        previousClose: price,
        change: 0,
        changePct: 0,
        dayHigh: null,
        dayLow: null,
        week52High: null,
        week52Low: null,
        source: 'fallback',
        lastUpdate: new Date()
      };
    }

    return null;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    const marketStatus = getMarketStatus();
    const cacheTtl = marketStatus.isOpen ? CACHE_TTL_MARKET_OPEN : CACHE_TTL_MARKET_CLOSED;

    // Check cache validity
    if (priceCache && (Date.now() - priceCache.lastUpdate.getTime()) < cacheTtl) {
      return NextResponse.json({
        success: true,
        data: {
          price: priceCache.price,
          previousClose: priceCache.previousClose,
          change: priceCache.change,
          changePct: priceCache.changePct,
          dayHigh: priceCache.dayHigh,
          dayLow: priceCache.dayLow,
          week52High: priceCache.week52High,
          week52Low: priceCache.week52Low,
          source: 'cache',
          originalSource: priceCache.source,
          lastUpdate: priceCache.lastUpdate.toISOString(),
          isMarketOpen: marketStatus.isOpen,
          marketStatus: marketStatus.status,
          nextUpdateMinutes: marketStatus.minutesToNextUpdate,
          nextOpen: marketStatus.nextOpen?.toISOString() || null,
        },
        metadata: {
          cached: true,
          latency: Date.now() - startTime
        }
      }, {
        headers: {
          'Cache-Control': `public, max-age=${Math.floor(cacheTtl / 1000)}`,
        }
      });
    }

    // Fetch fresh data based on market status
    let priceData: PriceCache | null = null;

    if (marketStatus.isOpen) {
      // During market hours: prefer TwelveData
      priceData = await fetchFromTwelveData();

      // Fallback to Investing.com if TwelveData fails
      if (!priceData) {
        console.log('[RealtimePrice] TwelveData failed, trying Investing.com');
        priceData = await fetchFromInvesting();
      }
    } else {
      // Outside market hours: prefer Investing.com for real-time
      priceData = await fetchFromInvesting();

      // Fallback to TwelveData (last market close)
      if (!priceData) {
        console.log('[RealtimePrice] Investing.com failed, trying TwelveData');
        priceData = await fetchFromTwelveData();
      }
    }

    // Final fallback
    if (!priceData) {
      console.log('[RealtimePrice] All sources failed, trying exchange rate API');
      priceData = await fetchFromDolarHoy();
    }

    // Use cache as last resort
    if (!priceData && priceCache) {
      priceData = { ...priceCache, source: 'cache' };
    }

    // If still no data, return error with last known price
    if (!priceData) {
      return NextResponse.json({
        success: false,
        error: 'Unable to fetch price from any source',
        data: {
          price: 4200, // Reasonable fallback
          previousClose: 4200,
          change: 0,
          changePct: 0,
          dayHigh: null,
          dayLow: null,
          week52High: null,
          week52Low: null,
          source: 'fallback',
          lastUpdate: new Date().toISOString(),
          isMarketOpen: marketStatus.isOpen,
          marketStatus: marketStatus.status,
        }
      }, { status: 503 });
    }

    // Update cache
    priceCache = priceData;

    return NextResponse.json({
      success: true,
      data: {
        price: priceData.price,
        previousClose: priceData.previousClose,
        change: priceData.change,
        changePct: priceData.changePct,
        dayHigh: priceData.dayHigh,
        dayLow: priceData.dayLow,
        week52High: priceData.week52High,
        week52Low: priceData.week52Low,
        source: priceData.source,
        lastUpdate: priceData.lastUpdate.toISOString(),
        isMarketOpen: marketStatus.isOpen,
        marketStatus: marketStatus.status,
        nextUpdateMinutes: marketStatus.minutesToNextUpdate,
        nextOpen: marketStatus.nextOpen?.toISOString() || null,
      },
      metadata: {
        cached: false,
        latency: Date.now() - startTime
      }
    }, {
      headers: {
        'Cache-Control': `public, max-age=${Math.floor(cacheTtl / 1000)}`,
      }
    });

  } catch (error: any) {
    console.error('[RealtimePrice] Error:', error);

    // Return cached data on error
    if (priceCache) {
      return NextResponse.json({
        success: true,
        data: {
          price: priceCache.price,
          previousClose: priceCache.previousClose,
          change: priceCache.change,
          changePct: priceCache.changePct,
          dayHigh: priceCache.dayHigh,
          dayLow: priceCache.dayLow,
          week52High: priceCache.week52High,
          week52Low: priceCache.week52Low,
          source: 'cache',
          lastUpdate: priceCache.lastUpdate.toISOString(),
          isMarketOpen: false,
          marketStatus: 'closed',
        },
        metadata: {
          error: error.message,
          latency: Date.now() - startTime
        }
      });
    }

    return NextResponse.json({
      success: false,
      error: error.message,
    }, { status: 500 });
  }
}
