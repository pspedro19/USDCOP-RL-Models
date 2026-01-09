// Signal Cache API

import { NextRequest, NextResponse } from 'next/server';
import { createApiResponse } from '@/lib/types/api';
import { getSignalCache } from '@/lib/services/cache';
import { withAuth } from '@/lib/auth/api-auth';

const signalCache = getSignalCache();

export const GET = withAuth(async (request, { user }) => {
  const startTime = Date.now();

  try {
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type') || 'latest'; // latest, history, stats, all
    const symbol = searchParams.get('symbol') || 'USDCOP';
    const limit = parseInt(searchParams.get('limit') || '50');

    if (type === 'latest') {
      const latest = await signalCache.getLatest();
      const latency = Date.now() - startTime;

      if (!latest) {
        return NextResponse.json(
          createApiResponse(false, 'cached', {
            message: 'No latest signal found',
            latency
          }),
          { status: 404 }
        );
      }

      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            type: 'latest',
            signal: latest,
          },
          latency,
          cacheHit: true
        })
      );
    }

    if (type === 'history') {
      const history = await signalCache.getHistory(symbol, limit);
      const latency = Date.now() - startTime;

      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            type: 'history',
            symbol,
            count: history.length,
            signals: history,
          },
          latency,
          cacheHit: true
        })
      );
    }

    if (type === 'stats') {
      const stats = await signalCache.getSignalStats(symbol);
      const latency = Date.now() - startTime;

      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            type: 'stats',
            symbol,
            stats,
          },
          latency,
          cacheHit: true
        })
      );
    }

    if (type === 'all') {
      const latestBySymbol = await signalCache.getLatestBySymbol();
      const latency = Date.now() - startTime;

      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            type: 'all',
            count: Object.keys(latestBySymbol).length,
            signals: latestBySymbol,
          },
          latency,
          cacheHit: true
        })
      );
    }

    if (type === 'range') {
      const startTimeParam = searchParams.get('start');
      const endTime = searchParams.get('end');

      if (!startTimeParam || !endTime) {
        const latency = Date.now() - startTime;
        return NextResponse.json(
          createApiResponse(false, 'none', {
            error: 'Start and end time are required for range query',
            latency
          }),
          { status: 400 }
        );
      }

      const signals = await signalCache.getSignalsInRange(
        symbol,
        new Date(startTimeParam),
        new Date(endTime)
      );
      const latency = Date.now() - startTime;

      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            type: 'range',
            symbol,
            start: startTimeParam,
            end: endTime,
            count: signals.length,
            signals,
          },
          latency,
          cacheHit: true
        })
      );
    }

    const latency = Date.now() - startTime;
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Invalid type parameter',
        latency
      }),
      { status: 400 }
    );
  } catch (error) {
    const latency = Date.now() - startTime;
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to get signal cache data',
        message: (error as Error).message,
        latency
      }),
      { status: 500 }
    );
  }
});

export const POST = withAuth(async (request, { user }) => {
  const startTime = Date.now();

  try {
    const body = await request.json();
    const { signal, action } = body;

    if (!signal) {
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(false, 'none', {
          error: 'Signal data is required',
          latency
        }),
        { status: 400 }
      );
    }

    if (action === 'set-latest') {
      await signalCache.setLatest(signal);
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            message: 'Latest signal set successfully',
            signal_id: signal.signal_id,
          },
          latency
        })
      );
    }

    if (action === 'add-to-history') {
      const symbol = signal.symbol || 'USDCOP';
      await signalCache.addToHistory(symbol, signal);
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            message: 'Signal added to history',
            symbol,
            signal_id: signal.signal_id,
          },
          latency
        })
      );
    }

    if (action === 'set-and-add') {
      await signalCache.setLatest(signal);
      const symbol = signal.symbol || 'USDCOP';
      await signalCache.addToHistory(symbol, signal);
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            message: 'Signal set as latest and added to history',
            symbol,
            signal_id: signal.signal_id,
          },
          latency
        })
      );
    }

    const latency = Date.now() - startTime;
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Invalid action',
        latency
      }),
      { status: 400 }
    );
  } catch (error) {
    const latency = Date.now() - startTime;
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to update signal cache',
        message: (error as Error).message,
        latency
      }),
      { status: 500 }
    );
  }
});

export const DELETE = withAuth(async (request, { user }) => {
  const startTime = Date.now();

  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');

    if (symbol) {
      await signalCache.clearHistory(symbol);
      const latency = Date.now() - startTime;
      return NextResponse.json(
        createApiResponse(true, 'cached', {
          data: {
            message: `Signal history cleared for symbol: ${symbol}`,
          },
          latency
        })
      );
    }

    await signalCache.clearAll();
    const latency = Date.now() - startTime;
    return NextResponse.json(
      createApiResponse(true, 'cached', {
        data: {
          message: 'All signal cache data cleared',
        },
        latency
      })
    );
  } catch (error) {
    const latency = Date.now() - startTime;
    return NextResponse.json(
      createApiResponse(false, 'none', {
        error: 'Failed to clear signal cache',
        message: (error as Error).message,
        latency
      }),
      { status: 500 }
    );
  }
});
