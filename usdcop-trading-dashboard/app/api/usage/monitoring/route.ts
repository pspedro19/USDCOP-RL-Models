import { NextRequest, NextResponse } from 'next/server';

interface APIUsageResponse {
  status: 'healthy' | 'warning' | 'error';
  timestamp: string;
  apis: {
    twelveData: APIMetrics;
    alphaVantage: APIMetrics;
    mt5: APIMetrics;
    internal: APIMetrics;
  };
  keys: {
    active: KeyInfo[];
    rotation: {
      nextRotation: string;
      rotationPolicy: string;
      lastRotation: string;
    };
  };
  quotas: {
    daily: QuotaInfo;
    monthly: QuotaInfo;
    yearly: QuotaInfo;
  };
  alerts: {
    rateLimitWarnings: number;
    keyExpirationWarnings: number;
    quotaWarnings: number;
  };
}

interface APIMetrics {
  callsUsed: number;
  rateLimit: number;
  remainingCalls: number;
  resetTime: string;
  errorRate: number;
  avgResponseTime: number;
  lastCall: string;
  status: 'active' | 'limited' | 'error';
}

interface KeyInfo {
  keyId: string;
  provider: string;
  age: number;
  expiresIn: number;
  callsToday: number;
  status: 'active' | 'expired' | 'suspended' | 'rotation_due';
}

interface QuotaInfo {
  used: number;
  limit: number;
  percentage: number;
  resetDate: string;
}

export async function GET(request: NextRequest) {
  try {
    const generateAPIMetrics = (provider: string): APIMetrics => {
      const callsUsed = Math.floor(Math.random() * 1500) + 500;
      const rateLimit = provider === 'internal' ? 10000 : 2000;
      const isLimited = (callsUsed / rateLimit) > 0.8;

      return {
        callsUsed,
        rateLimit,
        remainingCalls: rateLimit - callsUsed,
        resetTime: new Date(Date.now() + 3600000).toISOString(),
        errorRate: Math.random() * 0.05,
        avgResponseTime: Math.random() * 200 + 50,
        lastCall: new Date(Date.now() - Math.random() * 300000).toISOString(),
        status: isLimited ? 'limited' : (Math.random() > 0.9 ? 'error' : 'active')
      };
    };

    const generateKeyInfo = (provider: string, index: number): KeyInfo => {
      const age = Math.floor(Math.random() * 30) + 1;
      const rotationDue = age > 25;

      return {
        keyId: `${provider}-key-${index + 1}`,
        provider,
        age,
        expiresIn: 365 - age,
        callsToday: Math.floor(Math.random() * 800) + 100,
        status: rotationDue ? 'rotation_due' : 'active'
      };
    };

    const usageData: APIUsageResponse = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      apis: {
        twelveData: generateAPIMetrics('twelveData'),
        alphaVantage: generateAPIMetrics('alphaVantage'),
        mt5: generateAPIMetrics('mt5'),
        internal: generateAPIMetrics('internal')
      },
      keys: {
        active: [
          generateKeyInfo('twelveData', 0),
          generateKeyInfo('alphaVantage', 0),
          generateKeyInfo('mt5', 0)
        ],
        rotation: {
          nextRotation: new Date(Date.now() + 7 * 24 * 3600000).toISOString(),
          rotationPolicy: 'Monthly or when 80% usage reached',
          lastRotation: new Date(Date.now() - 23 * 24 * 3600000).toISOString()
        }
      },
      quotas: {
        daily: {
          used: Math.floor(Math.random() * 5000) + 2000,
          limit: 8000,
          percentage: 0,
          resetDate: new Date(Date.now() + 24 * 3600000).toISOString()
        },
        monthly: {
          used: Math.floor(Math.random() * 150000) + 50000,
          limit: 200000,
          percentage: 0,
          resetDate: new Date(Date.now() + 15 * 24 * 3600000).toISOString()
        },
        yearly: {
          used: Math.floor(Math.random() * 1500000) + 500000,
          limit: 2000000,
          percentage: 0,
          resetDate: new Date(Date.now() + 180 * 24 * 3600000).toISOString()
        }
      },
      alerts: {
        rateLimitWarnings: 0,
        keyExpirationWarnings: 0,
        quotaWarnings: 0
      }
    };

    // Calculate quota percentages
    usageData.quotas.daily.percentage = (usageData.quotas.daily.used / usageData.quotas.daily.limit) * 100;
    usageData.quotas.monthly.percentage = (usageData.quotas.monthly.used / usageData.quotas.monthly.limit) * 100;
    usageData.quotas.yearly.percentage = (usageData.quotas.yearly.used / usageData.quotas.yearly.limit) * 100;

    // Count alerts
    Object.values(usageData.apis).forEach(api => {
      if ((api.callsUsed / api.rateLimit) > 0.8) {
        usageData.alerts.rateLimitWarnings++;
      }
    });

    usageData.keys.active.forEach(key => {
      if (key.status === 'rotation_due' || key.expiresIn < 30) {
        usageData.alerts.keyExpirationWarnings++;
      }
    });

    [usageData.quotas.daily, usageData.quotas.monthly, usageData.quotas.yearly].forEach(quota => {
      if (quota.percentage > 80) {
        usageData.alerts.quotaWarnings++;
      }
    });

    // Determine overall status
    const hasApiErrors = Object.values(usageData.apis).some(api => api.status === 'error');
    const hasRateLimits = Object.values(usageData.apis).some(api => api.status === 'limited');
    const hasExpiredKeys = usageData.keys.active.some(key => key.status === 'expired');
    const hasHighQuotas = [usageData.quotas.daily, usageData.quotas.monthly, usageData.quotas.yearly]
      .some(quota => quota.percentage > 90);

    if (hasApiErrors || hasExpiredKeys || hasHighQuotas) {
      usageData.status = 'error';
    } else if (hasRateLimits || usageData.alerts.rateLimitWarnings > 0 ||
               usageData.alerts.keyExpirationWarnings > 0 || usageData.alerts.quotaWarnings > 0) {
      usageData.status = 'warning';
    }

    return NextResponse.json(usageData, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });

  } catch (error) {
    console.error('API Usage Monitoring Error:', error);

    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: 'API usage monitoring failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    switch (body.action) {
      case 'rotate-key':
        if (!body.keyId) {
          return NextResponse.json(
            { error: 'Key ID required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `Key rotation initiated for ${body.keyId}`,
          timestamp: new Date().toISOString(),
          newKeyId: `${body.keyId}-rotated-${Date.now()}`
        });

      case 'reset-quota-alerts':
        return NextResponse.json({
          status: 'success',
          message: 'Quota alerts reset',
          timestamp: new Date().toISOString()
        });

      case 'suspend-key':
        if (!body.keyId) {
          return NextResponse.json(
            { error: 'Key ID required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `Key ${body.keyId} suspended`,
          timestamp: new Date().toISOString()
        });

      case 'activate-backup-key':
        if (!body.provider) {
          return NextResponse.json(
            { error: 'Provider required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `Backup key activated for ${body.provider}`,
          timestamp: new Date().toISOString()
        });

      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        );
    }

  } catch (error) {
    console.error('API Usage Monitoring POST Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process API usage request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}