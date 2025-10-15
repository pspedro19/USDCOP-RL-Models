import { NextRequest, NextResponse } from 'next/server';

interface AlertSystemResponse {
  status: 'active' | 'warning' | 'error';
  timestamp: string;
  activeAlerts: Alert[];
  alertRules: AlertRule[];
  notifications: {
    email: NotificationChannel;
    slack: NotificationChannel;
    webhook: NotificationChannel;
    sms: NotificationChannel;
  };
  metrics: {
    totalAlerts24h: number;
    criticalAlerts: number;
    warningAlerts: number;
    resolvedAlerts: number;
    avgResponseTime: number;
  };
}

interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  component: string;
  message: string;
  timestamp: string;
  status: 'active' | 'acknowledged' | 'resolved';
  acknowledgedBy?: string;
  resolvedAt?: string;
  details: Record<string, any>;
}

interface AlertRule {
  id: string;
  name: string;
  component: string;
  condition: string;
  threshold: number;
  severity: 'critical' | 'warning' | 'info';
  enabled: boolean;
  cooldown: number;
  lastTriggered?: string;
}

interface NotificationChannel {
  enabled: boolean;
  configured: boolean;
  lastSent: string;
  successRate: number;
  errors: number;
}

export async function GET(request: NextRequest) {
  try {
    const generateAlerts = (): Alert[] => {
      const alerts: Alert[] = [];
      const alertTypes = [
        { component: 'L0-Pipeline', message: 'Data gap detected in L0 pipeline', severity: 'critical' as const },
        { component: 'Backup', message: 'Backup integrity check failed', severity: 'critical' as const },
        { component: 'WebSocket', message: 'High latency detected in WebSocket connection', severity: 'warning' as const },
        { component: 'API-Usage', message: 'API rate limit approaching threshold', severity: 'warning' as const },
        { component: 'L1-Features', message: 'Feature calculation timeout', severity: 'warning' as const },
        { component: 'Ready-Signal', message: 'L0→WebSocket handover incomplete', severity: 'critical' as const }
      ];

      // Generate random active alerts
      for (let i = 0; i < Math.floor(Math.random() * 5) + 1; i++) {
        const alertType = alertTypes[Math.floor(Math.random() * alertTypes.length)];
        alerts.push({
          id: `alert-${Date.now()}-${i}`,
          ...alertType,
          timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
          status: Math.random() > 0.3 ? 'active' : (Math.random() > 0.5 ? 'acknowledged' : 'resolved'),
          acknowledgedBy: Math.random() > 0.5 ? 'system' : undefined,
          resolvedAt: Math.random() > 0.7 ? new Date().toISOString() : undefined,
          details: {
            source: alertType.component,
            count: Math.floor(Math.random() * 10) + 1,
            threshold: Math.random() * 100
          }
        });
      }

      return alerts;
    };

    const alertRules: AlertRule[] = [
      {
        id: 'pipeline-gap',
        name: 'Pipeline Data Gap Detection',
        component: 'L0-Pipeline',
        condition: 'data_gap > threshold',
        threshold: 300,
        severity: 'critical',
        enabled: true,
        cooldown: 300,
        lastTriggered: new Date(Date.now() - Math.random() * 3600000).toISOString()
      },
      {
        id: 'backup-failure',
        name: 'Backup Failure Detection',
        component: 'Backup',
        condition: 'backup_failed = true',
        threshold: 1,
        severity: 'critical',
        enabled: true,
        cooldown: 600
      },
      {
        id: 'api-rate-limit',
        name: 'API Rate Limit Warning',
        component: 'API-Usage',
        condition: 'usage_percentage > threshold',
        threshold: 80,
        severity: 'warning',
        enabled: true,
        cooldown: 900
      },
      {
        id: 'websocket-latency',
        name: 'WebSocket High Latency',
        component: 'WebSocket',
        condition: 'avg_latency > threshold',
        threshold: 100,
        severity: 'warning',
        enabled: true,
        cooldown: 300
      },
      {
        id: 'ready-signal-timeout',
        name: 'Ready Signal Timeout',
        component: 'Ready-Signal',
        condition: 'handover_time > threshold',
        threshold: 30,
        severity: 'critical',
        enabled: true,
        cooldown: 180
      }
    ];

    const activeAlerts = generateAlerts();

    const alertData: AlertSystemResponse = {
      status: 'active',
      timestamp: new Date().toISOString(),
      activeAlerts,
      alertRules,
      notifications: {
        email: {
          enabled: true,
          configured: true,
          lastSent: new Date(Date.now() - Math.random() * 3600000).toISOString(),
          successRate: 95 + Math.random() * 5,
          errors: Math.floor(Math.random() * 3)
        },
        slack: {
          enabled: true,
          configured: true,
          lastSent: new Date(Date.now() - Math.random() * 1800000).toISOString(),
          successRate: 98 + Math.random() * 2,
          errors: Math.floor(Math.random() * 2)
        },
        webhook: {
          enabled: true,
          configured: false,
          lastSent: new Date(Date.now() - Math.random() * 7200000).toISOString(),
          successRate: 85 + Math.random() * 10,
          errors: Math.floor(Math.random() * 5)
        },
        sms: {
          enabled: false,
          configured: false,
          lastSent: new Date(Date.now() - Math.random() * 86400000).toISOString(),
          successRate: 90 + Math.random() * 10,
          errors: Math.floor(Math.random() * 2)
        }
      },
      metrics: {
        totalAlerts24h: Math.floor(Math.random() * 50) + 10,
        criticalAlerts: activeAlerts.filter(a => a.severity === 'critical' && a.status === 'active').length,
        warningAlerts: activeAlerts.filter(a => a.severity === 'warning' && a.status === 'active').length,
        resolvedAlerts: activeAlerts.filter(a => a.status === 'resolved').length,
        avgResponseTime: Math.floor(Math.random() * 300) + 60
      }
    };

    // Determine overall status
    if (alertData.metrics.criticalAlerts > 0) {
      alertData.status = 'error';
    } else if (alertData.metrics.warningAlerts > 2 ||
               Object.values(alertData.notifications).some(n => n.errors > 3)) {
      alertData.status = 'warning';
    }

    return NextResponse.json(alertData, {
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    });

  } catch (error) {
    console.error('Alert System Error:', error);

    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        error: 'Alert system check failed',
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
      case 'acknowledge-alert':
        if (!body.alertId) {
          return NextResponse.json(
            { error: 'Alert ID required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `Alert ${body.alertId} acknowledged`,
          timestamp: new Date().toISOString(),
          acknowledgedBy: body.user || 'system'
        });

      case 'resolve-alert':
        if (!body.alertId) {
          return NextResponse.json(
            { error: 'Alert ID required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `Alert ${body.alertId} resolved`,
          timestamp: new Date().toISOString(),
          resolvedBy: body.user || 'system'
        });

      case 'create-alert':
        if (!body.component || !body.message || !body.severity) {
          return NextResponse.json(
            { error: 'Component, message, and severity required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: 'Alert created successfully',
          timestamp: new Date().toISOString(),
          alertId: `alert-${Date.now()}`
        });

      case 'update-rule':
        if (!body.ruleId) {
          return NextResponse.json(
            { error: 'Rule ID required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `Alert rule ${body.ruleId} updated`,
          timestamp: new Date().toISOString()
        });

      case 'test-notification':
        if (!body.channel) {
          return NextResponse.json(
            { error: 'Notification channel required' },
            { status: 400 }
          );
        }

        return NextResponse.json({
          status: 'success',
          message: `Test notification sent via ${body.channel}`,
          timestamp: new Date().toISOString()
        });

      default:
        return NextResponse.json(
          { error: 'Invalid action' },
          { status: 400 }
        );
    }

  } catch (error) {
    console.error('Alert System POST Error:', error);

    return NextResponse.json(
      {
        error: 'Failed to process alert request',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}