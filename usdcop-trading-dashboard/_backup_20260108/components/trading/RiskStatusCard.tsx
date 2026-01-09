// components/trading/RiskStatusCard.tsx
'use client';

import React from 'react';
import { useRiskStatus } from '@/hooks/useRiskStatus';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, Shield, XCircle, Clock, CheckCircle, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';
import { formatPct } from '@/lib/config/models.config';

export function RiskStatusCard() {
    const {
        killSwitchActive,
        currentDrawdownPct,
        dailyPnlPct,
        tradesToday,
        consecutiveLosses,
        cooldownActive,
        cooldownUntil,
        limits,
        riskLevel,
        tradingAllowed,
        warningMessages,
        isLoading,
        isError
    } = useRiskStatus();

    // Loading State
    if (isLoading) {
        return (
            <Card className="animate-pulse border-border/50">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Shield className="h-5 w-5 text-muted-foreground" />
                        Risk Status
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-32 bg-muted/20 rounded-lg" />
                </CardContent>
            </Card>
        );
    }

    // Error State (Mock visual if API is down for demo purposes, or actual error state)
    if (isError) {
        return (
            <Card className="border-red-500/30">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-500">
                        <AlertTriangle className="h-5 w-5" />
                        Risk Monitor Offline
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-muted-foreground">Could not connect to Risk Manager service.</p>
                </CardContent>
            </Card>
        );
    }

    // Visual Configurations based on Risk Level
    const cardStyles = {
        safe: 'bg-green-500/5 border-green-500/20',
        warning: 'bg-yellow-500/5 border-yellow-500/20',
        danger: 'bg-orange-500/5 border-orange-500/20',
        critical: 'bg-red-500/5 border-red-500/20',
    };

    const badgeStyles = {
        safe: 'bg-green-500/10 text-green-500 hover:bg-green-500/20',
        warning: 'bg-yellow-500/10 text-yellow-600 hover:bg-yellow-500/20',
        danger: 'bg-orange-500/10 text-orange-600 hover:bg-orange-500/20',
        critical: 'bg-red-500/10 text-red-600 hover:bg-red-500/20 animate-pulse',
    };

    const getMetricColor = (val: number, limit: number, inverse = false) => {
        const ratio = val / limit;
        if (inverse) {
            // For PnL (negative is bad)
            if (val < -limit) return 'text-red-500';
            if (val < -limit * 0.5) return 'text-yellow-500';
            return 'text-green-500';
        }
        // For Drawdown/Counts (higher is bad)
        if (ratio >= 1.0) return 'text-red-500 font-bold';
        if (ratio >= 0.7) return 'text-yellow-500';
        return 'text-foreground';
    };

    return (
        <Card className={cn('transition-all duration-300', cardStyles[riskLevel])}>
            <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-lg font-medium">
                        <Shield className={cn("h-5 w-5",
                            riskLevel === 'safe' ? 'text-green-500' :
                                riskLevel === 'critical' ? 'text-red-500' : 'text-yellow-500'
                        )} />
                        Safety Layer
                    </CardTitle>
                    <Badge variant="outline" className={cn("uppercase tracking-wider font-semibold", badgeStyles[riskLevel])}>
                        {riskLevel}
                    </Badge>
                </div>
            </CardHeader>

            <CardContent className="space-y-4">
                {/* Critical Alerts Banner */}
                {killSwitchActive && (
                    <div className="flex items-center gap-3 p-3 bg-red-500/15 rounded-md border border-red-500/30">
                        <XCircle className="h-6 w-6 text-red-500 shrink-0" />
                        <div>
                            <p className="font-bold text-red-600 dark:text-red-400">KILL SWITCH ENGAGED</p>
                            <p className="text-xs text-red-600/80 dark:text-red-400/80">Trading activities halted globally.</p>
                        </div>
                    </div>
                )}

                {cooldownActive && cooldownUntil && !killSwitchActive && (
                    <div className="flex items-center gap-3 p-3 bg-yellow-500/15 rounded-md border border-yellow-500/30">
                        <Clock className="h-6 w-6 text-yellow-600 shrink-0 animate-pulse" />
                        <div>
                            <p className="font-bold text-yellow-700 dark:text-yellow-400">Cooldown Active</p>
                            <p className="text-xs text-yellow-700/80 dark:text-yellow-400/80">
                                Resuming at {cooldownUntil.toLocaleTimeString()}
                            </p>
                        </div>
                    </div>
                )}

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-4 pt-1">
                    {/* Drawdown Metric */}
                    <div className="space-y-1 p-2 rounded-lg hover:bg-muted/50 transition-colors">
                        <p className="text-xs font-medium text-muted-foreground uppercase">Drawdown</p>
                        <div className="flex items-baseline gap-2">
                            <span className={cn("text-2xl font-bold tracking-tight",
                                getMetricColor(currentDrawdownPct, limits?.max_drawdown_pct || 15)
                            )}>
                                {currentDrawdownPct.toFixed(2)}%
                            </span>
                        </div>
                        <p className="text-[10px] text-muted-foreground">
                            Limit: {limits?.max_drawdown_pct}%
                        </p>
                    </div>

                    {/* Daily PnL Metric */}
                    <div className="space-y-1 p-2 rounded-lg hover:bg-muted/50 transition-colors">
                        <p className="text-xs font-medium text-muted-foreground uppercase">Daily PnL</p>
                        <div className="flex items-baseline gap-2">
                            <span className={cn("text-2xl font-bold tracking-tight",
                                dailyPnlPct >= 0 ? "text-green-500" :
                                    dailyPnlPct < -(limits?.max_daily_loss_pct || 5) ? "text-red-500" : "text-red-400"
                            )}>
                                {formatPct(dailyPnlPct)}
                            </span>
                        </div>
                        <p className="text-[10px] text-muted-foreground">
                            Stop: -{limits?.max_daily_loss_pct}%
                        </p>
                    </div>

                    {/* Trade Count */}
                    <div className="space-y-1 p-2 rounded-lg hover:bg-muted/50 transition-colors">
                        <p className="text-xs font-medium text-muted-foreground uppercase">Trades Today</p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-xl font-semibold">
                                {tradesToday}
                            </span>
                            <span className="text-xs text-muted-foreground">
                                / {limits?.max_trades_per_day}
                            </span>
                        </div>
                        <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                            <div
                                className={cn("h-full rounded-full transition-all",
                                    tradesToday >= (limits?.max_trades_per_day || 20) ? "bg-red-500" : "bg-primary"
                                )}
                                style={{ width: `${Math.min((tradesToday / (limits?.max_trades_per_day || 1)) * 100, 100)}%` }}
                            />
                        </div>
                    </div>

                    {/* Consecutive Losses */}
                    <div className="space-y-1 p-2 rounded-lg hover:bg-muted/50 transition-colors">
                        <p className="text-xs font-medium text-muted-foreground uppercase">Consec. Losses</p>
                        <div className="flex items-center gap-2">
                            <span className={cn("text-xl font-semibold",
                                consecutiveLosses >= 3 ? "text-red-500" : "text-foreground"
                            )}>
                                {consecutiveLosses}
                            </span>
                            <Activity className="h-4 w-4 text-muted-foreground opacity-50" />
                        </div>
                        <p className="text-[10px] text-muted-foreground">
                            Cooldown at 3
                        </p>
                    </div>
                </div>

                {/* System Status Footer */}
                <div className="pt-3 mt-1 border-t flex items-center justify-between">
                    <span className="text-xs font-medium text-muted-foreground">System Status</span>
                    <div className="flex items-center gap-1.5">
                        {tradingAllowed ? (
                            <>
                                <div className="h-2 w-2 rounded-full bg-green-500 animate-[pulse_3s_infinite]" />
                                <span className="text-xs font-semibold text-green-600 dark:text-green-400">OPERATIONAL</span>
                            </>
                        ) : (
                            <>
                                <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
                                <span className="text-xs font-semibold text-red-600 dark:text-red-400">HALTED</span>
                            </>
                        )}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}
