'use client';

import { Card, CardContent } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Activity, AlertTriangle } from "lucide-react";
import { usePerformanceSummary } from "@/hooks/usePerformanceSummary";
import { Skeleton } from "@/components/ui/skeleton";

export function KPICards() {
    const { summary, isLoading } = usePerformanceSummary();

    if (isLoading || !summary) {
        return <KPICardsSkeleton />;
    }

    const { metrics, comparison_vs_backtest: diff } = summary;

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {/* Sharpe Ratio */}
            <KPICard
                title="SHARPE RATIO"
                value={metrics.sharpe_ratio}
                format="0.00"
                subtext={`vs 2.91 Backtest (${diff.sharpe_diff > 0 ? '+' : ''}${diff.sharpe_diff.toFixed(2)})`}
                icon={<Activity className="h-4 w-4 text-cyan-400" />}
                trend={null}
            />

            {/* Max Drawdown */}
            <KPICard
                title="MAX DRAWDOWN"
                value={metrics.max_drawdown_pct}
                format="0.00%"
                subtext={`vs 0.68% Backtest`}
                icon={<AlertTriangle className="h-4 w-4 text-orange-400" />}
                trend={metrics.max_drawdown_pct > 1.0 ? "negative" : "neutral"}
                inverse={true}
            />

            {/* Win Rate */}
            <KPICard
                title="WIN RATE"
                value={metrics.win_rate}
                format="0.0%"
                subtext={`vs 44.8% Backtest`}
                icon={<TrendingUp className="h-4 w-4 text-purple-400" />}
                trend={metrics.win_rate > 45 ? "positive" : "neutral"}
            />

            {/* Total Return */}
            <KPICard
                title="TOTAL RETURN"
                value={metrics.total_return_pct}
                format="+0.00%"
                subtext="Out-of-Sample (8 days)"
                icon={<TrendingUp className="h-4 w-4 text-green-400" />}
                trend={metrics.total_return_pct > 0 ? "positive" : "negative"}
            />
        </div>
    );
}

function KPICard({ title, value, format, subtext, icon, trend, inverse = false }: any) {
    let formattedValue = value.toString();

    if (format.includes('%')) {
        formattedValue = `${value.toFixed(2)}%`;
    } else {
        formattedValue = value.toFixed(2);
    }

    if (format.startsWith('+') && value > 0) {
        formattedValue = `+${formattedValue}`;
    }

    let valueColor = "text-white";
    if (trend === "positive") valueColor = inverse ? "text-red-400" : "text-green-400";
    if (trend === "negative") valueColor = inverse ? "text-green-400" : "text-red-400";

    return (
        <Card className="bg-slate-900/40 border-slate-800 backdrop-blur-sm">
            <CardContent className="p-4">
                <div className="flex justify-between items-start mb-2">
                    <div className="text-xs font-medium text-slate-400">{title}</div>
                    {icon}
                </div>
                <div className={`text-2xl font-bold ${valueColor} mb-1`}>
                    {formattedValue}
                </div>
                <div className="text-xs text-slate-500">
                    {subtext}
                </div>
            </CardContent>
        </Card>
    );
}

function KPICardsSkeleton() {
    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {[1, 2, 3, 4].map((i) => (
                <Card key={i} className="bg-slate-900/40 border-slate-800">
                    <CardContent className="p-4">
                        <Skeleton className="h-4 w-20 mb-2 bg-slate-800" />
                        <Skeleton className="h-8 w-24 mb-1 bg-slate-800" />
                        <Skeleton className="h-3 w-32 bg-slate-800" />
                    </CardContent>
                </Card>
            ))}
        </div>
    );
}
