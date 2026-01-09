'use client';

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useRiskStatus } from "@/hooks/useRiskStatus";
import { useLiveState } from "@/hooks/useLiveState";
import Link from "next/link";
import { ArrowLeft, ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { Progress } from "@/components/ui/progress";

export default function RiskPage() {
    const {
        riskLevel,
        warningMessages,
        limits,
        currentDrawdownPct,
        dailyPnlPct,
        tradesToday,
        consecutiveLosses,
        killSwitchActive,
        dailyBlocked,
        cooldownActive,
        cooldownUntil
    } = useRiskStatus();

    const { state } = useLiveState();
    const isMarketOpen = state?.market_status === 'OPEN';

    const getStatusColor = () => {
        if (riskLevel === 'critical') return "text-red-500";
        if (riskLevel === 'danger') return "text-orange-500";
        if (riskLevel === 'warning') return "text-yellow-500";
        return "text-green-500";
    };

    const getStatusIcon = () => {
        if (riskLevel === 'safe') return <ShieldCheck className="h-12 w-12 text-green-500" />;
        return <ShieldAlert className={`h-12 w-12 ${getStatusColor()}`} />;
    };

    return (
        <main className="min-h-screen bg-[#050816] text-slate-200">
            {/* Header */}
            <header className="border-b border-slate-800 bg-[#0A0E27]/50 backdrop-blur-md sticky top-0 z-50">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Link href="/">
                            <Button variant="ghost" size="icon" className="text-slate-400 hover:text-white">
                                <ArrowLeft className="h-5 w-5" />
                            </Button>
                        </Link>
                        <h1 className="text-xl font-bold text-white">Risk Management</h1>
                    </div>
                </div>
            </header>

            <div className="container mx-auto px-4 py-8 space-y-8">

                {/* System Status Banner */}
                <Card className="bg-[#0A0E27] border-slate-800">
                    <CardContent className="p-8 flex items-center justify-between">
                        <div className="flex items-center gap-6">
                            {getStatusIcon()}
                            <div>
                                <h2 className="text-2xl font-bold text-white mb-1">SYSTEM STATUS</h2>
                                <div className={`text-lg font-mono font-bold uppercase ${getStatusColor()}`}>
                                    {riskLevel}
                                </div>
                            </div>
                        </div>
                        <div className="flex gap-4">
                            <StatusBadge label="Kill Switch" active={killSwitchActive} />
                            <StatusBadge label="Daily Block" active={dailyBlocked} />
                            <StatusBadge label="Cooldown" active={cooldownActive} />
                        </div>
                    </CardContent>
                </Card>

                {/* Metrics & Limits Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {/* Limits Configuration */}
                    <Card className="bg-[#0A0E27] border-slate-800">
                        <CardHeader>
                            <CardTitle className="text-sm font-medium text-slate-400 uppercase">Risk Limits</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <LimitRow
                                label="Max Drawdown"
                                value={limits?.max_drawdown_pct}
                                current={currentDrawdownPct}
                                suffix="%"
                                inverse={true}
                            />
                            <LimitRow
                                label="Max Daily Loss"
                                value={limits?.max_daily_loss_pct}
                                current={-dailyPnlPct} // Invert PnL to show loss
                                suffix="%"
                                inverse={true}
                            />
                            <LimitRow
                                label="Max Trades / Day"
                                value={limits?.max_trades_per_day}
                                current={tradesToday}
                            />
                            <LimitRow
                                label="Cooldown Threshold"
                                value={3} // Hardcoded for simplified view or get from config
                                current={consecutiveLosses}
                                suffix=" losses"
                            />
                        </CardContent>
                    </Card>

                    {/* Operational Details */}
                    <Card className="bg-[#0A0E27] border-slate-800">
                        <CardHeader>
                            <CardTitle className="text-sm font-medium text-slate-400 uppercase">Operational Status</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <InfoRow label="Market Status" value={state?.market_status || 'UNKNOWN'} />
                            <InfoRow label="Trading Allowed" value={isMarketOpen && riskLevel === 'safe' ? "YES" : "NO"} highlight={isMarketOpen && riskLevel === 'safe'} />

                            {warningMessages.length > 0 && (
                                <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                                    <div className="flex items-center gap-2 text-yellow-500 font-bold mb-2">
                                        <AlertTriangle className="h-4 w-4" />
                                        Warnings
                                    </div>
                                    <ul className="list-disc list-inside space-y-1">
                                        {warningMessages.map((msg, idx) => (
                                            <li key={idx} className="text-sm text-yellow-200/80">{msg}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {cooldownActive && cooldownUntil && (
                                <div className="mt-6 p-4 bg-orange-500/10 border border-orange-500/30 rounded-lg">
                                    <div className="text-orange-500 font-bold mb-1">Cooldown Active</div>
                                    <div className="text-sm text-orange-200/80">
                                        Trading paused until {cooldownUntil.toLocaleTimeString()}
                                    </div>
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </div>
            </div>
        </main>
    );
}

function StatusBadge({ label, active }: { label: string, active: boolean }) {
    return (
        <div className={`flex flex-col items-center px-4 py-2 rounded-lg border ${active ? 'bg-red-500/20 border-red-500/50' : 'bg-slate-800 border-slate-700'}`}>
            <span className="text-xs text-slate-400 mb-1">{label}</span>
            <span className={`font-bold ${active ? 'text-red-400' : 'text-slate-500'}`}>
                {active ? 'ON' : 'OFF'}
            </span>
        </div>
    );
}

function LimitRow({ label, value, current, suffix = '', inverse = false }: any) {
    if (value === undefined || value === null) return null;

    // Calculate percentage for progress bar
    const safeCurrent = Math.max(0, current); // Don't show negative progress
    const pct = Math.min(100, (safeCurrent / value) * 100);

    // Color logic
    let progressColor = "bg-blue-500";
    if (pct > 50) progressColor = "bg-yellow-500";
    if (pct > 80) progressColor = "bg-red-500";

    return (
        <div>
            <div className="flex justify-between text-sm mb-1">
                <span className="text-slate-400">{label}</span>
                <span className="text-white font-mono">
                    <span className={pct > 80 ? "text-red-400" : "text-slate-200"}>
                        {current.toFixed(suffix === '%' ? 2 : 0)}
                    </span>
                    <span className="text-slate-600 mx-1">/</span>
                    {value}{suffix}
                </span>
            </div>
            <Progress value={pct} className="h-2 bg-slate-800" indicatorClassName={progressColor} />
        </div>
    );
}

function InfoRow({ label, value, highlight }: any) {
    return (
        <div className="flex justify-between items-center py-2 border-b border-slate-800/50 last:border-0">
            <span className="text-slate-400 text-sm">{label}</span>
            <span className={`font-bold ${highlight ? 'text-green-400' : 'text-slate-200'}`}>
                {value}
            </span>
        </div>
    );
}
