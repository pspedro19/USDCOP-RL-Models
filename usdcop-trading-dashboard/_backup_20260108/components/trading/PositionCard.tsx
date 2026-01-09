'use client';

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useLiveState } from "@/hooks/useLiveState";
import { Skeleton } from "@/components/ui/skeleton";
import { Clock, TrendingUp, TrendingDown } from "lucide-react";

export function PositionCard() {
    const { state, isLoading } = useLiveState();

    if (isLoading || !state) {
        return <PositionCardSkeleton />;
    }

    const isLong = state.position === 'LONG';
    const isShort = state.position === 'SHORT';
    const isFlat = state.position === 'FLAT';

    let positionColor = "bg-slate-700 text-slate-300";
    if (isLong) positionColor = "bg-green-500/20 text-green-400 border-green-500/50";
    if (isShort) positionColor = "bg-red-500/20 text-red-400 border-red-500/50";

    const pnlColor = state.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400";
    const pnlIcon = state.unrealized_pnl >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />;

    // Format timestamp to relative or local time
    const entryTime = state.entry_time ? new Date(state.entry_time).toLocaleTimeString() : '--:--';

    return (
        <Card className="bg-slate-900/40 border-slate-800 backdrop-blur-sm h-full">
            <CardContent className="p-6">
                <div className="flex justify-between items-start mb-6">
                    <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Current Position</h3>
                    <Badge variant="outline" className={`${positionColor} px-3 py-1 font-bold`}>
                        {state.position}
                    </Badge>
                </div>

                {isFlat ? (
                    <div className="flex flex-col items-center justify-center h-40 text-slate-500">
                        <div className="h-12 w-12 rounded-full bg-slate-800 flex items-center justify-center mb-3">
                            <Clock className="h-6 w-6" />
                        </div>
                        <p>Waiting for signal...</p>
                        <p className="text-xs mt-1">Market Status: {state.market_status}</p>
                    </div>
                ) : (
                    <div className="space-y-6">
                        {/* PnL Section */}
                        <div className="text-center p-4 bg-slate-800/30 rounded-lg border border-slate-700/30">
                            <div className="text-xs text-slate-400 mb-1">Unrealized P&L</div>
                            <div className={`text-3xl font-bold flex items-center justify-center gap-2 ${pnlColor}`}>
                                {state.unrealized_pnl >= 0 ? '+' : ''}{state.unrealized_pnl.toFixed(2)} USD
                            </div>
                            <div className={`text-sm font-medium ${pnlColor} mt-1`}>
                                ({state.unrealized_pnl_pct >= 0 ? '+' : ''}{state.unrealized_pnl_pct.toFixed(2)}%)
                            </div>
                        </div>

                        {/* Details Grid */}
                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <div className="text-slate-500 mb-1">Entry Price</div>
                                <div className="font-mono font-medium text-white">${state.entry_price.toFixed(2)}</div>
                            </div>
                            <div>
                                <div className="text-slate-500 mb-1">Current Price</div>
                                <div className="font-mono font-medium text-white">${state.current_price.toFixed(2)}</div>
                            </div>
                            <div>
                                <div className="text-slate-500 mb-1">Entry Time</div>
                                <div className="font-medium text-white">{entryTime}</div>
                            </div>
                            <div>
                                <div className="text-slate-500 mb-1">Duration</div>
                                <div className="font-medium text-white">{state.bars_in_position} bars</div>
                            </div>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

function PositionCardSkeleton() {
    return (
        <Card className="bg-slate-900/40 border-slate-800 h-full">
            <CardContent className="p-6">
                <div className="flex justify-between items-start mb-6">
                    <Skeleton className="h-4 w-32 bg-slate-800" />
                    <Skeleton className="h-6 w-16 bg-slate-800" />
                </div>
                <Skeleton className="h-32 w-full bg-slate-800 rounded-lg mb-4" />
                <div className="grid grid-cols-2 gap-4">
                    <Skeleton className="h-10 w-full bg-slate-800" />
                    <Skeleton className="h-10 w-full bg-slate-800" />
                </div>
            </CardContent>
        </Card>
    );
}
