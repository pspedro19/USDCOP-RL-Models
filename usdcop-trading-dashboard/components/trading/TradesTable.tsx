'use client';

import { useState, useMemo } from "react";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { useTradesHistory } from "@/hooks/useTradesHistory";
import { useSelectedModel } from "@/contexts/ModelContext";
import { Skeleton } from "@/components/ui/skeleton";
import { ChevronDown, ChevronUp, AlertTriangle } from "lucide-react";
import { TRAINING_PERIODS, getDataType } from "@/lib/config/training-periods";

interface TradesTableProps {
    initialLimit?: number;
    compact?: boolean;
}

// Market hours constants (COT timezone)
const MARKET_OPEN_HOUR = 8;      // 8:00 AM COT
const MARKET_OPEN_MINUTE = 0;
const MARKET_CLOSE_HOUR = 12;    // 12:55 PM COT
const MARKET_CLOSE_MINUTE = 55;

// Market hours in UTC (COT + 5 hours)
const MARKET_OPEN_UTC_HOUR = 13;  // 1:00 PM UTC = 8:00 AM COT
const MARKET_CLOSE_UTC_HOUR = 17; // 5:55 PM UTC = 12:55 PM COT
const MARKET_CLOSE_UTC_MINUTE = 55;

/**
 * Check if a timestamp is within market hours
 * Market hours: 8:00 AM - 12:55 PM COT, Monday-Friday
 *
 * Handles two timestamp conventions:
 * 1. Actual UTC (13:00-17:55 UTC) - V19 format
 * 2. COT stored as UTC (08:00-12:55 "UTC") - V20 format
 */
function isWithinMarketHours(timestamp: string | Date): boolean {
    const date = new Date(timestamp);

    // Check weekday using UTC day (works for both formats)
    const dayOfWeek = date.getUTCDay();
    if (dayOfWeek === 0 || dayOfWeek === 6) {
        return false; // Weekend
    }

    const hours = date.getUTCHours();
    const minutes = date.getUTCMinutes();
    const timeInMinutes = hours * 60 + minutes;

    // Market hours in COT format (8:00-12:55)
    const marketOpenCOT = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE;   // 480 (8:00)
    const marketCloseCOT = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE; // 775 (12:55)

    // Market hours in UTC format (13:00-17:55)
    const marketOpenUTC = MARKET_OPEN_UTC_HOUR * 60;  // 780 (13:00)
    const marketCloseUTC = MARKET_CLOSE_UTC_HOUR * 60 + MARKET_CLOSE_UTC_MINUTE; // 1075 (17:55)

    // Accept if within EITHER format (COT-as-UTC OR actual UTC)
    const withinCOTFormat = timeInMinutes >= marketOpenCOT && timeInMinutes <= marketCloseCOT;
    const withinUTCFormat = timeInMinutes >= marketOpenUTC && timeInMinutes <= marketCloseUTC;

    return withinCOTFormat || withinUTCFormat;
}

/**
 * Convert UTC timestamp to Colombia Time (COT = UTC-5)
 * Returns formatted date string in Colombian locale
 */
function formatToCOT(timestamp: string | Date, formatStr: 'short' | 'full' = 'short'): string {
    const date = new Date(timestamp);
    // Subtract 5 hours to convert UTC to COT
    const cotDate = new Date(date.getTime() - 5 * 60 * 60 * 1000);

    if (formatStr === 'short') {
        // Format: "06 Ene 09:50"
        const day = cotDate.getUTCDate().toString().padStart(2, '0');
        const months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
        const month = months[cotDate.getUTCMonth()];
        const hours = cotDate.getUTCHours().toString().padStart(2, '0');
        const minutes = cotDate.getUTCMinutes().toString().padStart(2, '0');
        return `${day} ${month} ${hours}:${minutes}`;
    } else {
        // Full format with day of week
        const days = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'];
        const dayOfWeek = days[cotDate.getUTCDay()];
        const day = cotDate.getUTCDate().toString().padStart(2, '0');
        const months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
        const month = months[cotDate.getUTCMonth()];
        const hours = cotDate.getUTCHours().toString().padStart(2, '0');
        const minutes = cotDate.getUTCMinutes().toString().padStart(2, '0');
        return `${dayOfWeek} ${day} ${month} ${hours}:${minutes}`;
    }
}

export function TradesTable({ initialLimit = 10, compact = false }: TradesTableProps) {
    const [isExpanded, setIsExpanded] = useState(false);
    const { modelId, model } = useSelectedModel();

    // Map frontend model ID to database model_id
    const dbModelId = modelId === 'ppo_v20_prod' ? 'ppo_v20_macro' : 'ppo_v1';

    // Fetch more trades to account for filtering
    // When expanded, fetch up to 200 trades (some will be filtered out)
    const fetchLimit = isExpanded ? 200 : Math.max(initialLimit * 3, 50);
    const { data, isLoading } = useTradesHistory(fetchLimit, dbModelId);

    // Filter trades to only show those within market hours (8am-12:55pm COT, Mon-Fri)
    const filteredTrades = useMemo(() => {
        if (!data?.trades) return [];

        return data.trades.filter((trade: any) => {
            const tradeTime = trade.timestamp || trade.entry_time || trade.open_time;
            if (!tradeTime) return false;
            return isWithinMarketHours(tradeTime);
        });
    }, [data?.trades]);

    // Count how many trades were filtered out (for debugging/info)
    const filteredOutCount = (data?.trades?.length || 0) - filteredTrades.length;

    if (isLoading || !data) {
        return <TradesTableSkeleton />;
    }

    // Get trades to display (limited by initialLimit when not expanded)
    const displayedTrades = isExpanded ? filteredTrades : filteredTrades.slice(0, initialLimit);
    const hasMoreTrades = filteredTrades.length > initialLimit;
    const totalTrades = filteredTrades.length;

    // Compact mode: no card wrapper, just the table
    if (compact) {
        return (
            <div className="bg-slate-900/40 border border-slate-800 rounded-lg overflow-hidden">
                <Table>
                    <TableHeader>
                        <TableRow className="border-slate-800 hover:bg-slate-800/20">
                            <TableHead className="text-slate-400 h-8 text-xs">Hora (COT)</TableHead>
                            <TableHead className="text-slate-400 h-8 text-xs">Lado</TableHead>
                            <TableHead className="text-slate-400 h-8 text-right text-xs">Entrada</TableHead>
                            <TableHead className="text-slate-400 h-8 text-right text-xs">Salida</TableHead>
                            <TableHead className="text-slate-400 h-8 text-right text-xs">PnL</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {displayedTrades.length === 0 ? (
                            <TableRow>
                                <TableCell colSpan={5} className="text-center text-slate-500 py-4 text-xs">
                                    Sin operaciones
                                </TableCell>
                            </TableRow>
                        ) : (
                            displayedTrades.map((trade: any) => (
                                <TableRow key={trade.trade_id} className="border-slate-800 hover:bg-slate-800/30 transition-colors">
                                    <TableCell className="text-xs text-slate-300 py-2">
                                        {formatToCOT(trade.timestamp || trade.entry_time)}
                                    </TableCell>
                                    <TableCell className="py-2">
                                        <Badge
                                            variant="outline"
                                            className={`text-[10px] h-4 px-1 ${['BUY', 'LONG'].includes(trade.side?.toUpperCase())
                                                    ? 'bg-green-500/10 text-green-400 border-green-500/30'
                                                    : 'bg-red-500/10 text-red-400 border-red-500/30'
                                                }`}
                                        >
                                            {['BUY', 'LONG'].includes(trade.side?.toUpperCase()) ? 'L' : 'S'}
                                        </Badge>
                                    </TableCell>
                                    <TableCell className="text-right font-mono text-xs text-slate-300 py-2">
                                        {trade.entry_price?.toFixed(0) || '-'}
                                    </TableCell>
                                    <TableCell className="text-right font-mono text-xs text-slate-300 py-2">
                                        {trade.exit_price ? trade.exit_price.toFixed(0) : '-'}
                                    </TableCell>
                                    <TableCell className={`text-right font-medium text-xs py-2 ${(trade.pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {(trade.pnl || 0) >= 0 ? '+' : ''}{(trade.pnl || 0).toFixed(0)}
                                    </TableCell>
                                </TableRow>
                            ))
                        )}
                    </TableBody>
                </Table>
            </div>
        );
    }

    return (
        <Card className="bg-slate-900/40 border-slate-800 backdrop-blur-sm">
            <CardHeader className="border-b border-slate-800/50 py-4">
                <CardTitle className="text-sm font-medium text-slate-300 uppercase tracking-wider flex justify-between items-center">
                    <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
                        <span>Historial de Trades</span>
                        {model && (
                            <Badge
                                variant="outline"
                                className="text-[10px] px-2"
                                style={{ backgroundColor: `${model.color}20`, color: model.color, borderColor: `${model.color}50` }}
                            >
                                {model.name}
                            </Badge>
                        )}
                        <Badge variant="outline" className="text-[10px] bg-cyan-500/10 text-cyan-400 border-cyan-500/30">
                            8am-12:55pm COT
                        </Badge>
                    </div>
                    <div className="flex items-center gap-3">
                        <span className="text-xs text-slate-500 font-normal">
                            {isExpanded ? `${totalTrades} operaciones` : `${Math.min(initialLimit, totalTrades)} de ${totalTrades}`}
                        </span>
                        {hasMoreTrades && (
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setIsExpanded(!isExpanded)}
                                className="h-7 px-2 text-xs text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10"
                            >
                                {isExpanded ? (
                                    <>
                                        <ChevronUp className="w-4 h-4 mr-1" />
                                        Comprimir
                                    </>
                                ) : (
                                    <>
                                        <ChevronDown className="w-4 h-4 mr-1" />
                                        Ver más
                                    </>
                                )}
                            </Button>
                        )}
                    </div>
                </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
                <div className={`overflow-auto ${isExpanded ? 'max-h-[600px]' : ''}`}>
                    <Table>
                        <TableHeader className="sticky top-0 bg-slate-900/95 z-10">
                            <TableRow className="border-slate-800 hover:bg-slate-800/20">
                                <TableHead className="text-slate-400 h-10 w-[60px]">#</TableHead>
                                <TableHead className="text-slate-400 h-10">Hora (COT)</TableHead>
                                <TableHead className="text-slate-400 h-10">Lado</TableHead>
                                <TableHead className="text-slate-400 h-10 text-right">Entrada</TableHead>
                                <TableHead className="text-slate-400 h-10 text-right">Salida</TableHead>
                                <TableHead className="text-slate-400 h-10 text-right">PnL</TableHead>
                                <TableHead className="text-slate-400 h-10 text-right">Dur.</TableHead>
                                <TableHead className="text-slate-400 h-10 text-center">Tipo</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {displayedTrades.length === 0 ? (
                                <TableRow>
                                    <TableCell colSpan={8} className="text-center text-slate-500 py-8">
                                        Sin operaciones completadas
                                    </TableCell>
                                </TableRow>
                            ) : (
                                displayedTrades.map((trade: any) => {
                                    // Determine data type based on timestamp using centralized config
                                    const tradeTime = new Date(trade.timestamp || trade.entry_time || trade.open_time);
                                    const periodType = getDataType(tradeTime);
                                    // Map period type to display: train/validation = backtest, test/oos = out_of_sample
                                    const dataType = trade.data_type || (periodType === 'train' || periodType === 'validation' ? 'backtest' : 'out_of_sample');

                                    return (
                                        <TableRow key={trade.trade_id} className="border-slate-800 hover:bg-slate-800/30 transition-colors">
                                            <TableCell className="font-mono text-xs text-slate-500">
                                                {trade.trade_id}
                                            </TableCell>
                                            <TableCell className="text-xs text-slate-300">
                                                {formatToCOT(trade.timestamp || trade.entry_time, 'full')}
                                            </TableCell>
                                            <TableCell>
                                                <Badge
                                                    variant="outline"
                                                    className={`text-[10px] h-5 px-1.5 ${['BUY', 'LONG'].includes(trade.side?.toUpperCase())
                                                            ? 'bg-green-500/10 text-green-400 border-green-500/30'
                                                            : 'bg-red-500/10 text-red-400 border-red-500/30'
                                                        }`}
                                                >
                                                    {['BUY', 'LONG'].includes(trade.side?.toUpperCase()) ? 'LONG' : 'SHORT'}
                                                </Badge>
                                            </TableCell>
                                            <TableCell className="text-right font-mono text-xs text-slate-300">
                                                {trade.entry_price?.toFixed(2) || '-'}
                                            </TableCell>
                                            <TableCell className="text-right font-mono text-xs text-slate-300">
                                                {trade.exit_price ? trade.exit_price.toFixed(2) : '-'}
                                            </TableCell>
                                            <TableCell className={`text-right font-medium text-xs ${(trade.pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                {(trade.pnl || 0) >= 0 ? '+' : ''}{(trade.pnl || 0).toFixed(2)}
                                            </TableCell>
                                            <TableCell className="text-right text-xs text-slate-500">
                                                {trade.duration_minutes ? `${trade.duration_minutes}m` : trade.duration_bars ? `${trade.duration_bars}b` : '-'}
                                            </TableCell>
                                            <TableCell className="text-center">
                                                <Badge
                                                    variant="outline"
                                                    className={`text-[10px] h-5 px-1.5 ${dataType === 'backtest'
                                                            ? 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                                                            : 'bg-purple-500/10 text-purple-400 border-purple-500/30'
                                                        }`}
                                                >
                                                    {dataType === 'backtest' ? 'BT' : 'OOS'}
                                                </Badge>
                                            </TableCell>
                                        </TableRow>
                                    );
                                })
                            )}
                        </TableBody>
                    </Table>
                </div>
            </CardContent>
        </Card>
    );
}

function TradesTableSkeleton() {
    return (
        <Card className="bg-slate-900/40 border-slate-800">
            <CardHeader className="border-b border-slate-800/50 py-4">
                <Skeleton className="h-4 w-32 bg-slate-800" />
            </CardHeader>
            <CardContent className="p-0">
                <div className="space-y-2 p-4">
                    {[1, 2, 3, 4, 5].map((i) => (
                        <Skeleton key={i} className="h-8 w-full bg-slate-800" />
                    ))}
                </div>
            </CardContent>
        </Card>
    );
}
