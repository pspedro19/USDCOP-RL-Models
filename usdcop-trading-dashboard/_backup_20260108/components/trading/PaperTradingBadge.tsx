'use client';

import { Badge } from '@/components/ui/badge';
import { Beaker } from 'lucide-react';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '@/components/ui/tooltip';

interface PaperTradingBadgeProps {
    isPaperMode: boolean;
}

export function PaperTradingBadge({ isPaperMode }: PaperTradingBadgeProps) {
    if (!isPaperMode) return null;

    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger>
                    <Badge
                        variant="outline"
                        className="flex items-center gap-1 bg-purple-500/10 border-purple-500 text-purple-600 hover:bg-purple-500/20"
                    >
                        <Beaker className="h-3 w-3" />
                        PAPER TRADING
                    </Badge>
                </TooltipTrigger>
                <TooltipContent>
                    <p>Sistema en modo simulación.</p>
                    <p>Las órdenes NO son enviadas al broker real.</p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
}
