'use client';

import { useRiskStatus } from '@/hooks/useRiskStatus';
import { AlertTriangle, ShieldAlert } from 'lucide-react';
import { cn } from '@/lib/utils';
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from '@/components/ui/tooltip';

export function KillSwitchIndicator() {
    const { killSwitchActive, riskLevel, cooldownActive, cooldownUntil } = useRiskStatus();

    // Nothing to show if safe
    if (!killSwitchActive && !cooldownActive && riskLevel === 'safe') {
        return null;
    }

    const getAlertContent = () => {
        if (killSwitchActive) {
            return {
                text: 'KILL SWITCH ACTIVADO',
                style: 'bg-red-600 text-white animate-pulse',
                icon: ShieldAlert,
                desc: 'Trading detenido por seguridad (Drawdown excesivo)'
            };
        }
        if (cooldownActive) {
            return {
                text: 'COOLDOWN ACTIVO',
                style: 'bg-orange-500 text-white',
                icon: AlertTriangle,
                desc: `Pausado tras pérdidas consecutivas hasta ${cooldownUntil?.toLocaleTimeString() ?? '...'}`
            };
        }
        // General Warning
        return {
            text: 'Riesgo Elevado',
            style: 'bg-yellow-500 text-black',
            icon: AlertTriangle,
            desc: 'Parámetros de riesgo cerca de límites'
        };
    };

    const alert = getAlertContent();
    const Icon = alert.icon;

    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger asChild>
                    <div className={cn(
                        'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold tracking-wide shadow-sm cursor-help transition-all duration-300',
                        alert.style
                    )}>
                        <Icon className="h-3.5 w-3.5" />
                        {alert.text}
                    </div>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-xs">
                    <p className="font-semibold">{alert.text}</p>
                    <p className="text-xs text-muted-foreground">{alert.desc}</p>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
}
