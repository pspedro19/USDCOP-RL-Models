# PLAN DE REMEDIACIÓN DE WORKFLOWS OPERACIONALES
## USD/COP RL Trading System - Enterprise Operations Upgrade

**Documento**: WORKFLOW_REMEDIATION_PLAN_20260117.md
**Versión**: 1.0.0
**Fecha**: 2026-01-17
**Autor**: Trading Operations Team
**Estado**: APROBADO PARA IMPLEMENTACIÓN

---

## RESUMEN EJECUTIVO

### Situación Actual
- **Score Auditoría**: 57% (114/200 puntos)
- **Clasificación**: Alto Riesgo Operacional
- **Gaps Críticos**: 10 identificados
- **Áreas Débiles**: Governance (15%), Incident Response (37%), Rollback (47%)

### Objetivo
Elevar el sistema a **≥85% de madurez operacional** implementando:
- Control operacional completo desde dashboard
- Automatización de safety mechanisms
- Governance y compliance enterprise-grade
- Incident response profesional

### Timeline General
| Fase | Duración | Objetivo |
|------|----------|----------|
| **Fase 1: Safety Critical** | Semana 1 | Kill switch, Rollback UI, Alerting |
| **Fase 2: Operational Control** | Semana 2-3 | Promotion UI, Notifications, Governance |
| **Fase 3: Observability** | Semana 4 | Lineage, Dashboards, Reporting |
| **Fase 4: Compliance** | Semana 5-6 | Documentation, Audit trails, Policies |

### Impacto Esperado
- Score post-remediación: **≥170/200 (85%)**
- Reducción de riesgo operacional: **60%**
- Tiempo de respuesta a incidentes: **-70%**

---

# FASE 1: SAFETY CRITICAL (Semana 1)

## 1.1 KILL SWITCH UI [P0-CRITICAL]

### Problema
No existe botón de emergencia visible en el dashboard para detener trading inmediatamente.

### Solución

#### 1.1.1 Backend: Kill Switch API

**Archivo**: `services/inference_api/routers/operations.py` (NUEVO)

```python
"""
Operations Router - Kill Switch and Emergency Controls
=======================================================
P0 Critical: Emergency trading controls accessible via API

Endpoints:
- POST /operations/kill-switch - Stop all trading immediately
- POST /operations/resume - Resume trading after kill
- GET /operations/status - Get current operational status
- POST /operations/pause - Soft pause (finish current, no new)
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional
import logging
import asyncpg

router = APIRouter(prefix="/operations", tags=["operations"])
logger = logging.getLogger(__name__)

# Global state (should be Redis-backed in production)
_kill_switch_state = {
    "active": False,
    "activated_at": None,
    "activated_by": None,
    "reason": None,
    "mode": "normal"  # normal, paused, killed
}


class KillSwitchRequest(BaseModel):
    reason: str
    activated_by: str = "dashboard"
    close_positions: bool = True
    notify_team: bool = True


class KillSwitchResponse(BaseModel):
    success: bool
    mode: str
    activated_at: Optional[str]
    message: str
    positions_closed: int = 0


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def activate_kill_switch(
    request: KillSwitchRequest,
    background_tasks: BackgroundTasks
):
    """
    EMERGENCY: Stop all trading immediately.

    Actions:
    1. Set global kill switch flag
    2. Optionally close all open positions
    3. Notify team via configured channels
    4. Log to audit trail
    """
    global _kill_switch_state

    _kill_switch_state = {
        "active": True,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "activated_by": request.activated_by,
        "reason": request.reason,
        "mode": "killed"
    }

    positions_closed = 0

    # Close positions if requested
    if request.close_positions:
        positions_closed = await _close_all_positions()

    # Send notifications
    if request.notify_team:
        background_tasks.add_task(
            _send_kill_switch_notification,
            request.reason,
            request.activated_by
        )

    # Log to audit
    await _log_kill_switch_event(request)

    logger.critical(
        f"KILL SWITCH ACTIVATED by {request.activated_by}: {request.reason}"
    )

    return KillSwitchResponse(
        success=True,
        mode="killed",
        activated_at=_kill_switch_state["activated_at"],
        message=f"Trading stopped. {positions_closed} positions closed.",
        positions_closed=positions_closed
    )


@router.post("/resume")
async def resume_trading(
    resumed_by: str = "dashboard",
    confirmation_code: str = None
):
    """Resume trading after kill switch. Requires confirmation."""
    global _kill_switch_state

    if not _kill_switch_state["active"]:
        return {"success": True, "message": "Trading already active"}

    # Require confirmation for safety
    if confirmation_code != "CONFIRM_RESUME":
        raise HTTPException(
            status_code=400,
            detail="Must provide confirmation_code='CONFIRM_RESUME'"
        )

    _kill_switch_state = {
        "active": False,
        "activated_at": None,
        "activated_by": None,
        "reason": None,
        "mode": "normal"
    }

    logger.info(f"Trading RESUMED by {resumed_by}")

    return {
        "success": True,
        "mode": "normal",
        "message": "Trading resumed successfully"
    }


@router.get("/status")
async def get_operations_status():
    """Get current operational status."""
    return {
        "mode": _kill_switch_state["mode"],
        "kill_switch_active": _kill_switch_state["active"],
        "activated_at": _kill_switch_state["activated_at"],
        "activated_by": _kill_switch_state["activated_by"],
        "reason": _kill_switch_state["reason"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.post("/pause")
async def pause_trading(paused_by: str = "dashboard", reason: str = "Manual pause"):
    """Soft pause - finish current trade, no new trades."""
    global _kill_switch_state

    _kill_switch_state["mode"] = "paused"
    _kill_switch_state["reason"] = reason

    logger.warning(f"Trading PAUSED by {paused_by}: {reason}")

    return {"success": True, "mode": "paused", "message": "Trading paused"}


async def _close_all_positions():
    """Close all open positions."""
    # Implementation depends on broker integration
    return 0


async def _send_kill_switch_notification(reason: str, activated_by: str):
    """Send notification to team."""
    # Slack/Email integration
    pass


async def _log_kill_switch_event(request: KillSwitchRequest):
    """Log to audit trail."""
    # Database logging
    pass
```

#### 1.1.2 Frontend: Kill Switch Component

**Archivo**: `usdcop-trading-dashboard/components/operations/KillSwitch.tsx` (NUEVO)

```typescript
'use client';

import { useState } from 'react';
import { AlertTriangle, Power, PlayCircle } from 'lucide-react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useToast } from '@/components/ui/use-toast';

interface KillSwitchProps {
  className?: string;
  compact?: boolean;
}

export function KillSwitch({ className, compact = false }: KillSwitchProps) {
  const [isKilled, setIsKilled] = useState(false);
  const [reason, setReason] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [confirmCode, setConfirmCode] = useState('');
  const { toast } = useToast();

  const activateKillSwitch = async () => {
    if (!reason.trim()) {
      toast({
        title: 'Error',
        description: 'Debe proporcionar una razón',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('/api/operations/kill-switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason,
          activated_by: 'dashboard',
          close_positions: true,
          notify_team: true,
        }),
      });

      if (response.ok) {
        setIsKilled(true);
        toast({
          title: 'KILL SWITCH ACTIVADO',
          description: 'Trading detenido. Todas las posiciones cerradas.',
          variant: 'destructive',
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'No se pudo activar kill switch',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const resumeTrading = async () => {
    if (confirmCode !== 'CONFIRM_RESUME') {
      toast({
        title: 'Error',
        description: 'Código de confirmación incorrecto',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('/api/operations/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          resumed_by: 'dashboard',
          confirmation_code: confirmCode,
        }),
      });

      if (response.ok) {
        setIsKilled(false);
        setConfirmCode('');
        toast({
          title: 'Trading Resumido',
          description: 'El sistema ha vuelto a operar normalmente.',
        });
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'No se pudo resumir trading',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (compact) {
    return (
      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button
            variant={isKilled ? 'outline' : 'destructive'}
            size="sm"
            className={className}
          >
            <Power className="h-4 w-4 mr-1" />
            {isKilled ? 'KILLED' : 'KILL'}
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          {/* Dialog content */}
        </AlertDialogContent>
      </AlertDialog>
    );
  }

  return (
    <div className={`p-4 rounded-lg border-2 ${
      isKilled ? 'border-red-500 bg-red-50' : 'border-gray-200'
    } ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className={`h-5 w-5 ${
            isKilled ? 'text-red-500' : 'text-yellow-500'
          }`} />
          <span className="font-semibold">
            {isKilled ? 'TRADING DETENIDO' : 'Kill Switch'}
          </span>
        </div>

        {!isKilled ? (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" size="lg">
                <Power className="h-5 w-5 mr-2" />
                ACTIVAR KILL SWITCH
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle className="text-red-600">
                  ⚠️ Activar Kill Switch
                </AlertDialogTitle>
                <AlertDialogDescription>
                  Esta acción detendrá TODO el trading inmediatamente y cerrará
                  todas las posiciones abiertas. Esta acción es irreversible
                  hasta que se reactive manualmente.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <div className="py-4">
                <label className="text-sm font-medium">
                  Razón (requerido):
                </label>
                <Input
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="Ej: Pérdida excesiva, error del sistema..."
                  className="mt-1"
                />
              </div>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancelar</AlertDialogCancel>
                <AlertDialogAction
                  onClick={activateKillSwitch}
                  className="bg-red-600 hover:bg-red-700"
                  disabled={isLoading || !reason.trim()}
                >
                  {isLoading ? 'Activando...' : 'CONFIRMAR KILL SWITCH'}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        ) : (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" size="lg">
                <PlayCircle className="h-5 w-5 mr-2" />
                RESUMIR TRADING
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Resumir Trading</AlertDialogTitle>
                <AlertDialogDescription>
                  Para resumir el trading, escriba CONFIRM_RESUME
                </AlertDialogDescription>
              </AlertDialogHeader>
              <div className="py-4">
                <Input
                  value={confirmCode}
                  onChange={(e) => setConfirmCode(e.target.value)}
                  placeholder="CONFIRM_RESUME"
                />
              </div>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancelar</AlertDialogCancel>
                <AlertDialogAction
                  onClick={resumeTrading}
                  disabled={isLoading || confirmCode !== 'CONFIRM_RESUME'}
                >
                  {isLoading ? 'Resumiendo...' : 'Resumir'}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>
    </div>
  );
}
```

#### 1.1.3 Integración en Dashboard Header

**Archivo**: `usdcop-trading-dashboard/components/layout/DashboardHeader.tsx`

```typescript
// Agregar al header principal del dashboard
import { KillSwitch } from '@/components/operations/KillSwitch';

// En el JSX del header:
<div className="flex items-center gap-4">
  <KillSwitch compact />
  {/* Otros elementos del header */}
</div>
```

### Criterios de Aceptación
- [ ] Botón rojo visible en header de dashboard
- [ ] Confirmación con razón requerida
- [ ] Cierre de posiciones automático
- [ ] Notificación al equipo
- [ ] Log en audit trail
- [ ] Resumir requiere código de confirmación

---

## 1.2 ROLLBACK UI [P0-CRITICAL]

### Problema
No existe interfaz para hacer rollback de modelo desde el dashboard.

### Solución

#### 1.2.1 Backend: Rollback API

**Archivo**: `services/inference_api/routers/models.py` (AGREGAR)

```python
class RollbackRequest(BaseModel):
    target_version: Optional[int] = None  # None = previous
    reason: str
    initiated_by: str = "dashboard"


class RollbackResponse(BaseModel):
    success: bool
    previous_model: str
    new_model: str
    rollback_time_ms: float
    message: str


@router.post("/models/rollback", response_model=RollbackResponse)
async def rollback_model(
    request: RollbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Rollback to previous model version.

    Process:
    1. Get current production model
    2. Find previous version (or specified version)
    3. Validate previous model exists and is valid
    4. Atomic swap: demote current, promote previous
    5. Reload inference engine
    6. Notify team
    7. Log to audit
    """
    import time
    start = time.time()

    try:
        # Get current production
        current = await _get_production_model()
        if not current:
            raise HTTPException(404, "No production model found")

        # Get target version
        if request.target_version:
            target = await _get_model_by_version(request.target_version)
        else:
            target = await _get_previous_production()

        if not target:
            raise HTTPException(404, "No rollback target found")

        # Validate target model
        validation = await _validate_model_for_production(target)
        if not validation["valid"]:
            raise HTTPException(400, f"Target invalid: {validation['reason']}")

        # Atomic rollback
        async with get_db_transaction() as tx:
            # Demote current
            await tx.execute(
                "UPDATE model_registry SET status='archived' WHERE model_id=$1",
                current["model_id"]
            )
            # Promote target
            await tx.execute(
                "UPDATE model_registry SET status='deployed' WHERE model_id=$1",
                target["model_id"]
            )
            # Log rollback
            await tx.execute("""
                INSERT INTO model_audit_log
                (action, from_model, to_model, reason, initiated_by, timestamp)
                VALUES ('rollback', $1, $2, $3, $4, NOW())
            """, current["model_id"], target["model_id"],
                request.reason, request.initiated_by)

        # Reload inference engine
        await _reload_inference_engine()

        # Notify team
        background_tasks.add_task(
            _notify_rollback,
            current["model_id"],
            target["model_id"],
            request.reason
        )

        elapsed = (time.time() - start) * 1000

        logger.warning(
            f"ROLLBACK: {current['model_id']} -> {target['model_id']} "
            f"by {request.initiated_by}: {request.reason}"
        )

        return RollbackResponse(
            success=True,
            previous_model=current["model_id"],
            new_model=target["model_id"],
            rollback_time_ms=elapsed,
            message=f"Rollback completed in {elapsed:.0f}ms"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(500, f"Rollback failed: {str(e)}")


@router.get("/models/rollback-targets")
async def get_rollback_targets():
    """Get available models for rollback."""
    models = await _get_archived_models(limit=5)
    return {
        "current_production": await _get_production_model(),
        "available_targets": models,
        "recommendation": models[0] if models else None
    }
```

#### 1.2.2 Frontend: Rollback Component

**Archivo**: `usdcop-trading-dashboard/components/models/RollbackPanel.tsx` (NUEVO)

```typescript
'use client';

import { useState, useEffect } from 'react';
import { RotateCcw, AlertCircle, CheckCircle } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { useToast } from '@/components/ui/use-toast';

interface RollbackTarget {
  model_id: string;
  version: string;
  archived_at: string;
  metrics: {
    sharpe: number;
    win_rate: number;
  };
}

export function RollbackPanel() {
  const [targets, setTargets] = useState<RollbackTarget[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<string>('');
  const [reason, setReason] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    if (isOpen) {
      fetchTargets();
    }
  }, [isOpen]);

  const fetchTargets = async () => {
    try {
      const response = await fetch('/api/v1/models/rollback-targets');
      const data = await response.json();
      setTargets(data.available_targets);
      if (data.recommendation) {
        setSelectedTarget(data.recommendation.model_id);
      }
    } catch (error) {
      console.error('Failed to fetch rollback targets:', error);
    }
  };

  const executeRollback = async () => {
    if (!selectedTarget || !reason.trim()) {
      toast({
        title: 'Error',
        description: 'Seleccione un modelo y proporcione una razón',
        variant: 'destructive',
      });
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/models/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_version: null, // Use selected model
          reason,
          initiated_by: 'dashboard',
        }),
      });

      const result = await response.json();

      if (response.ok) {
        toast({
          title: 'Rollback Exitoso',
          description: `Modelo cambiado en ${result.rollback_time_ms.toFixed(0)}ms`,
        });
        setIsOpen(false);
      } else {
        throw new Error(result.detail);
      }
    } catch (error: any) {
      toast({
        title: 'Rollback Fallido',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2">
          <RotateCcw className="h-4 w-4" />
          Rollback
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <RotateCcw className="h-5 w-5" />
            Rollback de Modelo
          </DialogTitle>
          <DialogDescription>
            Revertir a una versión anterior del modelo de producción.
            Esta acción es inmediata y afecta el trading en vivo.
          </DialogDescription>
        </DialogHeader>

        <div className="py-4 space-y-4">
          <div>
            <Label className="text-sm font-medium">
              Seleccionar versión de destino:
            </Label>
            <RadioGroup
              value={selectedTarget}
              onValueChange={setSelectedTarget}
              className="mt-2 space-y-2"
            >
              {targets.map((target) => (
                <div
                  key={target.model_id}
                  className="flex items-center space-x-2 p-3 border rounded-lg hover:bg-gray-50"
                >
                  <RadioGroupItem value={target.model_id} id={target.model_id} />
                  <Label htmlFor={target.model_id} className="flex-1 cursor-pointer">
                    <div className="flex justify-between">
                      <span className="font-medium">{target.model_id}</span>
                      <span className="text-sm text-gray-500">
                        v{target.version}
                      </span>
                    </div>
                    <div className="text-sm text-gray-500">
                      Sharpe: {target.metrics.sharpe.toFixed(2)} |
                      Win Rate: {(target.metrics.win_rate * 100).toFixed(1)}%
                    </div>
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>

          <div>
            <Label htmlFor="reason">Razón del rollback (requerido):</Label>
            <Input
              id="reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Ej: Degradación de performance, error detectado..."
              className="mt-1"
            />
          </div>

          <div className="flex items-start gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <AlertCircle className="h-5 w-5 text-yellow-600 mt-0.5" />
            <div className="text-sm text-yellow-800">
              <strong>Advertencia:</strong> El rollback es inmediato. El modelo
              actual será archivado y el modelo seleccionado tomará el control
              del trading en vivo.
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Cancelar
          </Button>
          <Button
            onClick={executeRollback}
            disabled={isLoading || !selectedTarget || !reason.trim()}
            className="bg-orange-600 hover:bg-orange-700"
          >
            {isLoading ? 'Ejecutando...' : 'Ejecutar Rollback'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
```

### Criterios de Aceptación
- [ ] Botón Rollback visible en sección de modelos
- [ ] Muestra últimas 5 versiones disponibles
- [ ] Métricas de cada versión visible
- [ ] Razón requerida
- [ ] Rollback completa en <60 segundos
- [ ] Notificación automática al equipo
- [ ] Log en audit trail

---

## 1.3 AUTOMATIC ROLLBACK TRIGGERS [P0-CRITICAL]

### Problema
No existe rollback automático cuando el modelo falla o degrada.

### Solución

#### 1.3.1 Auto-Rollback Service

**Archivo**: `services/inference_api/services/auto_rollback.py` (NUEVO)

```python
"""
Automatic Rollback Service
==========================
Monitors model health and triggers rollback when thresholds exceeded.

Triggers:
- Error rate > 5% in 15 min window
- Inference latency P99 > 2 seconds
- Consecutive losses > configured limit
- Daily drawdown > configured limit
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Callable
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RollbackTrigger:
    name: str
    threshold: float
    current_value: float
    triggered: bool
    message: str


@dataclass
class AutoRollbackConfig:
    # Error monitoring
    max_error_rate: float = 0.05  # 5%
    error_window_minutes: int = 15

    # Latency monitoring
    max_latency_p99_ms: float = 2000  # 2 seconds
    latency_window_minutes: int = 5

    # Trading performance
    max_consecutive_losses: int = 10
    max_daily_drawdown_pct: float = 0.05  # 5%

    # Rollback settings
    cooldown_minutes: int = 60  # Min time between auto-rollbacks
    enabled: bool = True


class AutoRollbackMonitor:
    """
    Continuous monitoring for automatic rollback triggers.
    """

    def __init__(
        self,
        config: AutoRollbackConfig,
        rollback_callback: Callable,
        notify_callback: Callable
    ):
        self.config = config
        self.rollback_callback = rollback_callback
        self.notify_callback = notify_callback

        # Metrics windows
        self._errors = deque(maxlen=1000)
        self._latencies = deque(maxlen=1000)
        self._trades = deque(maxlen=100)

        # State
        self._last_rollback: Optional[datetime] = None
        self._running = False

    async def start(self):
        """Start monitoring loop."""
        self._running = True
        logger.info("AutoRollbackMonitor started")

        while self._running:
            try:
                await self._check_triggers()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)

    def stop(self):
        """Stop monitoring loop."""
        self._running = False
        logger.info("AutoRollbackMonitor stopped")

    def record_inference(self, success: bool, latency_ms: float):
        """Record an inference result."""
        now = datetime.now()
        self._errors.append((now, not success))
        self._latencies.append((now, latency_ms))

    def record_trade(self, pnl: float):
        """Record a trade result."""
        now = datetime.now()
        self._trades.append((now, pnl))

    async def _check_triggers(self) -> Optional[RollbackTrigger]:
        """Check all rollback triggers."""
        if not self.config.enabled:
            return None

        # Check cooldown
        if self._last_rollback:
            cooldown_end = self._last_rollback + timedelta(
                minutes=self.config.cooldown_minutes
            )
            if datetime.now() < cooldown_end:
                return None

        triggers = [
            self._check_error_rate(),
            self._check_latency(),
            self._check_consecutive_losses(),
            self._check_daily_drawdown(),
        ]

        for trigger in triggers:
            if trigger and trigger.triggered:
                await self._execute_rollback(trigger)
                return trigger

        return None

    def _check_error_rate(self) -> RollbackTrigger:
        """Check inference error rate."""
        cutoff = datetime.now() - timedelta(
            minutes=self.config.error_window_minutes
        )
        recent = [(t, e) for t, e in self._errors if t > cutoff]

        if len(recent) < 10:  # Need minimum samples
            return RollbackTrigger(
                name="error_rate",
                threshold=self.config.max_error_rate,
                current_value=0,
                triggered=False,
                message="Insufficient samples"
            )

        error_rate = sum(1 for _, e in recent if e) / len(recent)
        triggered = error_rate > self.config.max_error_rate

        return RollbackTrigger(
            name="error_rate",
            threshold=self.config.max_error_rate,
            current_value=error_rate,
            triggered=triggered,
            message=f"Error rate {error_rate:.1%} > {self.config.max_error_rate:.1%}"
        )

    def _check_latency(self) -> RollbackTrigger:
        """Check inference latency P99."""
        cutoff = datetime.now() - timedelta(
            minutes=self.config.latency_window_minutes
        )
        recent = [lat for t, lat in self._latencies if t > cutoff]

        if len(recent) < 10:
            return RollbackTrigger(
                name="latency_p99",
                threshold=self.config.max_latency_p99_ms,
                current_value=0,
                triggered=False,
                message="Insufficient samples"
            )

        p99 = sorted(recent)[int(len(recent) * 0.99)]
        triggered = p99 > self.config.max_latency_p99_ms

        return RollbackTrigger(
            name="latency_p99",
            threshold=self.config.max_latency_p99_ms,
            current_value=p99,
            triggered=triggered,
            message=f"Latency P99 {p99:.0f}ms > {self.config.max_latency_p99_ms:.0f}ms"
        )

    def _check_consecutive_losses(self) -> RollbackTrigger:
        """Check consecutive losing trades."""
        losses = 0
        for _, pnl in reversed(list(self._trades)):
            if pnl < 0:
                losses += 1
            else:
                break

        triggered = losses >= self.config.max_consecutive_losses

        return RollbackTrigger(
            name="consecutive_losses",
            threshold=self.config.max_consecutive_losses,
            current_value=losses,
            triggered=triggered,
            message=f"{losses} consecutive losses >= {self.config.max_consecutive_losses}"
        )

    def _check_daily_drawdown(self) -> RollbackTrigger:
        """Check daily drawdown percentage."""
        today = datetime.now().date()
        today_trades = [pnl for t, pnl in self._trades if t.date() == today]

        if not today_trades:
            return RollbackTrigger(
                name="daily_drawdown",
                threshold=self.config.max_daily_drawdown_pct,
                current_value=0,
                triggered=False,
                message="No trades today"
            )

        cumsum = 0
        peak = 0
        max_dd = 0

        for pnl in today_trades:
            cumsum += pnl
            peak = max(peak, cumsum)
            dd = (peak - cumsum) / max(abs(peak), 1)
            max_dd = max(max_dd, dd)

        triggered = max_dd > self.config.max_daily_drawdown_pct

        return RollbackTrigger(
            name="daily_drawdown",
            threshold=self.config.max_daily_drawdown_pct,
            current_value=max_dd,
            triggered=triggered,
            message=f"Daily DD {max_dd:.1%} > {self.config.max_daily_drawdown_pct:.1%}"
        )

    async def _execute_rollback(self, trigger: RollbackTrigger):
        """Execute automatic rollback."""
        logger.critical(
            f"AUTO-ROLLBACK TRIGGERED: {trigger.name} - {trigger.message}"
        )

        try:
            # Notify first
            await self.notify_callback(
                f"🚨 AUTO-ROLLBACK: {trigger.name}\n{trigger.message}"
            )

            # Execute rollback
            result = await self.rollback_callback(
                reason=f"Auto-rollback: {trigger.message}",
                initiated_by="auto_rollback_monitor"
            )

            self._last_rollback = datetime.now()

            logger.info(f"Auto-rollback completed: {result}")

        except Exception as e:
            logger.error(f"Auto-rollback failed: {e}")
            await self.notify_callback(
                f"🚨 AUTO-ROLLBACK FAILED: {e}"
            )
```

### Criterios de Aceptación
- [ ] Monitoreo continuo cada 30 segundos
- [ ] Rollback automático si error rate > 5%
- [ ] Rollback automático si latencia P99 > 2s
- [ ] Rollback automático si 10+ pérdidas consecutivas
- [ ] Rollback automático si drawdown diario > 5%
- [ ] Cooldown de 60 min entre auto-rollbacks
- [ ] Notificación inmediata al equipo

---

## 1.4 NOTIFICATION SYSTEM [P0-CRITICAL]

### Problema
No hay notificaciones Slack/email para eventos críticos.

### Solución

**Archivo**: `services/shared/notifications/notifier.py` (NUEVO)

```python
"""
Notification Service
====================
Unified notification system for critical events.

Channels:
- Slack (primary)
- Email (backup)
- PagerDuty (P0 only)
"""

import os
import logging
import aiohttp
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    P0 = "p0"


@dataclass
class NotificationConfig:
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#trading-alerts"
    email_recipients: List[str] = None
    pagerduty_key: Optional[str] = None
    enabled: bool = True


class NotificationService:
    """
    Unified notification service.
    """

    LEVEL_EMOJI = {
        NotificationLevel.INFO: "ℹ️",
        NotificationLevel.WARNING: "⚠️",
        NotificationLevel.CRITICAL: "🚨",
        NotificationLevel.P0: "🔴",
    }

    def __init__(self, config: NotificationConfig = None):
        self.config = config or NotificationConfig(
            slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
            email_recipients=os.environ.get("ALERT_EMAILS", "").split(","),
            pagerduty_key=os.environ.get("PAGERDUTY_KEY"),
        )

    async def notify(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        title: Optional[str] = None,
        details: Optional[dict] = None,
        channels: List[str] = None
    ):
        """
        Send notification to configured channels.

        Args:
            message: Main notification message
            level: Notification severity level
            title: Optional title
            details: Optional additional details
            channels: Override default channels ['slack', 'email', 'pagerduty']
        """
        if not self.config.enabled:
            logger.debug(f"Notifications disabled, would send: {message}")
            return

        channels = channels or self._get_channels_for_level(level)

        tasks = []

        if "slack" in channels and self.config.slack_webhook_url:
            tasks.append(self._send_slack(message, level, title, details))

        if "email" in channels and self.config.email_recipients:
            tasks.append(self._send_email(message, level, title, details))

        if "pagerduty" in channels and self.config.pagerduty_key:
            tasks.append(self._send_pagerduty(message, level, title, details))

        if tasks:
            import asyncio
            await asyncio.gather(*tasks, return_exceptions=True)

    def _get_channels_for_level(self, level: NotificationLevel) -> List[str]:
        """Get appropriate channels for notification level."""
        if level == NotificationLevel.P0:
            return ["slack", "email", "pagerduty"]
        elif level == NotificationLevel.CRITICAL:
            return ["slack", "email"]
        elif level == NotificationLevel.WARNING:
            return ["slack"]
        else:
            return ["slack"]

    async def _send_slack(
        self,
        message: str,
        level: NotificationLevel,
        title: Optional[str],
        details: Optional[dict]
    ):
        """Send Slack notification."""
        emoji = self.LEVEL_EMOJI.get(level, "")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title or level.value.upper()}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            }
        ]

        if details:
            fields = [
                {"type": "mrkdwn", "text": f"*{k}:* {v}"}
                for k, v in details.items()
            ]
            blocks.append({
                "type": "section",
                "fields": fields[:10]  # Slack limit
            })

        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} COT"
                }
            ]
        })

        payload = {
            "channel": self.config.slack_channel,
            "blocks": blocks
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Slack notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Slack notification error: {e}")

    async def _send_email(
        self,
        message: str,
        level: NotificationLevel,
        title: Optional[str],
        details: Optional[dict]
    ):
        """Send email notification."""
        # Implementation with SMTP or SendGrid
        pass

    async def _send_pagerduty(
        self,
        message: str,
        level: NotificationLevel,
        title: Optional[str],
        details: Optional[dict]
    ):
        """Send PagerDuty alert."""
        # Implementation with PagerDuty Events API
        pass


# Convenience functions
_notifier: Optional[NotificationService] = None


def get_notifier() -> NotificationService:
    global _notifier
    if _notifier is None:
        _notifier = NotificationService()
    return _notifier


async def notify_kill_switch(reason: str, activated_by: str):
    """Notify kill switch activation."""
    await get_notifier().notify(
        message=f"Kill switch activated: {reason}",
        level=NotificationLevel.P0,
        title="KILL SWITCH ACTIVATED",
        details={
            "Activated By": activated_by,
            "Action": "All trading stopped",
            "Runbook": "https://docs/runbook#kill-switch"
        }
    )


async def notify_rollback(from_model: str, to_model: str, reason: str):
    """Notify model rollback."""
    await get_notifier().notify(
        message=f"Model rollback executed: {from_model} → {to_model}",
        level=NotificationLevel.CRITICAL,
        title="MODEL ROLLBACK",
        details={
            "Previous Model": from_model,
            "New Model": to_model,
            "Reason": reason,
            "Runbook": "https://docs/runbook#rollback"
        }
    )


async def notify_promotion(model_id: str, stage: str, promoted_by: str):
    """Notify model promotion."""
    await get_notifier().notify(
        message=f"Model {model_id} promoted to {stage}",
        level=NotificationLevel.WARNING,
        title="MODEL PROMOTED",
        details={
            "Model": model_id,
            "Stage": stage,
            "Promoted By": promoted_by
        }
    )
```

### Criterios de Aceptación
- [ ] Slack webhook configurado
- [ ] Notificaciones para: kill switch, rollback, promoción, errores
- [ ] Diferentes niveles: INFO, WARNING, CRITICAL, P0
- [ ] PagerDuty para P0 únicamente
- [ ] Timestamps en COT
- [ ] Links a runbooks en mensajes

---

# FASE 2: OPERATIONAL CONTROL (Semanas 2-3)

## 2.1 PROMOTION UI [P1-HIGH]

### Problema
No existe botón de promoción en el dashboard.

### Solución

**Archivo**: `usdcop-trading-dashboard/components/models/PromoteButton.tsx` (NUEVO)

```typescript
'use client';

import { useState } from 'react';
import { ArrowUpCircle, Shield, CheckCircle2 } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import { useToast } from '@/components/ui/use-toast';
import { Badge } from '@/components/ui/badge';

interface PromoteButtonProps {
  modelId: string;
  currentStage: 'registered' | 'staging' | 'deployed';
  metrics: {
    sharpe: number;
    win_rate: number;
    max_drawdown: number;
    total_trades: number;
  };
}

const PROMOTION_THRESHOLDS = {
  staging: {
    min_sharpe: 0.5,
    min_win_rate: 0.45,
    max_drawdown: -0.15,
    min_trades: 50,
  },
  production: {
    min_sharpe: 1.0,
    min_win_rate: 0.50,
    max_drawdown: -0.10,
    min_trades: 100,
    min_staging_days: 7,
  },
};

export function PromoteButton({ modelId, currentStage, metrics }: PromoteButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [reason, setReason] = useState('');
  const [checklist, setChecklist] = useState({
    backtest_reviewed: false,
    metrics_acceptable: false,
    team_notified: false,
  });
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const targetStage = currentStage === 'registered' ? 'staging' : 'production';
  const thresholds = PROMOTION_THRESHOLDS[targetStage];

  const metricsPass = {
    sharpe: metrics.sharpe >= thresholds.min_sharpe,
    win_rate: metrics.win_rate >= thresholds.min_win_rate,
    drawdown: metrics.max_drawdown >= thresholds.max_drawdown,
    trades: metrics.total_trades >= thresholds.min_trades,
  };

  const allMetricsPass = Object.values(metricsPass).every(Boolean);
  const allChecklistComplete = Object.values(checklist).every(Boolean);
  const canPromote = allMetricsPass && allChecklistComplete && reason.trim();

  const handlePromote = async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`/api/v1/models/${modelId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_stage: targetStage,
          reason,
          promoted_by: 'dashboard',
          checklist,
        }),
      });

      if (response.ok) {
        toast({
          title: 'Promoción Exitosa',
          description: `${modelId} promovido a ${targetStage}`,
        });
        setIsOpen(false);
      } else {
        const error = await response.json();
        throw new Error(error.detail);
      }
    } catch (error: any) {
      toast({
        title: 'Promoción Fallida',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (currentStage === 'deployed') {
    return (
      <Badge variant="default" className="bg-green-600">
        <CheckCircle2 className="h-3 w-3 mr-1" />
        Production
      </Badge>
    );
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="default" size="sm" className="gap-2">
          <ArrowUpCircle className="h-4 w-4" />
          Promover a {targetStage === 'staging' ? 'Staging' : 'Production'}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Promover {modelId} a {targetStage}
          </DialogTitle>
          <DialogDescription>
            Revise los requisitos antes de promover.
          </DialogDescription>
        </DialogHeader>

        <div className="py-4 space-y-6">
          {/* Metrics validation */}
          <div>
            <h4 className="font-medium mb-2">Validación de Métricas</h4>
            <div className="grid grid-cols-2 gap-2">
              <MetricCheck
                label="Sharpe Ratio"
                value={metrics.sharpe.toFixed(2)}
                threshold={`≥ ${thresholds.min_sharpe}`}
                passed={metricsPass.sharpe}
              />
              <MetricCheck
                label="Win Rate"
                value={`${(metrics.win_rate * 100).toFixed(1)}%`}
                threshold={`≥ ${(thresholds.min_win_rate * 100).toFixed(0)}%`}
                passed={metricsPass.win_rate}
              />
              <MetricCheck
                label="Max Drawdown"
                value={`${(metrics.max_drawdown * 100).toFixed(1)}%`}
                threshold={`≥ ${(thresholds.max_drawdown * 100).toFixed(0)}%`}
                passed={metricsPass.drawdown}
              />
              <MetricCheck
                label="Total Trades"
                value={metrics.total_trades.toString()}
                threshold={`≥ ${thresholds.min_trades}`}
                passed={metricsPass.trades}
              />
            </div>
          </div>

          {/* Checklist */}
          <div>
            <h4 className="font-medium mb-2">Checklist de Promoción</h4>
            <div className="space-y-2">
              <ChecklistItem
                id="backtest"
                label="He revisado el reporte de backtest completo"
                checked={checklist.backtest_reviewed}
                onChange={(checked) =>
                  setChecklist({ ...checklist, backtest_reviewed: checked })
                }
              />
              <ChecklistItem
                id="metrics"
                label="Las métricas son aceptables para producción"
                checked={checklist.metrics_acceptable}
                onChange={(checked) =>
                  setChecklist({ ...checklist, metrics_acceptable: checked })
                }
              />
              <ChecklistItem
                id="team"
                label="El equipo ha sido notificado de esta promoción"
                checked={checklist.team_notified}
                onChange={(checked) =>
                  setChecklist({ ...checklist, team_notified: checked })
                }
              />
            </div>
          </div>

          {/* Reason */}
          <div>
            <h4 className="font-medium mb-2">Razón de la Promoción</h4>
            <Textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Describa por qué este modelo debe ser promovido..."
              rows={3}
            />
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setIsOpen(false)}>
            Cancelar
          </Button>
          <Button
            onClick={handlePromote}
            disabled={!canPromote || isLoading}
            className={targetStage === 'production' ? 'bg-green-600' : ''}
          >
            {isLoading ? 'Promoviendo...' : `Promover a ${targetStage}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function MetricCheck({
  label,
  value,
  threshold,
  passed,
}: {
  label: string;
  value: string;
  threshold: string;
  passed: boolean;
}) {
  return (
    <div
      className={`p-2 rounded border ${
        passed ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'
      }`}
    >
      <div className="text-xs text-gray-500">{label}</div>
      <div className="flex justify-between items-center">
        <span className="font-medium">{value}</span>
        <span className={`text-xs ${passed ? 'text-green-600' : 'text-red-600'}`}>
          {threshold}
        </span>
      </div>
    </div>
  );
}

function ChecklistItem({
  id,
  label,
  checked,
  onChange,
}: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-center space-x-2">
      <Checkbox
        id={id}
        checked={checked}
        onCheckedChange={onChange}
      />
      <label htmlFor={id} className="text-sm cursor-pointer">
        {label}
      </label>
    </div>
  );
}
```

---

## 2.2 MODEL GOVERNANCE POLICIES [P1-HIGH]

### Problema
No existen políticas formales de governance para modelos.

### Solución

**Archivo**: `docs/MODEL_GOVERNANCE_POLICY.md` (NUEVO)

```markdown
# MODEL GOVERNANCE POLICY
## USD/COP RL Trading System

**Version**: 1.0.0
**Effective Date**: 2026-01-17
**Owner**: Trading Operations Team
**Review Frequency**: Quarterly

---

## 1. MODEL LIFECYCLE STAGES

### 1.1 Stage Definitions

| Stage | Description | Duration | Exit Criteria |
|-------|-------------|----------|---------------|
| **Development** | Training and initial validation | Variable | Backtest passes thresholds |
| **Registered** | Backtest validated, pending review | ≤7 days | Manual promotion to Staging |
| **Staging** | Shadow mode validation | ≥7 days | Shadow metrics pass |
| **Production** | Live trading | Indefinite | Degradation or replacement |
| **Archived** | Retired from active use | Indefinite | None |

### 1.2 Stage Transitions

```
Development → Registered: Automatic (backtest passes)
Registered → Staging: Manual approval required
Staging → Production: Manual approval + 7-day minimum
Production → Archived: On replacement or manual retirement
Any Stage → Archived: Emergency retirement
```

---

## 2. PROMOTION REQUIREMENTS

### 2.1 Registered → Staging

**Quantitative Requirements:**
- Sharpe Ratio ≥ 0.5
- Win Rate ≥ 45%
- Max Drawdown ≤ 15%
- Total Trades ≥ 50 (in backtest)
- Out-of-sample validation completed

**Qualitative Requirements:**
- Backtest report reviewed by model owner
- Training artifacts logged in MLflow
- norm_stats.json hash verified
- dataset version tracked in DVC

**Approvers:**
- Model Owner (required)

### 2.2 Staging → Production

**Quantitative Requirements:**
- Sharpe Ratio ≥ 1.0
- Win Rate ≥ 50%
- Max Drawdown ≤ 10%
- Total Trades ≥ 100 (staging period)
- Minimum 7 days in Staging
- Shadow mode agreement rate ≥ 85%

**Qualitative Requirements:**
- Staging performance report reviewed
- No critical alerts during staging
- Team notified of promotion intent
- Rollback plan documented

**Approvers:**
- Model Owner (required)
- Engineering Lead (required for first production deployment)

### 2.3 Emergency Promotion

In exceptional circumstances, requirements may be waived by:
- CTO approval (documented in writing)
- Post-promotion review within 48 hours
- Enhanced monitoring for 14 days

---

## 3. MODEL OWNERSHIP

### 3.1 Roles and Responsibilities

**Model Owner:**
- Primary accountability for model performance
- Reviews and approves promotions
- Responds to degradation alerts
- Documents model decisions

**Engineering Lead:**
- Secondary approval for production deployments
- Reviews promotion process compliance
- Escalation point for issues

**On-Call Engineer:**
- Monitors model health during off-hours
- Executes emergency procedures (kill switch, rollback)
- Documents incidents

### 3.2 Assignment

Every model in Production or Staging MUST have:
- Assigned Model Owner (individual, not team)
- Backup Owner for coverage
- Documented in model_registry.owner_id field

---

## 4. MONITORING AND ALERTING

### 4.1 Required Metrics

All Production models must have:
- Sharpe ratio (rolling 30-day)
- Win rate (rolling 100 trades)
- Daily P&L
- Maximum drawdown (daily)
- Inference latency (P50, P99)
- Error rate

### 4.2 Alert Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Sharpe (30d) | < 0.8 | < 0.5 | Review model |
| Win Rate | < 48% | < 45% | Investigate |
| Daily Drawdown | > 3% | > 5% | Auto-pause |
| Error Rate | > 1% | > 5% | Auto-rollback |
| Latency P99 | > 1s | > 2s | Auto-rollback |

### 4.3 Drift Detection

- Feature drift checked hourly
- Concept drift checked daily
- Threshold: PSI > 0.2 triggers alert
- Action: Initiate retraining evaluation

---

## 5. RETRAINING POLICY

### 5.1 Triggers

**Automatic Triggers:**
- Feature drift PSI > 0.2 for 3 consecutive days
- Sharpe ratio < 0.5 for 7 consecutive days
- Model age > 90 days

**Manual Triggers:**
- Market regime change
- New data sources available
- Performance degradation investigation

### 5.2 Retraining Process

1. Automated trigger or manual request
2. Data preparation (last 6 months)
3. Model training (L3 DAG)
4. Automatic backtest validation
5. Comparison vs. current model
6. If better: Promote to Staging
7. If worse: Discard, alert team

### 5.3 Constraints

- Maximum 1 retraining per week (automatic)
- Cooldown: 48 hours between manual retrainings
- Production model continues during retraining

---

## 6. RETIREMENT POLICY

### 6.1 Criteria for Retirement

A model should be retired when:
- Replaced by a better-performing model
- Fundamental strategy change
- Regulatory requirement
- Persistent underperformance (>30 days)

### 6.2 Retirement Process

1. Model Owner documents retirement reason
2. Model transitioned to Archived status
3. Artifacts retained for 2 years minimum
4. Notification sent to team

### 6.3 Retention

- Model artifacts: 2 years
- Training data: 7 years (regulatory)
- Trade history: 7 years (regulatory)
- Audit logs: 7 years (regulatory)

---

## 7. AUDIT AND COMPLIANCE

### 7.1 Audit Trail

All model actions logged:
- Promotions (who, when, reason)
- Rollbacks (who, when, reason)
- Parameter changes
- Retraining events

### 7.2 Quarterly Review

Every quarter:
- Review all Production models
- Validate monitoring effectiveness
- Update governance policy if needed
- Document review in meeting notes

### 7.3 Annual Audit

Annually:
- Full model inventory review
- Compliance check against this policy
- Policy update if required
- Training for new team members

---

## 8. EXCEPTIONS

Exceptions to this policy require:
- Written justification
- CTO approval
- Time-limited (max 30 days)
- Post-exception review

---

## APPENDIX A: Model Card Template

```yaml
model_id: ppo_v20_20260115
version: 20
owner: trading_team
backup_owner: ml_team
created_date: 2026-01-15
promoted_to_production: 2026-01-22

training:
  dataset_hash: abc123...
  dataset_period: 2024-01-01 to 2025-12-31
  norm_stats_hash: def456...
  mlflow_run_id: run_xyz

performance:
  backtest_sharpe: 1.85
  backtest_win_rate: 0.54
  backtest_max_drawdown: -0.08
  staging_sharpe: 1.62
  staging_win_rate: 0.52

risks:
  - High volatility regime sensitivity
  - Dependent on DXY correlation

mitigations:
  - Circuit breaker at 5% daily DD
  - Kill switch accessible
```

---

*Document Owner: Trading Operations Team*
*Last Review: 2026-01-17*
*Next Review: 2026-04-17*
```

---

## 2.3 INCIDENT RESPONSE DOCUMENTATION [P1-HIGH]

**Archivo**: `docs/INCIDENT_RESPONSE_PLAYBOOK.md` (NUEVO)

```markdown
# INCIDENT RESPONSE PLAYBOOK
## USD/COP Trading System

**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## 1. INCIDENT SEVERITY LEVELS

| Level | Name | Criteria | Response Time | Escalation |
|-------|------|----------|---------------|------------|
| **P0** | Critical | System down, active losses | Immediate (24/7) | CTO within 15 min |
| **P1** | High | Major feature broken, degraded | 15 minutes | Eng Lead within 30 min |
| **P2** | Medium | Partial feature broken | 1 hour | On-call within 2 hours |
| **P3** | Low | Minor bug, no trading impact | Next business day | Team lead next standup |

---

## 2. ESCALATION CONTACTS

| Role | Name | Phone | Email | Hours | Slack |
|------|------|-------|-------|-------|-------|
| On-Call Primary | [Name] | [Phone] | [Email] | 24/7 | @oncall-primary |
| On-Call Secondary | [Name] | [Phone] | [Email] | 24/7 | @oncall-secondary |
| Engineering Lead | [Name] | [Phone] | [Email] | 9-18 COT | @eng-lead |
| Trading Lead | [Name] | [Phone] | [Email] | 8-17 COT | @trading-lead |
| CTO | [Name] | [Phone] | [Email] | Emergency | @cto |

---

## 3. INCIDENT PROCEDURES

### 3.1 P0: Trading System Down

**Symptoms:**
- No trades executing
- API returning 5xx errors
- Dashboard shows "System Down"
- Kill switch triggered unexpectedly

**Immediate Actions (0-5 min):**
1. Acknowledge incident in #trading-incidents
2. Check system status: `curl http://trading-api:8000/api/v1/health`
3. Check kill switch status: `curl http://trading-api:8000/api/v1/operations/status`
4. If kill switch active unintentionally → Resume with confirmation

**Diagnosis (5-15 min):**
```bash
# Check service logs
docker logs trading-api --tail 100
docker logs inference-api --tail 100

# Check database
psql -c "SELECT * FROM health_check"

# Check model status
curl http://inference-api:8000/api/v1/models
```

**Resolution:**
- If API down → Restart: `docker-compose restart trading-api`
- If model error → Rollback: Use dashboard or CLI
- If database down → Failover to replica

**Escalation:**
- 15 min without resolution → Engineering Lead
- 30 min without resolution → CTO

---

### 3.2 P1: Model Performance Degradation

**Symptoms:**
- Sharpe ratio dropped significantly
- Unusual loss streak
- High error rate in predictions
- Drift alerts triggered

**Immediate Actions:**
1. Check drift dashboard in Grafana
2. Compare current vs historical metrics
3. Review recent trades for anomalies

**Diagnosis:**
```bash
# Check model metrics
curl http://inference-api:8000/api/v1/models/router/status

# Check feature quality
curl http://inference-api:8000/api/v1/health/consistency/ppo_primary

# Review recent trades
SELECT * FROM trades_history
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;
```

**Resolution Options:**
1. **Minor degradation:** Monitor closely, no action
2. **Significant degradation:** Activate shadow mode comparison
3. **Severe degradation:** Rollback to previous model

---

### 3.3 P1: Data Source Down (TwelveData)

**Symptoms:**
- OHLCV data stale (>10 min old)
- Macro indicators not updating
- API timeout errors in logs

**Immediate Actions:**
1. Check TwelveData status page
2. Verify API key validity
3. Check rate limits

**Mitigation:**
```bash
# Check last data timestamp
SELECT MAX(timestamp) FROM ohlcv_usdcop_5m;

# If stale, trading should auto-pause
# Verify pause active
curl http://trading-api:8000/api/v1/operations/status
```

**Resolution:**
- Wait for TwelveData recovery
- If prolonged (>1 hour): Manual kill switch
- If critical: Switch to backup data source (if available)

---

## 4. POST-MORTEM TEMPLATE

```markdown
# Post-Mortem: [Incident Title]

**Date:** YYYY-MM-DD
**Duration:** HH:MM - HH:MM (X hours Y minutes)
**Severity:** P0/P1/P2/P3
**Author:** [Name]
**Reviewers:** [Names]

## Summary
Brief description of what happened.

## Impact
- Trading paused for X minutes
- Y trades missed
- Z losses incurred
- N users affected

## Timeline (all times in COT)
- HH:MM - First alert triggered
- HH:MM - On-call acknowledged
- HH:MM - Root cause identified
- HH:MM - Mitigation applied
- HH:MM - Service restored
- HH:MM - Incident closed

## Root Cause
Detailed explanation of what went wrong.

## Contributing Factors
- Factor 1
- Factor 2

## Resolution
What was done to fix the issue.

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Implement monitoring for X | [Name] | YYYY-MM-DD | Open |
| Update runbook for Y | [Name] | YYYY-MM-DD | Open |

## Lessons Learned
- What went well
- What could be improved

## Appendix
- Relevant logs
- Screenshots
- Metrics
```

---

## 5. COMMUNICATION TEMPLATES

### 5.1 Incident Start
```
🚨 INCIDENT: [Title]
Severity: P[X]
Status: Investigating
Impact: [Brief impact]
Updates: #trading-incidents
```

### 5.2 Incident Update
```
📢 UPDATE: [Title]
Status: [Investigating/Mitigating/Resolved]
Progress: [What's happening]
ETA: [If known]
```

### 5.3 Incident Resolved
```
✅ RESOLVED: [Title]
Duration: [X hours Y minutes]
Root Cause: [Brief]
Post-mortem: [Link when available]
```

---

## 6. RUNBOOK QUICK REFERENCE

| Scenario | Command/Action |
|----------|----------------|
| Kill all trading | Dashboard: Click Kill Switch OR `POST /operations/kill-switch` |
| Rollback model | Dashboard: Models → Rollback OR `scripts/promote_model.py demote` |
| Restart trading API | `docker-compose restart trading-api` |
| Check system health | `curl http://trading-api:8000/api/v1/health` |
| View recent errors | `docker logs trading-api --tail 200 \| grep ERROR` |
| Force model reload | `POST /api/v1/models/reload` |
| Check database | `psql -c "SELECT 1"` |
| Check Redis | `redis-cli ping` |

---

*Document maintained by: Trading Operations*
*Emergency contact: #trading-incidents*
```

---

# FASE 3: OBSERVABILITY (Semana 4)

## 3.1 DATA LINEAGE VISUALIZATION [P2-MEDIUM]

**Archivo**: `usdcop-trading-dashboard/components/lineage/LineageGraph.tsx` (NUEVO)

```typescript
// Implementar visualización de lineage usando React Flow o similar
// Mostrar: Data → Features → Model → Trades
```

## 3.2 INCIDENT DASHBOARD [P2-MEDIUM]

**Archivo**: `usdcop-trading-dashboard/app/incidents/page.tsx` (NUEVO)

```typescript
// Dashboard de incidentes activos y históricos
// Integración con sistema de alertas
```

## 3.3 AUTOMATED REPORTS [P2-MEDIUM]

**Archivo**: `airflow/dags/l6_daily_reports.py` (NUEVO)

```python
# DAG para generar reportes diarios y semanales
# Enviar por email automáticamente
```

---

# FASE 4: COMPLIANCE (Semanas 5-6)

## 4.1 MODEL CARDS [P3-LOW]

Generar model cards para cada modelo en producción siguiendo el template en governance policy.

## 4.2 CHANGELOG [P3-LOW]

**Archivo**: `CHANGELOG.md` (NUEVO)

Documentar todos los cambios significativos siguiendo formato Keep a Changelog.

## 4.3 AUDIT EXPORT [P3-LOW]

Implementar endpoint para exportar audit trail en formato regulatorio.

---

# RESUMEN DE ENTREGABLES

## Fase 1 (Semana 1)
| Entregable | Archivo | Status |
|------------|---------|--------|
| Kill Switch API | `services/inference_api/routers/operations.py` | 🔲 |
| Kill Switch UI | `components/operations/KillSwitch.tsx` | 🔲 |
| Rollback API | `services/inference_api/routers/models.py` | 🔲 |
| Rollback UI | `components/models/RollbackPanel.tsx` | 🔲 |
| Auto-Rollback Service | `services/inference_api/services/auto_rollback.py` | 🔲 |
| Notification Service | `services/shared/notifications/notifier.py` | 🔲 |

## Fase 2 (Semanas 2-3)
| Entregable | Archivo | Status |
|------------|---------|--------|
| Promote Button UI | `components/models/PromoteButton.tsx` | 🔲 |
| Governance Policy | `docs/MODEL_GOVERNANCE_POLICY.md` | 🔲 |
| Incident Playbook | `docs/INCIDENT_RESPONSE_PLAYBOOK.md` | 🔲 |
| Post-Mortem Template | `docs/templates/POST_MORTEM.md` | 🔲 |
| Escalation Contacts | Update `docs/RUNBOOK.md` | 🔲 |

## Fase 3 (Semana 4)
| Entregable | Archivo | Status |
|------------|---------|--------|
| Lineage Graph | `components/lineage/LineageGraph.tsx` | 🔲 |
| Incidents Dashboard | `app/incidents/page.tsx` | 🔲 |
| Daily Report DAG | `airflow/dags/l6_daily_reports.py` | 🔲 |

## Fase 4 (Semanas 5-6)
| Entregable | Archivo | Status |
|------------|---------|--------|
| Model Cards | Per-model documentation | 🔲 |
| Changelog | `CHANGELOG.md` | 🔲 |
| Audit Export API | `services/inference_api/routers/audit.py` | 🔲 |

---

# MÉTRICAS DE ÉXITO

## Criterios de Aceptación del Proyecto

| Métrica | Objetivo | Medición |
|---------|----------|----------|
| Score Auditoría | ≥85% (170/200) | Re-auditoría completa |
| Kill Switch Response | <10 segundos | Test de stress |
| Rollback Time | <60 segundos | Test en staging |
| Notification Latency | <30 segundos | Monitoreo |
| Uptime Dashboard | 99.9% | Prometheus |

## KPIs Post-Implementación

| KPI | Target | Frecuencia |
|-----|--------|------------|
| Mean Time to Detect (MTTD) | <5 min | Mensual |
| Mean Time to Resolve (MTTR) | <30 min | Mensual |
| Change Failure Rate | <5% | Semanal |
| Deployment Frequency | ≥1/week | Semanal |

---

**Documento preparado por**: Trading Operations Team
**Fecha**: 2026-01-17
**Próxima revisión**: 2026-02-17
