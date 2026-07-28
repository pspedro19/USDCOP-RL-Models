/**
 * POST /api/production/approve — Voto 2/2 humano (aprobar o rechazar la promoción).
 *
 * File-based: lee + escribe `<repo>/data/approvals/approval_state*.json` (CXD-057 — el
 * artefacto salió de `public/` porque lleva gates / DSR / métricas de backtest, que el
 * SSOT reserva a `research:read`). Store SSOT: `lib/approvals/store.ts`.
 *
 * Reglas que gobiernan esta ruta (`.claude/rules/approval-gates.md`):
 *   §1 dos votos siempre · §2 se vota sobre los números del bundle publicado ·
 *   §3 los botones viven solo en `/dashboard` · §4 **el servidor re-valida; la UI no es
 *   la autoridad** · §5 **solo `admin` vota, y queda en `audit_log`** · §6 estados.
 *
 * Dos correcciones P0 (hallazgo CODEX, tests
 * `tests/unit/api/approval-vote2-authz-cas.test.ts`):
 *
 *  1. **Autorización en el handler.** Antes llamaba a `protectApiRoute(request)` a
 *     secas: solo SESIÓN. El middleware exigía `approval:vote`, pero §4 dice que la
 *     autoridad es el servidor en cada capa — un `subscriber` autenticado que llegara
 *     al handler sin atravesar el edge emitía el Voto 2. Ahora el permiso se exige aquí
 *     (`requireApprovalVote`), con la sesión Y el set efectivo sellado por el middleware.
 *
 *  2. **Transición compare-and-set.** Antes: leer PENDING → mutar en memoria →
 *     `fs.writeFile`. Dos peticiones simultáneas ⇒ dos 200 contradictorios, y el APPROVE
 *     perdedor ya había disparado el deploy. Ahora `commitApprovalTransition` releé bajo
 *     lock interproceso (`O_EXCL`), verifica la precondición y publica con rename
 *     atómico: exactamente un 200, el otro 409, y **el deploy solo se dispara después
 *     del commit ganador**.
 *
 * Body: { action: 'APPROVE' | 'REJECT', notes?: string, reviewer?: string }
 */
import { NextRequest, NextResponse } from 'next/server';
import { requireApprovalVote } from '@/lib/auth/approval-authz';
import { commitApprovalTransition, readApprovalState } from '@/lib/approvals/store';
import { query } from '@/lib/db/postgres-client';
import type { ApprovalState, ApproveRequest, ApproveResponse } from '@/lib/contracts/production-approval.contract';

/**
 * Origen del auto-deploy interno (llamada del servidor a sí mismo).
 *
 * **Nunca `request.nextUrl.origin`**: ese valor sale de la cabecera `Host` (o
 * `x-forwarded-host`), que la controla el CLIENTE — y a ese origen se le reenvía la
 * cookie de sesión del admin aprobador. `Host: evil.tld` bastaba para exfiltrar la
 * sesión que puede promover a producción. El destino interno es configuración del
 * servidor, no dato de la petición.
 *
 * **Tampoco `NEXTAUTH_URL`**: es la URL de cara al NAVEGADOR (`http://localhost:5000`
 * en compose) mientras el contenedor escucha en `:3000` — el mapeo de puertos solo
 * existe fuera. Por eso el default es loopback + el puerto REAL de escucha, y así este
 * fix además repara el auto-deploy en contenedor, que con el origen de la petición
 * apuntaba a `localhost:5000` desde dentro y moría en el `.catch()` silencioso.
 */
function internalOrigin(): string {
  const configured = (process.env.INTERNAL_BASE_URL || '').trim();
  if (configured) {
    try {
      return new URL(configured).origin;
    } catch {
      /* configuración inválida ⇒ loopback */
    }
  }
  return `http://127.0.0.1:${process.env.PORT || '3000'}`;
}

/** Fila append-only del Voto 2. Best-effort en DB; el rastro DURABLE va en el artefacto. */
async function auditVote(
  userId: string | null,
  action: 'APPROVE' | 'REJECT',
  strategy: string,
  detail: Record<string, unknown>,
  req: NextRequest,
): Promise<void> {
  try {
    await query(
      `INSERT INTO audit_log (user_id, action, object_type, object_id, detail, ip)
       VALUES ($1, $2, 'approval', $3, $4::jsonb, $5)`,
      [
        userId,
        action === 'APPROVE' ? 'approval_vote2_approve' : 'approval_vote2_reject',
        strategy,
        JSON.stringify({ ...detail, via: '/api/production/approve' }),
        req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() ?? null,
      ],
    );
  } catch (e) {
    // La DB puede no estar; el cambio de estado NO queda sin auditar porque
    // `audit_trail` viaja dentro del mismo commit atómico del artefacto.
    console.error('[Approve] audit_log row failed (file audit_trail still recorded):', e);
  }
}

export async function POST(request: NextRequest) {
  try {
    // ── (1) Autorización server-side EN EL HANDLER (approval-gates §4/§5).
    const gate = await requireApprovalVote(request);
    if (!gate.ok) {
      return NextResponse.json(
        { success: false, status: 'PENDING_APPROVAL', message: gate.message } as ApproveResponse,
        { status: gate.status },
      );
    }

    const body: ApproveRequest & { strategy_id?: string } = await request.json();

    if (!body.action || !['APPROVE', 'REJECT'].includes(body.action)) {
      return NextResponse.json(
        { success: false, status: 'PENDING_APPROVAL', message: 'Invalid action. Must be APPROVE or REJECT.' } as ApproveResponse,
        { status: 400 }
      );
    }
    const action = body.action;

    // ── (2) Resolución del artefacto. El fallback per-strategy → singleton es
    // load-bearing y compartido verbatim con el DAG H5-L4b: deben coincidir o el
    // Voto 2 aprueba y el deploy 404ea.
    const record = await readApprovalState(body.strategy_id ?? null);
    if (!record) {
      return NextResponse.json(
        { success: false, status: 'PENDING_APPROVAL', message: 'No approval state file found. Run backtest first.' } as ApproveResponse,
        { status: 404 }
      );
    }

    // Chequeo temprano SOLO para el mensaje de usuario: la decisión autoritativa la
    // toma la precondición del CAS con el estado releído bajo lock.
    if (record.state.status !== 'PENDING_APPROVAL') {
      return NextResponse.json(
        { success: false, status: record.state.status, message: `Cannot ${action.toLowerCase()} — current status is ${record.state.status}` } as ApproveResponse,
        { status: 409 }
      );
    }

    const now = new Date().toISOString();
    const reviewer = gate.reviewer; // principal AUTENTICADO, nunca `body.reviewer`

    // ── (3) COMPARE-AND-SET. Un ganador escribe; el resto ve el estado ya resuelto.
    const outcome = await commitApprovalTransition(
      record.file,
      (current) =>
        current.status === 'PENDING_APPROVAL'
          ? null
          : `Cannot ${action.toLowerCase()} — current status is ${current.status}`,
      (current) => {
        const next: ApprovalState = { ...current };
        if (action === 'APPROVE') {
          next.status = 'APPROVED';
          next.approved_by = reviewer;
          next.approved_at = now;
          next.reviewer_notes = body.notes || '';
        } else {
          next.status = 'REJECTED';
          next.rejected_by = reviewer;
          next.rejected_at = now;
          next.rejection_reason = body.notes || '';
        }
        next.last_updated = now;
        // Rastro append-only dentro del propio commit atómico: el cambio de estado
        // NUNCA queda sin auditar aunque Postgres esté caído (approval-gates §5).
        next.audit_trail = [
          ...(current.audit_trail ?? []),
          {
            at: now,
            actor: reviewer,
            role: gate.role,
            action,
            from: current.status,
            to: action === 'APPROVE' ? 'APPROVED' : 'REJECTED',
            notes: body.notes || '',
          },
        ];
        return next;
      },
    );

    if (!outcome.ok) {
      const status = outcome.code === 'MISSING' ? 404 : 409;
      return NextResponse.json(
        { success: false, status: outcome.status ?? 'PENDING_APPROVAL', message: outcome.message } as ApproveResponse,
        { status }
      );
    }

    // ── (4) Auditoría en DB (mirror best-effort del rastro ya comprometido en disco).
    await auditVote(gate.userId, action, outcome.state.strategy, { notes: body.notes || '', file: record.repoPath }, request);

    const response: ApproveResponse = {
      success: true,
      status: outcome.state.status,
      message: action === 'APPROVE'
        ? `Strategy approved by ${reviewer}. Deploy starting automatically...`
        : `Strategy rejected by ${reviewer}. Daily runner will remain in dry-run mode.`,
    };

    // ── (5) Efecto: SOLO después del commit ganador (antes se disparaba aunque otra
    // petición acabara escribiendo REJECTED encima).
    if (action === 'APPROVE') {
      try {
        fetch(`${internalOrigin()}/api/production/deploy`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            // La sesión del aprobador se reenvía al origen INTERNO configurado y a
            // ningún otro; el deploy vuelve a validar `approval:vote` y APPROVED.
            cookie: request.headers.get('cookie') ?? '',
          },
          body: JSON.stringify({ strategy_id: body.strategy_id ?? null }),
        }).catch(() => {
          // Non-blocking: deploy failure doesn't affect approval
        });
      } catch {
        // Non-blocking: if fetch itself throws, approval still succeeds
      }
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error('Error processing approval:', error);
    return NextResponse.json(
      { success: false, status: 'PENDING_APPROVAL', message: 'Internal server error' } as ApproveResponse,
      { status: 500 }
    );
  }
}
