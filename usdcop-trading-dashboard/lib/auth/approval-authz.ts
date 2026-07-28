/**
 * Autorización del Voto 2/2 — DENTRO del handler, no solo en el edge.
 * ====================================================================
 *
 * `.claude/rules/approval-gates.md`:
 *   §4 «El deploy re-valida server-side. La UI no es la autoridad.»
 *   §5 «Solo `admin` puede emitir Vote 2, promover o accionar el kill global.»
 *
 * El middleware ya gatea `/api/production/approve|deploy` con `approval:vote`
 * (`rbac.contract.ts` API_MATRIX), pero un handler que solo comprueba SESIÓN queda a
 * merced de cualquier invocación que no atraviese el edge (rewrite interno, un runtime
 * que no ejecute middleware para esa ruta, un `import` desde otra ruta, un test que se
 * creyó verde). Este módulo es la SEGUNDA barrera, y es la que decide.
 *
 * Fuentes de verdad, en este orden y en CONJUNCIÓN (ambas deben permitir):
 *
 *  1. **La sesión** (`protectApiRoute` → `getServerSession`). El rol viene de
 *     `sb_users.role`, no del cliente. Un rol desconocido/ausente ⇒ DENEGADO
 *     (fail-closed: nunca "por defecto admin").
 *  2. **El set EFECTIVO sellado por el middleware** (`x-user-perms`, migración 056),
 *     *cuando está presente*. El middleware borra las cabeceras entrantes antes de
 *     sellarlas, así que no es falsificable; y el "Ver como" solo REBAJA
 *     (`intersectPerms`). Exigirlo también impide que un admin previsualizando como
 *     `subscriber` emita el voto. Ausente ⇒ no suma permiso, solo la sesión decide.
 *
 * La intersección es deliberada: cada fuente puede DENEGAR, ninguna puede CONCEDER sola.
 */
import type { NextRequest } from 'next/server';

import { protectApiRoute } from '@/lib/auth/api-auth';
import {
  isRole,
  permsHave,
  roleHasPermission,
  type Permission,
} from '@/lib/contracts/rbac.contract';

export interface ApprovalPrincipal {
  ok: true;
  /** Identidad AUTENTICADA que queda en `audit_log` — jamás un nombre del cuerpo. */
  reviewer: string;
  userId: string | null;
  role: string;
}

export interface ApprovalDenied {
  ok: false;
  status: number;
  message: string;
}

/**
 * Exige `approval:vote` sobre la petición. Devuelve el principal o la denegación
 * con el código HTTP que corresponde (401 sin sesión, 403 sin permiso, 429 rate limit).
 */
export async function requirePermissionInHandler(
  request: NextRequest,
  permission: Permission,
): Promise<ApprovalPrincipal | ApprovalDenied> {
  const auth = await protectApiRoute(request);
  if (!auth.authenticated || !auth.user) {
    return { ok: false, status: auth.status || 401, message: auth.error || 'Unauthorized' };
  }

  const role = auth.user.role as string | undefined;

  // (1) La sesión: rol conocido del contrato RBAC + permiso en la matriz.
  if (!isRole(role) || !roleHasPermission(role, permission)) {
    return {
      ok: false,
      status: 403,
      message: `Permission denied: ${permission} is required (role: ${role ?? 'unknown'}).`,
    };
  }

  // (2) El set efectivo sellado por el middleware, si viajó con la petición.
  const stamped = request.headers.get('x-user-perms');
  if (stamped !== null && !permsHave(stamped.split(',').filter(Boolean), permission)) {
    return {
      ok: false,
      status: 403,
      message: `Permission denied: ${permission} is not in the effective permission set.`,
    };
  }

  return {
    ok: true,
    reviewer: auth.user.email || auth.user.username || auth.user.id || 'operator',
    userId: auth.user.id ?? null,
    role,
  };
}

/** Azúcar para las dos rutas del Voto 2 (approve + deploy). */
export const requireApprovalVote = (request: NextRequest) =>
  requirePermissionInHandler(request, 'approval:vote');
