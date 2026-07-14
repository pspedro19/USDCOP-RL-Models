/**
 * Dynamic RBAC resolver (CTR-RBAC-001 extension, migration 056).
 * ============================================================================
 * Role→permission and per-user grants/denies are DATA (`rbac_role_permissions`,
 * `rbac_user_overrides`) editable from the admin "Roles y vistas" console. This
 * module resolves them, with the STATIC `ROLE_PERMISSIONS` matrix as seed +
 * fallback so the system is always deny-by-default and never fails OPEN:
 *
 *   - table empty / DB down  ⇒ fall back to the compile-time matrix (fail to CODE)
 *   - unknown role           ⇒ empty set (deny-by-default preserved)
 *
 * `effectivePermissions()` is called by the token-mint paths (baked into the JWT
 * at login) and by the admin endpoints. A 60s in-process TTL cache mirrors
 * `entitlements.ts` — permission edits apply on the user's next login (JWT
 * re-mint) or within the cache TTL for server-side reads.
 */
import { query } from '@/lib/db/postgres-client';
import {
  PERMISSIONS,
  ROLE_PERMISSIONS,
  ROLES,
  type Permission,
  type Role,
} from '@/lib/contracts/rbac.contract';

const CACHE_TTL_MS = 60_000;
const PERM_SET = new Set<string>(PERMISSIONS);

function isPermission(p: string): p is Permission {
  return PERM_SET.has(p);
}

// ── role→permission map (dynamic, DB-backed, static fallback) ────────────────
let roleMapCache: { at: number; value: Record<Role, Permission[]> } | null = null;

/** Full role→permission matrix from the DB; static ROLE_PERMISSIONS on empty/error. */
export async function allRolePermissions(): Promise<Record<Role, Permission[]>> {
  if (roleMapCache && Date.now() - roleMapCache.at < CACHE_TTL_MS) return roleMapCache.value;

  const fallback = (): Record<Role, Permission[]> =>
    Object.fromEntries(ROLES.map((r) => [r, [...ROLE_PERMISSIONS[r]]])) as Record<Role, Permission[]>;

  try {
    const res = await query<{ role: string; permission: string }>(
      'SELECT role, permission FROM rbac_role_permissions',
    );
    if (res.rows.length === 0) return fallback();
    const map = Object.fromEntries(ROLES.map((r) => [r, [] as Permission[]])) as Record<Role, Permission[]>;
    for (const { role, permission } of res.rows) {
      if ((ROLES as readonly string[]).includes(role) && isPermission(permission)) {
        map[role as Role].push(permission);
      }
    }
    roleMapCache = { at: Date.now(), value: map };
    return map;
  } catch {
    return fallback(); // DB down ⇒ static matrix (fail to code, never open)
  }
}

/** Permissions for one role (dynamic, static fallback). Unknown role ⇒ []. */
export async function roleDynamicPermissions(role: Role | string | undefined): Promise<Permission[]> {
  if (!role || !(ROLES as readonly string[]).includes(role)) return [];
  const map = await allRolePermissions();
  return map[role as Role] ?? [];
}

// ── per-user overrides ───────────────────────────────────────────────────────
export interface UserOverride {
  permission: Permission;
  effect: 'grant' | 'deny';
}

/** Raw override rows for one user (admin editor). Empty on error. */
export async function getUserOverrides(userId: string): Promise<UserOverride[]> {
  try {
    const res = await query<{ permission: string; effect: string }>(
      'SELECT permission, effect FROM rbac_user_overrides WHERE user_id = $1',
      [userId],
    );
    return res.rows
      .filter((r) => isPermission(r.permission) && (r.effect === 'grant' || r.effect === 'deny'))
      .map((r) => ({ permission: r.permission as Permission, effect: r.effect as 'grant' | 'deny' }));
  } catch {
    return [];
  }
}

// ── effective permissions (role ∪ grants − denies) ───────────────────────────
const effCache = new Map<string, { at: number; value: Permission[] }>();

/**
 * Effective permission set for a user: dynamic role permissions, plus per-user
 * grants, minus per-user denies. This is what gets baked into the JWT at login
 * and stamped on `x-user-perms`. Deny-by-default: unknown role + no grants ⇒ [].
 */
export async function effectivePermissions(
  userId: string | null | undefined,
  role: Role | string | undefined,
): Promise<Permission[]> {
  const cacheKey = `${userId ?? 'anon'}|${role ?? 'none'}`;
  const hit = effCache.get(cacheKey);
  if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.value;

  const base = new Set<Permission>(await roleDynamicPermissions(role));
  if (userId) {
    for (const o of await getUserOverrides(userId)) {
      if (o.effect === 'grant') base.add(o.permission);
      else base.delete(o.permission);
    }
  }
  const value = [...base];
  effCache.set(cacheKey, { at: Date.now(), value });
  return value;
}

/** Invalidate caches after an admin edit so reads reflect it within the request. */
export function invalidateRbacCaches(): void {
  roleMapCache = null;
  effCache.clear();
}
