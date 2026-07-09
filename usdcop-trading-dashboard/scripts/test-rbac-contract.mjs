#!/usr/bin/env node
/**
 * RBAC contract semantics test (pure functions, no framework).
 * Locks: role→permission matrix, route resolution, subscriber relabel, expiry degradation.
 * Run: `node --experimental-strip-types scripts/test-rbac-contract.mjs` (exit 1 on failure).
 * (Node ≥22.6 strips TS types natively — the contract is dependency-free TS.)
 */
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');

let failures = 0;
const t = (name, cond) => {
  if (!cond) { failures += 1; console.error(`FAIL: ${name}`); }
  else console.log(`ok: ${name}`);
};

{
  const c = await import(pathToFileURL(join(ROOT, 'lib', 'contracts', 'rbac.contract.ts')).href);

  // role → permission
  t('admin has approval:vote', c.roleHasPermission('admin', 'approval:vote'));
  t('developer NEVER approval:vote', !c.roleHasPermission('developer', 'approval:vote'));
  t('developer has research:read', c.roleHasPermission('developer', 'research:read'));
  t('subscriber NEVER research:read', !c.roleHasPermission('subscriber', 'research:read'));
  t('subscriber has execution:self', c.roleHasPermission('subscriber', 'execution:self'));
  t('subscriber NEVER execution:global', !c.roleHasPermission('subscriber', 'execution:global'));
  t('free only delayed content', !c.roleHasPermission('free', 'signals:read')
    && c.roleHasPermission('free', 'forecast:read'));
  t('unknown role has nothing', !c.roleHasPermission('hacker', 'forecast:read'));

  // route resolution (deny-by-default + longest prefix)
  t('promote is admin-only perm', c.requiredPermissionFor('/api/registry/promote', c.API_ROUTES) === 'approval:vote');
  t('registry read is research', c.requiredPermissionFor('/api/registry', c.API_ROUTES) === 'research:read');
  t('signalbridge system is global', c.requiredPermissionFor('/api/signalbridge/system/kill', c.API_ROUTES) === 'execution:global');
  t('unknown api → null (deny floor)', c.requiredPermissionFor('/api/whatever-new', c.API_ROUTES) === null);
  t('landing is public', c.requiredPermissionFor('/', c.PAGE_ROUTES) === 'public');
  t('dashboard needs research', c.requiredPermissionFor('/dashboard', c.PAGE_ROUTES) === 'research:read');

  // nav
  const subNav = c.navFor('subscriber');
  t('subscriber nav has Señales, no Backtest',
    subNav.some((e) => e.label === 'Señales') && !subNav.some((e) => e.href === '/dashboard'));
  const devNav = c.navFor('developer');
  t('developer nav keeps Producción label', devNav.some((e) => e.label === 'Producción'));

  // entitlements
  const expired = { ...c.PLAN_DEFAULTS.auto, expires_at: '2020-01-01T00:00:00Z' };
  t('expired plan degrades to free', c.effectiveEntitlements(expired).plan === 'free');
  t('null entitlements = free', c.effectiveEntitlements(null).plan === 'free');
  const partial = { plan: 'free' };
  const eff = c.effectiveEntitlements(partial);
  t('partial row merges defaults (assets defined)', Array.isArray(eff.assets) && eff.analysis_delay_days === 7);
}

if (failures) { console.error(`${failures} failure(s)`); process.exit(1); }
console.log('ALL RBAC CONTRACT TESTS PASS');
