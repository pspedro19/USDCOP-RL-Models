#!/usr/bin/env node
/**
 * RBAC coverage gate (CTR-RBAC-001, R2) — deny-by-default is only real if the matrix is
 * EXHAUSTIVE. This script enumerates every actual route in `app/**` and fails (exit 1)
 * if any API route has no matching prefix in `API_ROUTES` or any page has none in
 * `PAGE_ROUTES`. Wire into CI (and `npm run rbac:check`).
 *
 * Zero deps: extracts prefixes from lib/contracts/rbac.contract.ts textually so it runs
 * without a TS build step.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const CONTRACT = readFileSync(join(ROOT, 'lib', 'contracts', 'rbac.contract.ts'), 'utf8');

function extractPrefixes(blockName) {
  const m = CONTRACT.match(new RegExp(`${blockName}[^=]*=\\s*\\[([\\s\\S]*?)\\]\\s*as const`));
  if (!m) throw new Error(`cannot find ${blockName} in rbac.contract.ts`);
  return [...m[1].matchAll(/prefix:\s*'([^']+)'/g)].map((x) => x[1]);
}

const apiPrefixes = extractPrefixes('API_ROUTES');
const pagePrefixes = extractPrefixes('PAGE_ROUTES');

function walk(dir, out = []) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) walk(p, out);
    else out.push(p);
  }
  return out;
}

function routeOf(file, kind) {
  const rel = file.slice(join(ROOT, 'app').length).split(sep).join('/');
  const route = rel
    .replace(new RegExp(`/${kind}\\.(ts|tsx|js|jsx)$`), '')
    .replace(/\/\([^)]+\)/g, '') // route groups
    .replace(/\/@[^/]+/g, '');   // parallel routes
  return route === '' ? '/' : route;
}

const files = walk(join(ROOT, 'app'));
const apiRoutes = files.filter((f) => /route\.(ts|js)$/.test(f) && f.includes(`${sep}api${sep}`))
  .map((f) => routeOf(f, 'route'));
const pageRoutes = files.filter((f) => /page\.(tsx|jsx|ts|js)$/.test(f))
  .map((f) => routeOf(f, 'page'));

const covered = (route, prefixes) =>
  route === '/' || prefixes.some((p) => p !== '/' && route.startsWith(p));

const missApi = [...new Set(apiRoutes.filter((r) => !covered(r, apiPrefixes)))];
const missPage = [...new Set(pageRoutes.filter((r) => !covered(r, pagePrefixes)))];

console.log(`RBAC coverage: ${apiRoutes.length} API routes, ${pageRoutes.length} pages`);
if (missApi.length || missPage.length) {
  missApi.forEach((r) => console.error(`  UNCOVERED API : ${r}`));
  missPage.forEach((r) => console.error(`  UNCOVERED PAGE: ${r}`));
  console.error('FAIL: add the routes above to lib/contracts/rbac.contract.ts (deny-by-default)');
  process.exit(1);
}
console.log('OK: every route has a permission entry');
