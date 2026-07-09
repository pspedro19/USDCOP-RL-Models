#!/usr/bin/env node
/**
 * R0 surface audit generator (CTR-RBAC-001) — writes docs/rbac/PHASE0-surface-audit.md:
 * every real route in app/** with its effective permission from rbac.contract.ts.
 * Companion to check-rbac-coverage.mjs (same parsing).
 */
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const CONTRACT = readFileSync(join(ROOT, 'lib', 'contracts', 'rbac.contract.ts'), 'utf8');

function rules(blockName) {
  const m = CONTRACT.match(new RegExp(`${blockName}[^=]*=\\s*\\[([\\s\\S]*?)\\]\\s*as const`));
  if (!m) throw new Error(`cannot find ${blockName}`);
  return [...m[1].matchAll(/prefix:\s*'([^']+)',\s*permission:\s*'([^']+)'/g)]
    .map((x) => ({ prefix: x[1], permission: x[2] }));
}

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
    .replace(/\/\([^)]+\)/g, '');
  return route === '' ? '/' : route;
}

const permOf = (route, ruleSet) => {
  if (route === '/') return 'public';
  const m = ruleSet
    .filter((r) => r.prefix !== '/' && route.startsWith(r.prefix))
    .sort((a, b) => b.prefix.length - a.prefix.length)[0];
  return m ? m.permission : 'authenticated (floor)';
};

const api = rules('API_ROUTES');
const pages = rules('PAGE_ROUTES');
const files = walk(join(ROOT, 'app'));
const apiRoutes = [...new Set(
  files.filter((f) => /route\.(ts|js)$/.test(f) && f.includes(`${sep}api${sep}`))
    .map((f) => routeOf(f, 'route')))].sort();
const pageRoutes = [...new Set(
  files.filter((f) => /page\.(tsx|jsx)$/.test(f)).map((f) => routeOf(f, 'page')))].sort();

let md = `# PHASE0 — Surface audit (generado ${new Date().toISOString().slice(0, 10)})\n\n`;
md += '> Cada ruta real de `app/**` y su permiso efectivo según `lib/contracts/rbac.contract.ts`\n';
md += '> (regenerar: `node scripts/generate-surface-audit.mjs`). Estáticos: `/data/**` y\n';
md += '> `/forecasting/**` exigen sesión en el edge (middleware); `public/` restante = branding.\n\n';
md += `## API routes (${apiRoutes.length})\n\n| Ruta | Permiso |\n|---|---|\n`;
for (const r of apiRoutes) md += `| \`${r}\` | \`${permOf(r, api)}\` |\n`;
md += `\n## Pages (${pageRoutes.length})\n\n| Página | Permiso |\n|---|---|\n`;
for (const r of pageRoutes) md += `| \`${r}\` | \`${permOf(r, pages)}\` |\n`;

const outDir = join(ROOT, '..', 'docs', 'rbac');
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, 'PHASE0-surface-audit.md'), md);
console.log(`written docs/rbac/PHASE0-surface-audit.md — ${apiRoutes.length} APIs, ${pageRoutes.length} pages`);
