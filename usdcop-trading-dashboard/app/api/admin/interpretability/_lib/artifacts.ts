/**
 * Acceso seguro a los artefactos de interpretabilidad (C-006/BL-20-UI · CXD-040).
 *
 * Los artefactos viven FUERA de public/ — en `<repo>/data/interpretability/
 * <surface>/<asset>/<model_id>/<version>/summary.json` — para que la ÚNICA vía de
 * acceso sea la API con `requirePermission('admin:all')` (los estáticos de public/
 * solo exigían sesión en el middleware ⇒ bypass del gate admin).
 *
 * Defensas (fail-closed en todas):
 *  - Segmentos whitelisteados (SEG) + rechazo de `..`/absolutos ANTES de resolver.
 *  - Resolución con `path.resolve` + `fs.realpath` del base Y del candidato:
 *    el realpath del fichero debe quedar bajo el realpath del base (bloquea
 *    traversal codificado, absolutos y symlinks que apunten fuera del base).
 *  - Size-cap (2MB) por `stat` ANTES de leer.
 *  - Validación runtime contra el JSON Schema COMPARTIDO
 *    (`../_schema/interp-summary.schema.json`, el mismo que valida el test Python
 *    del generador): campos conocidos, unknown-field STRIP, números finitos.
 *  - Cualquier error ⇒ `null`; el caller responde SIEMPRE con mensaje genérico.
 */
import fs from 'fs/promises';
import path from 'path';

import { validateAndStrip } from './artifact-schema';

/** Cap de tamaño del artefacto servible (bytes). */
export const MAX_ARTIFACT_BYTES = 2 * 1024 * 1024;

/** Segmentos de path válidos — bloquea traversal y nombres raros de raíz. */
export const SEG = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/**
 * Raíz de artefactos. Default: `<repo>/data/interpretability` (cwd del dashboard es
 * `usdcop-trading-dashboard/`). Override por env para contenedores/tests:
 * `INTERPRETABILITY_DATA_DIR`.
 */
export function artifactsRoot(): string {
  const env = process.env.INTERPRETABILITY_DATA_DIR;
  if (env && env.trim()) return path.resolve(env.trim());
  return path.resolve(process.cwd(), '..', 'data', 'interpretability');
}

/** ¿Los 4 segmentos pasan el whitelist? (rechaza `..`, absolutos, separadores). */
export function validSegments(parts: string[]): boolean {
  return (
    parts.length === 4 &&
    parts.every((p) => SEG.test(p) && !p.includes('..') && !path.isAbsolute(p))
  );
}

/**
 * Resuelve `<base>/<surface>/<asset>/<model_id>/<version>/summary.json` de forma
 * canónica (realpath) y verifica que el resultado REAL quede bajo el base REAL.
 * `null` ⇒ no existe o escapa del base (el caller responde 404 genérico).
 */
export async function resolveSummaryFile(parts: string[]): Promise<string | null> {
  if (!validSegments(parts)) return null;
  let realBase: string;
  try {
    realBase = await fs.realpath(artifactsRoot());
  } catch {
    return null; // sin directorio base → no hay artefactos
  }
  const candidate = path.resolve(realBase, ...parts, 'summary.json');
  const rel = path.relative(realBase, candidate);
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null;
  let realFile: string;
  try {
    realFile = await fs.realpath(candidate); // resuelve symlinks/junctions
  } catch {
    return null; // ENOENT / no accesible
  }
  if (!realFile.startsWith(realBase + path.sep)) return null; // symlink fuera del base
  return realFile;
}

/**
 * Lee y valida el summary.json YA resuelto: size-cap → parse → schema compartido
 * (strip de campos desconocidos, números finitos). `null` ⇒ artefacto inválido;
 * jamás propaga el error interno.
 */
export async function readValidatedSummary(file: string): Promise<unknown | null> {
  try {
    const st = await fs.stat(file);
    if (!st.isFile() || st.size > MAX_ARTIFACT_BYTES) return null;
    const parsed: unknown = JSON.parse(await fs.readFile(file, 'utf-8'));
    return validateAndStrip(parsed);
  } catch {
    return null;
  }
}

/** Subdirectorios cuyo nombre pasa el whitelist (para el índice). */
export async function safeSubdirs(p: string): Promise<string[]> {
  try {
    return (await fs.readdir(p, { withFileTypes: true }))
      .filter((d) => d.isDirectory() && SEG.test(d.name))
      .map((d) => d.name)
      .sort();
  } catch {
    return []; // sin directorio → índice vacío (degradación C5, no 500)
  }
}
