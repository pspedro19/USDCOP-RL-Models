/**
 * Validador runtime de artefactos de interpretabilidad contra el JSON Schema
 * COMPARTIDO (`../_schema/interp-summary.schema.json`) — el MISMO archivo que valida
 * el test Python del generador (tests/test_interpretability_schema.py), así el
 * contrato productor↔consumidor no puede divergir (hallazgo CXD-040 #3).
 *
 * Semántica (fail-closed):
 *  - `required` + tipos + `const` se verifican estrictamente; cualquier violación ⇒ rechazo.
 *  - Objetos con `additionalProperties: false` ⇒ los campos DESCONOCIDOS se STRIPean
 *    (no se sirven al cliente); el lado Python valida estricto (rechaza extras) porque
 *    el productor no debe emitirlos.
 *  - Todo `number` debe ser FINITO (JSON.parse('1e999') === Infinity — JSON Schema no
 *    puede expresarlo; se impone aquí por código).
 *
 * Intérprete mínimo del subconjunto draft-07 usado por el schema ($ref a
 * #/definitions, oneOf, type, const, enum, properties/required/additionalProperties,
 * items).
 * Sin dependencias nuevas (ajv no está declarado en package.json).
 */
import rawSchema from '../_schema/interp-summary.schema.json';

type SchemaNode = {
  $ref?: string;
  oneOf?: SchemaNode[];
  type?: string | string[];
  const?: unknown;
  enum?: unknown[];
  properties?: Record<string, SchemaNode>;
  required?: string[];
  additionalProperties?: boolean | SchemaNode;
  items?: SchemaNode;
  definitions?: Record<string, SchemaNode>;
};

const SCHEMA = rawSchema as SchemaNode;

function deref(node: SchemaNode): SchemaNode {
  if (!node.$ref) return node;
  const m = /^#\/definitions\/([A-Za-z0-9_-]+)$/.exec(node.$ref);
  const target = m ? SCHEMA.definitions?.[m[1]] : undefined;
  if (!target) throw new Error(`unresolvable $ref: ${node.$ref}`);
  return deref(target);
}

function typeOf(v: unknown): string {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  return typeof v; // 'object' | 'number' | 'string' | 'boolean'
}

/** Valida `value` contra `node`; devuelve la copia stripeada o `null` si viola el schema. */
function walk(value: unknown, node: SchemaNode): unknown | null {
  const s = deref(node);

  if (s.oneOf) {
    for (const branch of s.oneOf) {
      const out = walk(value, branch);
      if (out !== null) return out;
    }
    return null;
  }

  if (s.const !== undefined) {
    return Object.is(value, s.const) ? value : null;
  }

  if (s.enum !== undefined) {
    return s.enum.some((e) => Object.is(value, e)) ? value : null;
  }

  const t = typeOf(value);
  if (s.type !== undefined) {
    const allowed = Array.isArray(s.type) ? s.type : [s.type];
    if (!allowed.includes(t)) return null;
  }

  if (t === 'number' && !Number.isFinite(value as number)) return null; // 1e999 ⇒ Infinity

  if (t === 'array') {
    if (!s.items) return value;
    const out: unknown[] = [];
    for (const item of value as unknown[]) {
      const v = walk(item, s.items);
      if (v === null) return null;
      out.push(v);
    }
    return out;
  }

  if (t === 'object') {
    const obj = value as Record<string, unknown>;
    for (const req of s.required ?? []) {
      if (!(req in obj)) return null;
    }
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      const propSchema = s.properties?.[k];
      if (propSchema) {
        const w = walk(v, propSchema);
        if (w === null && !(v === null && isNullAllowed(propSchema))) return null;
        out[k] = w === null ? null : w;
        continue;
      }
      if (s.additionalProperties === false || s.additionalProperties === undefined) {
        continue; // STRIP del campo desconocido — no se sirve al cliente
      }
      if (s.additionalProperties === true) {
        out[k] = v;
        continue;
      }
      const w = walk(v, s.additionalProperties);
      if (w === null && !(v === null && isNullAllowed(s.additionalProperties))) return null;
      out[k] = w === null ? null : w;
    }
    return out;
  }

  return value; // string | boolean | null ya tipados arriba
}

/** ¿El schema admite `null` como valor válido? (para no confundirlo con "rechazado"). */
function isNullAllowed(node: SchemaNode): boolean {
  const s = deref(node);
  if (s.const !== undefined) return s.const === null;
  if (s.oneOf) return s.oneOf.some(isNullAllowed);
  if (s.type === undefined) return true;
  const allowed = Array.isArray(s.type) ? s.type : [s.type];
  return allowed.includes('null');
}

/**
 * Valida un summary parseado contra el schema compartido.
 * Devuelve la copia validada y STRIPeada, o `null` (rechazo fail-closed).
 */
export function validateAndStrip(value: unknown): unknown | null {
  try {
    return walk(value, SCHEMA);
  } catch {
    return null;
  }
}
