'use client';

/**
 * Sparkline — named export of the GM sparkline (VISUAL-SPEC-CHECKLIST / plan F2).
 *
 * The implementation lives in `./Spark` (prototype Var B `spark()`, verbatim: SVG
 * 56×22 / 84×34, `--gm-*` colors, real-series mode that never invents data). This file
 * is the canonical name the design checklist references; it re-exports so both
 * `<Spark/>` (existing call sites) and `<Sparkline/>` resolve to one implementation.
 */
export { Spark, Spark as Sparkline } from './Spark';
export type { SparkProps, SparkProps as SparklineProps, SparkTone, SparkSize } from './Spark';
