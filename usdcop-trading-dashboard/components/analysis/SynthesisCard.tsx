'use client';

import { motion } from 'framer-motion';
import { FileText } from 'lucide-react';
import { useGmT } from '@/lib/i18n/gm-core';
import { GM, GMT } from '@/lib/ui/gm-tokens';

import { ANALYSIS_DICT } from './gm-analysis';
import { AnalysisMarkdown } from './AnalysisMarkdown';

/**
 * Executive synthesis panel — renders the LangGraph `synthesis_markdown` (the ~3KB
 * LLM-authored weekly synthesis) via the shared AnalysisMarkdown renderer.
 *
 * USD/COP rich weeks only: Gold/BTC (and stale weeks) omit the field, so this
 * renders nothing when the field is missing or blank (never an empty box).
 */
export function SynthesisCard({ markdown }: { markdown?: string | null }) {
  const t = useGmT(ANALYSIS_DICT);

  if (typeof markdown !== 'string' || markdown.trim().length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${GM.panel} gm-contain p-6`}
    >
      <h2 className={`${GMT.panelTitle} ${GM.textStrong} flex items-center gap-2 mb-4`}>
        <FileText className={`w-4 h-4 ${GM.accent}`} />
        {t('synthTitle')}
      </h2>
      <AnalysisMarkdown content={markdown} />
    </motion.div>
  );
}
