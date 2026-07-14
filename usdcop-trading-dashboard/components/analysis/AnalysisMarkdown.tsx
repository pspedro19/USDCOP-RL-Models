'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { GM, GMT } from '@/lib/ui/gm-tokens';

interface AnalysisMarkdownProps {
  content: string;
  className?: string;
}

export function AnalysisMarkdown({ content, className = '' }: AnalysisMarkdownProps) {
  return (
    <div className={`prose prose-invert prose-sm max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h2: ({ children }) => (
            <h2 className={`${GMT.h2} ${GM.headline} mt-6 mb-3 border-b border-[var(--gm-border)] pb-2`}>
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className={`text-base font-semibold ${GM.text} mt-4 mb-2`}>{children}</h3>
          ),
          p: ({ children }) => (
            <p className={`${GM.textSec} ${GMT.body} leading-relaxed mb-3`}>{children}</p>
          ),
          ul: ({ children }) => (
            <ul className={`list-disc list-inside ${GM.textSec} ${GMT.body} space-y-1 mb-3`}>{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className={`list-decimal list-inside ${GM.textSec} ${GMT.body} space-y-1 mb-3`}>{children}</ol>
          ),
          li: ({ children }) => (
            <li className={`${GM.textSec} ${GMT.body}`}>{children}</li>
          ),
          strong: ({ children }) => (
            <strong className={`${GM.textStrong} font-semibold`}>{children}</strong>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto mb-4">
              <table className={`w-full ${GMT.body} text-left border-collapse`}>{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="border-b border-[var(--gm-border)] bg-[rgba(148,163,184,.05)]">{children}</thead>
          ),
          th: ({ children }) => (
            <th className={`px-3 py-2 ${GMT.label} ${GM.textMuted}`}>{children}</th>
          ),
          td: ({ children }) => (
            <td className={`px-3 py-2 ${GM.textSec} border-b border-[var(--gm-border)]`}>{children}</td>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className={`${GM.accent} hover:opacity-80 underline decoration-[rgba(34,211,238,.3)] transition-opacity duration-[var(--gm-dur-fast)] ${GM.focus} rounded`}
            >
              {children}
            </a>
          ),
          blockquote: ({ children }) => (
            <blockquote className={`border-l-2 border-[var(--gm-accent)] pl-4 italic ${GM.textMuted} my-3`}>{children}</blockquote>
          ),
          code: ({ children, className: codeClassName }) => {
            const isBlock = codeClassName?.includes('language-');
            if (isBlock) {
              return (
                <pre className={`${GM.panelInner} p-3 overflow-x-auto mb-3`}>
                  <code className={`${GMT.body} ${GM.textSec} ${GMT.mono}`}>{children}</code>
                </pre>
              );
            }
            return (
              <code className={`${GM.panelInner} rounded px-1.5 py-0.5 ${GM.accent} ${GMT.body} ${GMT.mono}`}>{children}</code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
