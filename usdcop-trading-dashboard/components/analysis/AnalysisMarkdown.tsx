'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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
            <h2 className="text-lg font-semibold text-white mt-6 mb-3 border-b border-gray-700/50 pb-2">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold text-gray-200 mt-4 mb-2">{children}</h3>
          ),
          p: ({ children }) => (
            <p className="text-gray-300 text-sm leading-relaxed mb-3">{children}</p>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-inside text-gray-300 text-sm space-y-1 mb-3">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside text-gray-300 text-sm space-y-1 mb-3">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="text-gray-300 text-sm">{children}</li>
          ),
          strong: ({ children }) => (
            <strong className="text-white font-semibold">{children}</strong>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto mb-4">
              <table className="w-full text-sm text-left border-collapse">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="border-b border-gray-700 bg-gray-800/30">{children}</thead>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 text-gray-400 font-medium text-xs uppercase tracking-wider">{children}</th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-gray-300 border-b border-gray-800/50">{children}</td>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 hover:text-cyan-300 underline decoration-cyan-400/30 hover:decoration-cyan-300/60 transition-colors"
            >
              {children}
            </a>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-cyan-500/50 pl-4 italic text-gray-400 my-3">{children}</blockquote>
          ),
          code: ({ children, className: codeClassName }) => {
            const isBlock = codeClassName?.includes('language-');
            if (isBlock) {
              return (
                <pre className="bg-gray-900/80 rounded-lg p-3 overflow-x-auto mb-3">
                  <code className="text-sm text-gray-300">{children}</code>
                </pre>
              );
            }
            return (
              <code className="bg-gray-800/80 rounded px-1.5 py-0.5 text-cyan-300 text-sm">{children}</code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
