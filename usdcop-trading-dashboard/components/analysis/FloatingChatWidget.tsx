'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, X, Send, RotateCcw, Zap } from 'lucide-react';
import { useAnalysisChatStore } from '@/stores/useAnalysisChatStore';
import type { ChatMessage } from '@/lib/contracts/weekly-analysis.contract';
import { AnalysisMarkdown } from './AnalysisMarkdown';

const QUICK_ACTIONS = [
  { label: 'Resumen', prompt: 'Dame un resumen rapido de la semana' },
  { label: 'Macro', prompt: 'Como estan los indicadores macro clave?' },
  { label: 'Señales', prompt: 'Que dicen las señales de los modelos H1 y H5?' },
  { label: 'Perspectiva', prompt: 'Cual es la perspectiva para la proxima semana?' },
];

export function FloatingChatWidget() {
  const {
    isOpen, toggle, messages, sessionId, contextYear, contextWeek,
    isTyping, addMessage, setTyping, addTokens, newSession,
  } = useAnalysisChatStore();

  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 200);
    }
  }, [isOpen]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || isTyping) return;

    const userMsg: ChatMessage = {
      id: `msg_${Date.now()}_user`,
      role: 'user',
      content: text.trim(),
      timestamp: new Date().toISOString(),
    };
    addMessage(userMsg);
    setInput('');
    setTyping(true);

    try {
      const res = await fetch('/api/analysis/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text.trim(),
          session_id: sessionId,
          year: contextYear,
          week: contextWeek,
        }),
      });

      const data = await res.json();

      const assistantMsg: ChatMessage = {
        id: `msg_${Date.now()}_assistant`,
        role: 'assistant',
        content: data.reply || 'Error procesando respuesta.',
        timestamp: new Date().toISOString(),
        tokens_used: data.tokens_used,
      };
      addMessage(assistantMsg);

      if (data.tokens_used) {
        addTokens(data.tokens_used);
      }
    } catch {
      addMessage({
        id: `msg_${Date.now()}_error`,
        role: 'assistant',
        content: 'Error de conexion. Intenta de nuevo.',
        timestamp: new Date().toISOString(),
      });
    } finally {
      setTyping(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  return (
    <>
      {/* FAB button */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            onClick={toggle}
            className="fixed bottom-6 right-6 z-40 w-14 h-14 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-shadow flex items-center justify-center"
          >
            <MessageSquare className="w-6 h-6" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Chat panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-6 right-6 z-50 w-[380px] max-h-[560px] bg-gray-900/95 backdrop-blur-xl rounded-2xl border border-gray-800 shadow-2xl shadow-black/50 flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800/50 bg-gray-900/80">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-sm font-semibold text-white">Asistente USDCOP</span>
                <span className="text-[10px] text-gray-600 bg-gray-800/60 rounded px-1.5 py-0.5">
                  {contextYear}-W{String(contextWeek).padStart(2, '0')}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={newSession}
                  className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
                  title="Nueva sesion"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-gray-500" />
                </button>
                <button
                  onClick={toggle}
                  className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
                >
                  <X className="w-4 h-4 text-gray-400" />
                </button>
              </div>
            </div>

            {/* Messages area */}
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 min-h-[200px] max-h-[380px]">
              {messages.length === 0 && (
                <div className="text-center py-6">
                  <MessageSquare className="w-8 h-8 text-gray-700 mx-auto mb-2" />
                  <p className="text-sm text-gray-500 mb-3">
                    Pregunta sobre el analisis semanal del USD/COP
                  </p>

                  {/* Quick actions */}
                  <div className="flex flex-wrap gap-1.5 justify-center">
                    {QUICK_ACTIONS.map((qa) => (
                      <button
                        key={qa.label}
                        onClick={() => sendMessage(qa.prompt)}
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 bg-gray-800/60 border border-gray-700/40 rounded-full text-xs text-gray-400 hover:text-white hover:border-cyan-500/30 transition-all"
                      >
                        <Zap className="w-3 h-3" />
                        {qa.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg) => (
                <ChatBubble key={msg.id} message={msg} />
              ))}

              {isTyping && (
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="flex gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-gray-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  Pensando...
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <form
              onSubmit={handleSubmit}
              className="flex items-center gap-2 px-4 py-3 border-t border-gray-800/50 bg-gray-900/60"
            >
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Escribe tu pregunta..."
                disabled={isTyping}
                className="flex-1 bg-gray-800/60 border border-gray-700/40 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={!input.trim() || isTyping}
                className="p-2 rounded-lg bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] rounded-xl px-3 py-2 ${
        isUser
          ? 'bg-cyan-500/20 border border-cyan-500/20 text-white'
          : 'bg-gray-800/60 border border-gray-700/30 text-gray-200'
      }`}>
        {isUser ? (
          <p className="text-sm">{message.content}</p>
        ) : (
          <AnalysisMarkdown content={message.content} className="text-sm" />
        )}
        {message.tokens_used && (
          <p className="text-[9px] text-gray-600 mt-1 text-right">{message.tokens_used} tokens</p>
        )}
      </div>
    </div>
  );
}
