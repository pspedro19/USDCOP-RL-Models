import { create } from 'zustand';
import type { ChatMessage } from '@/lib/contracts/weekly-analysis.contract';

interface AnalysisChatState {
  isOpen: boolean;
  messages: ChatMessage[];
  sessionId: string;
  contextYear: number;
  contextWeek: number;
  isTyping: boolean;
  totalTokens: number;

  // Actions
  toggle: () => void;
  open: () => void;
  close: () => void;
  setContext: (year: number, week: number) => void;
  addMessage: (message: ChatMessage) => void;
  setTyping: (typing: boolean) => void;
  addTokens: (tokens: number) => void;
  clearMessages: () => void;
  newSession: () => void;
}

function generateSessionId(): string {
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export const useAnalysisChatStore = create<AnalysisChatState>((set) => ({
  isOpen: false,
  messages: [],
  sessionId: generateSessionId(),
  contextYear: 2026,
  contextWeek: 1,
  isTyping: false,
  totalTokens: 0,

  toggle: () => set((s) => ({ isOpen: !s.isOpen })),
  open: () => set({ isOpen: true }),
  close: () => set({ isOpen: false }),

  setContext: (year, week) => set({ contextYear: year, contextWeek: week }),

  addMessage: (message) =>
    set((s) => ({ messages: [...s.messages, message] })),

  setTyping: (typing) => set({ isTyping: typing }),

  addTokens: (tokens) =>
    set((s) => ({ totalTokens: s.totalTokens + tokens })),

  clearMessages: () => set({ messages: [] }),

  newSession: () =>
    set({
      messages: [],
      sessionId: generateSessionId(),
      totalTokens: 0,
    }),
}));
