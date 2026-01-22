import { create } from 'zustand'

export interface Toast {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title?: string
  message: string
  duration?: number
}

interface UIState {
  sidebarCollapsed: boolean
  sidebarMobileOpen: boolean
  toasts: Toast[]
  isGlobalLoading: boolean

  // Actions
  toggleSidebar: () => void
  setSidebarCollapsed: (collapsed: boolean) => void
  toggleMobileSidebar: () => void
  setMobileSidebarOpen: (open: boolean) => void
  addToast: (toast: Omit<Toast, 'id'>) => void
  removeToast: (id: string) => void
  clearToasts: () => void
  setGlobalLoading: (loading: boolean) => void
}

let toastId = 0

export const useUIStore = create<UIState>((set, get) => ({
  sidebarCollapsed: false,
  sidebarMobileOpen: false,
  toasts: [],
  isGlobalLoading: false,

  toggleSidebar: () => {
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }))
  },

  setSidebarCollapsed: (collapsed: boolean) => {
    set({ sidebarCollapsed: collapsed })
  },

  toggleMobileSidebar: () => {
    set((state) => ({ sidebarMobileOpen: !state.sidebarMobileOpen }))
  },

  setMobileSidebarOpen: (open: boolean) => {
    set({ sidebarMobileOpen: open })
  },

  addToast: (toast: Omit<Toast, 'id'>) => {
    const id = `toast-${++toastId}`
    const newToast: Toast = { ...toast, id }

    set((state) => ({
      toasts: [...state.toasts, newToast],
    }))

    // Auto remove after duration
    const duration = toast.duration ?? 5000
    if (duration > 0) {
      setTimeout(() => {
        get().removeToast(id)
      }, duration)
    }
  },

  removeToast: (id: string) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }))
  },

  clearToasts: () => {
    set({ toasts: [] })
  },

  setGlobalLoading: (loading: boolean) => {
    set({ isGlobalLoading: loading })
  },
}))

// Helper functions for common toast types
export const toast = {
  success: (message: string, title?: string) => {
    useUIStore.getState().addToast({ type: 'success', message, title })
  },
  error: (message: string, title?: string) => {
    useUIStore.getState().addToast({ type: 'error', message, title })
  },
  warning: (message: string, title?: string) => {
    useUIStore.getState().addToast({ type: 'warning', message, title })
  },
  info: (message: string, title?: string) => {
    useUIStore.getState().addToast({ type: 'info', message, title })
  },
}
