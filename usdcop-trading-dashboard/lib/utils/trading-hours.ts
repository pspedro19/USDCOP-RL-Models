export function isTradingHours(): boolean {
  const now = new Date()
  const hour = now.getHours()
  return hour >= 8 && hour <= 17
}

export function getNextTradingSession(): Date {
  const now = new Date()
  const next = new Date(now)
  next.setHours(8, 0, 0, 0)
  if (now.getHours() >= 17) {
    next.setDate(next.getDate() + 1)
  }
  return next
}

export function getTradingSessionInfo() {
  const isOpen = isTradingHours()
  const nextSession = getNextTradingSession()
  const timeToOpen = nextSession.getTime() - Date.now()

  return {
    isOpen,
    nextSession,
    timeToOpen,
    status: isOpen ? 'OPEN' : 'CLOSED'
  }
}

export function formatTimeToOpen(timeMs: number): string {
  const hours = Math.floor(timeMs / (1000 * 60 * 60))
  const minutes = Math.floor((timeMs % (1000 * 60 * 60)) / (1000 * 60))

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  return `${minutes}m`
}
