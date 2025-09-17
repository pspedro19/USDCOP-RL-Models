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
