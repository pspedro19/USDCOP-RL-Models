import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * SWR fetcher function for API calls
 * Used by hooks like useLiveState, useTradesHistory, etc.
 */
export async function fetcher<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
  })

  if (!res.ok) {
    const error = new Error('An error occurred while fetching the data.')
    throw error
  }

  return res.json()
}
