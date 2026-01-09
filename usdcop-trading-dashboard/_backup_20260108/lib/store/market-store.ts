import { create } from 'zustand'

interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
  timestamp: number
}

interface MarketStore {
  data: MarketData[]
  isReplayMode: boolean
  currentTime: number
  speed: number
  isPlaying: boolean
  setReplayMode: (mode: boolean) => void
  setCurrentTime: (time: number) => void
  setSpeed: (speed: number) => void
  setIsPlaying: (playing: boolean) => void
  updateData: (data: MarketData[]) => void
}

export const useMarketStore = create<MarketStore>((set) => ({
  data: [],
  isReplayMode: false,
  currentTime: Date.now(),
  speed: 1,
  isPlaying: false,
  setReplayMode: (mode) => set({ isReplayMode: mode }),
  setCurrentTime: (time) => set({ currentTime: time }),
  setSpeed: (speed) => set({ speed }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  updateData: (data) => set({ data }),
}))
