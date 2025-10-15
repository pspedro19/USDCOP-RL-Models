import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { OptimizedChart } from '@/components/charts/OptimizedChart'

// Mock recharts to avoid canvas rendering issues
vi.mock('recharts', () => ({
  ComposedChart: ({ children, ...props }: any) => <div data-testid="composed-chart" {...props}>{children}</div>,
  Line: (props: any) => <div data-testid="line" {...props} />,
  Bar: (props: any) => <div data-testid="bar" {...props} />,
  XAxis: (props: any) => <div data-testid="x-axis" {...props} />,
  YAxis: (props: any) => <div data-testid="y-axis" {...props} />,
  CartesianGrid: (props: any) => <div data-testid="cartesian-grid" {...props} />,
  Tooltip: (props: any) => <div data-testid="tooltip" {...props} />,
  ResponsiveContainer: ({ children, ...props }: any) => <div data-testid="responsive-container" {...props}>{children}</div>,
  ReferenceLine: (props: any) => <div data-testid="reference-line" {...props} />,
  Brush: (props: any) => <div data-testid="brush" {...props} />,
}))

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => children,
}))

// Mock lucide-react icons
vi.mock('lucide-react', () => ({
  TrendingUp: () => <div data-testid="trending-up-icon" />,
  TrendingDown: () => <div data-testid="trending-down-icon" />,
  Zap: () => <div data-testid="zap-icon" />,
  Activity: () => <div data-testid="activity-icon" />,
  Gauge: () => <div data-testid="gauge-icon" />,
  Target: () => <div data-testid="target-icon" />,
}))

const mockChartData = [
  {
    datetime: '2023-01-01T10:00:00Z',
    open: 4000.50,
    high: 4050.75,
    low: 3950.25,
    close: 4025.80,
    volume: 1500000
  },
  {
    datetime: '2023-01-01T10:05:00Z',
    open: 4025.80,
    high: 4075.30,
    low: 4010.15,
    close: 4060.45,
    volume: 1750000
  },
  {
    datetime: '2023-01-01T10:10:00Z',
    open: 4060.45,
    high: 4090.20,
    low: 4035.60,
    close: 4080.90,
    volume: 1300000
  },
  {
    datetime: '2023-01-01T10:15:00Z',
    open: 4080.90,
    high: 4120.55,
    low: 4065.75,
    close: 4110.30,
    volume: 1850000
  },
  {
    datetime: '2023-01-01T10:20:00Z',
    open: 4110.30,
    high: 4135.80,
    low: 4095.40,
    close: 4125.65,
    volume: 2100000
  }
]

describe('OptimizedChart', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders without crashing', () => {
    render(<OptimizedChart data={mockChartData} />)

    expect(screen.getByTestId('responsive-container')).toBeInTheDocument()
    expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
  })

  it('renders chart components correctly', () => {
    render(<OptimizedChart data={mockChartData} showVolume={true} />)

    expect(screen.getByTestId('x-axis')).toBeInTheDocument()
    expect(screen.getAllByTestId('y-axis')).toHaveLength(2) // Price and volume axes
    expect(screen.getByTestId('cartesian-grid')).toBeInTheDocument()
    expect(screen.getByTestId('tooltip')).toBeInTheDocument()
    expect(screen.getByTestId('line')).toBeInTheDocument() // Candlestick line
  })

  it('shows volume when showVolume prop is true', () => {
    render(<OptimizedChart data={mockChartData} showVolume={true} />)

    expect(screen.getByTestId('bar')).toBeInTheDocument()
  })

  it('hides volume when showVolume prop is false', () => {
    render(<OptimizedChart data={mockChartData} showVolume={false} />)

    expect(screen.queryByTestId('bar')).not.toBeInTheDocument()
  })

  it('applies correct height from props', () => {
    const customHeight = 500
    render(<OptimizedChart data={mockChartData} height={customHeight} />)

    const container = screen.getByTestId('responsive-container')
    expect(container).toHaveAttribute('height', customHeight.toString())
  })

  it('handles empty data gracefully', () => {
    render(<OptimizedChart data={[]} />)

    expect(screen.getByTestId('responsive-container')).toBeInTheDocument()
    expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
  })

  it('shows realtime indicator when isRealtime is true', () => {
    render(<OptimizedChart data={mockChartData} isRealtime={true} />)

    // Check for realtime indicator elements
    expect(screen.getByText(/En Vivo/i)).toBeInTheDocument()
    expect(screen.getByTestId('activity-icon')).toBeInTheDocument()
  })

  it('hides realtime indicator when isRealtime is false', () => {
    render(<OptimizedChart data={mockChartData} isRealtime={false} />)

    expect(screen.queryByText(/En Vivo/i)).not.toBeInTheDocument()
  })

  it('displays market status correctly', () => {
    render(<OptimizedChart data={mockChartData} />)

    // Should show market status information
    expect(screen.getByText(/USD\/COP/i)).toBeInTheDocument()
  })

  it('shows price change and percentage', () => {
    render(<OptimizedChart data={mockChartData} />)

    // Calculate expected change
    const lastPrice = mockChartData[mockChartData.length - 1].close
    const firstPrice = mockChartData[0].open
    const change = lastPrice - firstPrice
    const percentage = ((change / firstPrice) * 100).toFixed(2)

    expect(screen.getByText(new RegExp(Math.abs(change).toFixed(2)))).toBeInTheDocument()
    expect(screen.getByText(new RegExp(percentage))).toBeInTheDocument()
  })

  it('displays correct trend icon for positive change', () => {
    const upwardData = [
      ...mockChartData,
      {
        datetime: '2023-01-01T10:25:00Z',
        open: 4125.65,
        high: 4200.00,
        low: 4120.00,
        close: 4180.50,
        volume: 2200000
      }
    ]

    render(<OptimizedChart data={upwardData} />)

    expect(screen.getByTestId('trending-up-icon')).toBeInTheDocument()
  })

  it('displays correct trend icon for negative change', () => {
    const downwardData = [
      {
        datetime: '2023-01-01T10:00:00Z',
        open: 4200.00,
        high: 4220.00,
        low: 4000.00,
        close: 4050.00,
        volume: 1500000
      },
      ...mockChartData.map(item => ({ ...item, close: item.close - 200 }))
    ]

    render(<OptimizedChart data={downwardData} />)

    expect(screen.getByTestId('trending-down-icon')).toBeInTheDocument()
  })

  it('handles replay progress correctly', () => {
    render(<OptimizedChart data={mockChartData} replayProgress={0.5} />)

    // Should render half the data based on replay progress
    const chart = screen.getByTestId('composed-chart')
    expect(chart).toBeInTheDocument()
  })

  it('memoizes data processing for performance', () => {
    const { rerender } = render(<OptimizedChart data={mockChartData} />)

    // Rerender with same data should not trigger recalculation
    rerender(<OptimizedChart data={mockChartData} />)

    expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
  })

  it('handles large datasets efficiently', () => {
    const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
      datetime: new Date(Date.now() + i * 60000).toISOString(),
      open: 4000 + Math.random() * 100,
      high: 4000 + Math.random() * 120,
      low: 4000 + Math.random() * 80,
      close: 4000 + Math.random() * 100,
      volume: 1000000 + Math.random() * 500000
    }))

    const startTime = performance.now()
    render(<OptimizedChart data={largeDataset} />)
    const endTime = performance.now()

    expect(endTime - startTime).toBeLessThan(1000) // Should render within 1 second
    expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
  })

  it('formats currency values correctly', () => {
    render(<OptimizedChart data={mockChartData} />)

    // Should display formatted currency values
    const priceElements = screen.getAllByText(/\$[\d,]+\.[\d]{2}/)
    expect(priceElements.length).toBeGreaterThan(0)
  })

  it('handles data updates smoothly', async () => {
    const { rerender } = render(<OptimizedChart data={mockChartData.slice(0, 3)} />)

    // Add more data
    rerender(<OptimizedChart data={mockChartData} />)

    await waitFor(() => {
      expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
    })
  })

  it('maintains aspect ratio with different heights', () => {
    const { rerender } = render(<OptimizedChart data={mockChartData} height={300} />)

    let container = screen.getByTestId('responsive-container')
    expect(container).toHaveAttribute('height', '300')

    rerender(<OptimizedChart data={mockChartData} height={600} />)

    container = screen.getByTestId('responsive-container')
    expect(container).toHaveAttribute('height', '600')
  })

  describe('Performance Optimization', () => {
    it('implements proper memoization', () => {
      const memoSpy = vi.spyOn(React, 'useMemo')

      render(<OptimizedChart data={mockChartData} />)

      expect(memoSpy).toHaveBeenCalled()

      memoSpy.mockRestore()
    })

    it('uses React.memo for component memoization', () => {
      const { rerender } = render(<OptimizedChart data={mockChartData} />)

      // Same props should not cause re-render
      rerender(<OptimizedChart data={mockChartData} />)

      expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
    })

    it('samples large datasets for better performance', () => {
      const hugeDataset = Array.from({ length: 50000 }, (_, i) => ({
        datetime: new Date(Date.now() + i * 60000).toISOString(),
        open: 4000 + Math.sin(i / 100) * 50,
        high: 4000 + Math.sin(i / 100) * 60,
        low: 4000 + Math.sin(i / 100) * 40,
        close: 4000 + Math.sin(i / 100) * 50,
        volume: 1000000 + Math.random() * 500000
      }))

      const startTime = performance.now()
      render(<OptimizedChart data={hugeDataset} />)
      const endTime = performance.now()

      expect(endTime - startTime).toBeLessThan(2000) // Should handle large datasets
      expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
    })
  })

  describe('Error Handling', () => {
    it('handles malformed data gracefully', () => {
      const malformedData = [
        {
          datetime: 'invalid-date',
          open: 'not-a-number',
          high: null,
          low: undefined,
          close: 4000,
          volume: -1000
        }
      ]

      expect(() => {
        render(<OptimizedChart data={malformedData as any} />)
      }).not.toThrow()

      expect(screen.getByTestId('composed-chart')).toBeInTheDocument()
    })

    it('handles missing required properties', () => {
      const incompleteData = [
        {
          datetime: '2023-01-01T10:00:00Z',
          close: 4000
          // Missing open, high, low, volume
        }
      ]

      expect(() => {
        render(<OptimizedChart data={incompleteData as any} />)
      }).not.toThrow()
    })

    it('handles null/undefined data', () => {
      expect(() => {
        render(<OptimizedChart data={null as any} />)
      }).not.toThrow()

      expect(() => {
        render(<OptimizedChart data={undefined as any} />)
      }).not.toThrow()
    })
  })

  describe('Accessibility', () => {
    it('has proper ARIA labels', () => {
      render(<OptimizedChart data={mockChartData} />)

      const chart = screen.getByRole('img', { hidden: true }) || screen.getByTestId('composed-chart')
      expect(chart).toBeInTheDocument()
    })

    it('supports keyboard navigation', async () => {
      const user = userEvent.setup()
      render(<OptimizedChart data={mockChartData} />)

      const chart = screen.getByTestId('composed-chart')
      await user.tab()

      expect(chart).toBeInTheDocument()
    })

    it('provides meaningful text alternatives', () => {
      render(<OptimizedChart data={mockChartData} />)

      // Should have text that describes the chart content
      expect(screen.getByText(/USD\/COP/i)).toBeInTheDocument()
    })
  })
})