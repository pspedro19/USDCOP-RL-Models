# ChartPro - Institutional Grade Trading Charts

A professional-grade charting engine built on TradingView Lightweight Charts v5, designed to rival Bloomberg Terminal's charting capabilities with modern web technologies.

## 🚀 Features

### Core Charting
- **TradingView Lightweight Charts v5** - Industry-leading chart performance
- **Professional Themes** - Bloomberg Terminal inspired dark/light themes
- **Real-time Updates** - 60 FPS performance with WebGL acceleration
- **Multiple Chart Types** - Candlestick, line, area, volume, and custom types

### Drawing Tools (Fabric.js Integration)
- **Trendlines** - Support/resistance lines with magnetic snap
- **Fibonacci Tools** - Retracements, extensions, and fan lines
- **Geometric Shapes** - Rectangles, ellipses, arrows, and polygons
- **Text Annotations** - Rich text with formatting options
- **Advanced Tools** - Parallel channels, pitchforks, Gann fans
- **Persistent Storage** - Save and restore drawings across sessions

### Volume Profile (Apache ECharts)
- **Point of Control (POC)** - Highest volume price level
- **Value Area** - 70% volume distribution zone
- **Volume Distribution** - Horizontal volume histogram
- **Market Profile** - Time-based volume analysis
- **VWAP Integration** - Volume weighted average price

### Technical Indicators
- **Plugin Architecture** - Extensible indicator system
- **Built-in Indicators** - 20+ professional indicators
- **Custom Indicators** - Easy to create custom calculations
- **Real-time Updates** - Live indicator calculations
- **Multiple Timeframes** - Indicator calculations across timeframes

### Export & Print
- **High-Quality Export** - PNG, SVG, PDF formats
- **Print Support** - Professional print layouts
- **Clipboard Copy** - Quick sharing capabilities
- **Batch Export** - Multiple formats simultaneously
- **Metadata Inclusion** - Chart details in exports

### Performance Optimization
- **WebGL Acceleration** - Hardware-accelerated rendering
- **Data Sampling** - Intelligent data reduction for large datasets
- **Level of Detail** - Dynamic quality adjustment
- **Memory Management** - Automatic cleanup and optimization
- **Performance Monitoring** - Real-time performance metrics

## 📦 Installation

The chart engine is built into the USDCOP Trading Dashboard. All dependencies are already included:

```json
{
  "lightweight-charts": "^5.0.8",
  "fabric": "^6.7.1",
  "echarts": "^6.0.0",
  "uplot": "^1.6.32",
  "technicalindicators": "^3.1.0"
}
```

## 🎯 Quick Start

### Basic Usage

```tsx
import { ChartPro } from './components/charts/chart-engine';

const MyChart = () => {
  const data = {
    candlesticks: [
      { time: 1642427876, open: 4000, high: 4010, low: 3990, close: 4005 },
      // ... more data
    ],
    volume: [
      { time: 1642427876, value: 1000000, color: '#26a69a' },
      // ... more data
    ],
    indicators: {
      'SMA 20': [
        { time: 1642427876, value: 4002 },
        // ... more data
      ]
    }
  };

  return (
    <ChartPro
      data={data}
      height={600}
      theme="dark"
      enableDrawingTools={true}
      enableVolumeProfile={true}
      enableIndicators={true}
    />
  );
};
```

### Advanced Configuration

```tsx
import { ChartPro, createOptimalChartConfig } from './components/charts/chart-engine';

const AdvancedChart = () => {
  const config = createOptimalChartConfig({
    performance: {
      enableWebGL: true,
      maxDataPoints: 100000,
      updateFrequency: 16
    },
    features: {
      enableDrawingTools: true,
      enableVolumeProfile: true,
      enableTechnicalIndicators: true
    }
  });

  return (
    <ChartPro
      data={data}
      config={config}
      onCrosshairMove={(price, time) => {
        console.log('Price:', price, 'Time:', time);
      }}
      onVisibleRangeChange={(range) => {
        console.log('Visible range:', range);
      }}
    />
  );
};
```

## 🔧 Component Architecture

### Core Components

```
chart-engine/
├── ChartPro.tsx              # Main chart component
├── ChartProDemo.tsx           # Full-featured demo
├── core/
│   └── ChartConfig.ts         # Chart configuration
├── tools/
│   └── DrawingToolsManager.ts # Drawing tools with fabric.js
├── volume/
│   └── VolumeProfileManager.ts # Volume profile with ECharts
├── indicators/
│   └── IndicatorManager.ts    # Technical indicators
├── export/
│   └── ExportManager.ts       # Export functionality
├── performance/
│   └── PerformanceMonitor.ts  # Performance optimization
└── index.ts                   # Main exports
```

### Manager Classes

Each manager is a self-contained module that can be used independently:

- **DrawingToolsManager** - Handles all drawing tools using fabric.js
- **VolumeProfileManager** - Renders volume profile using Apache ECharts
- **IndicatorManager** - Plugin-based technical indicator system
- **ExportManager** - High-quality export in multiple formats
- **PerformanceMonitor** - Real-time performance monitoring and optimization

## 🎨 Themes and Customization

### Built-in Themes

```tsx
import { INSTITUTIONAL_DARK_THEME, PROFESSIONAL_LIGHT_THEME } from './chart-engine';

// Dark theme (Bloomberg Terminal style)
<ChartPro theme="dark" config={INSTITUTIONAL_DARK_THEME} />

// Light theme (Professional trading)
<ChartPro theme="light" config={PROFESSIONAL_LIGHT_THEME} />
```

### Custom Theme

```tsx
const customTheme = {
  layout: {
    background: { type: 'solid', color: '#1a1a1a' },
    textColor: '#ffffff'
  },
  grid: {
    vertLines: { color: 'rgba(255, 255, 255, 0.1)' },
    horzLines: { color: 'rgba(255, 255, 255, 0.1)' }
  },
  // ... more customization
};
```

## 🛠 Drawing Tools

### Available Tools

- **Crosshair** - Default selection tool
- **Trendline** - Draw trend and support/resistance lines
- **Horizontal Line** - Price level lines
- **Vertical Line** - Time-based lines
- **Rectangle** - Area selection and zones
- **Ellipse** - Circular and oval shapes
- **Fibonacci** - Retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- **Text** - Annotations and labels
- **Arrow** - Directional indicators
- **Parallel Channel** - Channel projections
- **Pitchfork** - Andrews pitchfork
- **Gann Fan** - Gann angle lines
- **Elliott Wave** - Wave analysis tools

### Drawing Tools API

```tsx
// Access drawing tools manager
const drawingToolsRef = useRef<DrawingToolsManager>();

// Set active tool
drawingToolsRef.current?.setActiveTool('trendline');

// Clear all drawings
drawingToolsRef.current?.clearAllDrawings();

// Export drawings
const drawings = drawingToolsRef.current?.exportDrawings();

// Import drawings
drawingToolsRef.current?.importDrawings(savedDrawings);
```

## 📊 Volume Profile

### Features

- **Point of Control (POC)** - Price level with highest volume
- **Value Area** - Range containing 70% of volume
- **Volume Distribution** - Horizontal histogram showing volume by price
- **Customizable Levels** - Adjustable number of price levels
- **Real-time Updates** - Live volume profile calculations

### Volume Profile API

```tsx
// Calculate volume profile
const profileData = volumeProfileManager.calculateVolumeProfile(
  candleData,
  volumeData
);

// Show/hide volume profile
volumeProfileManager.show();
volumeProfileManager.hide();

// Update configuration
volumeProfileManager.updateConfig({
  numberOfLevels: 100,
  valueAreaPercentage: 70,
  position: 'right'
});
```

## 📈 Technical Indicators

### Built-in Indicators

**Moving Averages:**
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Weighted Moving Average (WMA)
- Double/Triple EMA (DEMA/TEMA)

**Oscillators:**
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Commodity Channel Index (CCI)
- Williams %R

**Volatility:**
- Bollinger Bands
- Average True Range (ATR)

**Volume:**
- On Balance Volume (OBV)
- Money Flow Index (MFI)
- Volume Weighted Average Price (VWAP)

### Custom Indicators

```tsx
// Create custom indicator plugin
const customIndicator: IndicatorPlugin = {
  id: 'custom_rsi',
  name: 'Custom RSI',
  description: 'RSI with custom parameters',
  parameters: [
    {
      name: 'period',
      type: 'number',
      defaultValue: 14,
      min: 2,
      max: 100,
      description: 'RSI calculation period'
    }
  ],
  calculate: (data, params) => {
    // Custom calculation logic
    return { main: calculatedData };
  },
  defaultStyle: {
    color: '#ff6b6b',
    width: 2,
    style: 'solid'
  },
  isOverlay: false
};

// Register custom indicator
indicatorManager.registerPlugin(customIndicator);

// Add to chart
const indicatorId = indicatorManager.addIndicator({
  type: 'custom_rsi',
  parameters: { period: 21 }
});
```

## 💾 Export Functionality

### Supported Formats

- **PNG** - High-resolution raster images
- **SVG** - Scalable vector graphics
- **PDF** - Professional documents with metadata

### Export API

```tsx
// Single format export
await exportManager.export('png', {
  quality: 0.9,
  scale: 2,
  filename: 'chart_analysis',
  includeWatermark: true
});

// Batch export
await exportManager.exportBatch(['png', 'svg', 'pdf'], {
  quality: 0.9,
  scale: 2
});

// Copy to clipboard
await exportManager.copyToClipboard();

// Print
await exportManager.print();
```

## ⚡ Performance Optimization

### Automatic Optimizations

- **Data Sampling** - Reduces data points for large datasets
- **Level of Detail** - Adjusts visual quality based on zoom level
- **WebGL Acceleration** - Hardware-accelerated rendering when available
- **Batch Updates** - Groups multiple updates for efficiency
- **Memory Management** - Automatic cleanup of unused resources

### Performance Monitoring

```tsx
// Monitor performance metrics
performanceMonitor.onStatsUpdate = (metrics) => {
  console.log('FPS:', metrics.fps);
  console.log('Render Time:', metrics.renderTime);
  console.log('Memory Usage:', metrics.memoryUsage);
};

// Handle performance issues
performanceMonitor.onPerformanceIssue = (issue, metrics) => {
  console.warn('Performance Issue:', issue);
  // Automatic optimizations will be applied
};
```

### Manual Optimizations

```tsx
// Enable specific optimizations
performanceMonitor.enableOptimization('datasampling');
performanceMonitor.enableOptimization('webGLAcceleration');

// Set performance thresholds
performanceMonitor.setThresholds({
  minFPS: 30,
  maxRenderTime: 16,
  maxDataPoints: 50000
});
```

## 🎮 Event Handling

### Chart Events

```tsx
<ChartPro
  onCrosshairMove={(price, time) => {
    // Handle crosshair movement
    setCurrentPrice(price);
    setCurrentTime(time);
  }}

  onVisibleRangeChange={(range) => {
    // Handle zoom/pan changes
    console.log('Visible range:', range);
  }}

  onSeriesClick={(series, point) => {
    // Handle series clicks
    console.log('Clicked:', series, point);
  }}
/>
```

### Drawing Events

```tsx
drawingToolsManager.onDrawingComplete = (drawing) => {
  console.log('Drawing completed:', drawing);
  saveDrawing(drawing);
};

drawingToolsManager.onDrawingSelected = (drawing) => {
  console.log('Drawing selected:', drawing);
  showDrawingProperties(drawing);
};
```

## 🔄 Real-time Updates

### Live Data Integration

```tsx
// Update chart data in real-time
useEffect(() => {
  const ws = new WebSocket('wss://api.example.com/market-data');

  ws.onmessage = (event) => {
    const newData = JSON.parse(event.data);

    // Update candlestick data
    chartRef.current?.updateData({
      candlesticks: [...existingData, newData.candle],
      volume: [...existingVolume, newData.volume]
    });
  };

  return () => ws.close();
}, []);
```

### Performance Considerations

- Use `batchUpdates: true` for multiple simultaneous updates
- Implement data windowing for very large datasets
- Consider using Web Workers for heavy calculations
- Monitor performance metrics during live updates

## 🧪 Testing

### Unit Tests

```tsx
// Test chart rendering
test('renders chart with data', () => {
  render(<ChartPro data={mockData} />);
  expect(screen.getByTestId('chart-container')).toBeInTheDocument();
});

// Test drawing tools
test('drawing tool functionality', () => {
  const manager = new DrawingToolsManager(mockChart, mockContainer);
  manager.setActiveTool('trendline');
  expect(manager.getActiveTool()).toBe('trendline');
});
```

### Performance Tests

```tsx
// Test performance with large datasets
test('handles large datasets efficiently', async () => {
  const largeDataset = generateMockData(100000);

  const startTime = performance.now();
  render(<ChartPro data={largeDataset} />);
  const renderTime = performance.now() - startTime;

  expect(renderTime).toBeLessThan(1000); // Should render in under 1 second
});
```

## 🚀 Deployment

### Production Build

```bash
npm run build
```

### Performance Checklist

- [ ] WebGL support enabled
- [ ] Data sampling configured
- [ ] Performance monitoring active
- [ ] Memory limits set
- [ ] Error boundaries implemented
- [ ] Fallback themes available

## 📊 Browser Support

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| Core Charts | ✅ | ✅ | ✅ | ✅ |
| WebGL | ✅ | ✅ | ✅ | ✅ |
| Drawing Tools | ✅ | ✅ | ✅ | ✅ |
| Export | ✅ | ✅ | ✅ | ✅ |
| Performance API | ✅ | ✅ | ✅ | ✅ |

### Minimum Requirements

- Modern browser with ES2018 support
- Canvas 2D context support
- WebGL support (recommended)
- 4GB RAM (8GB recommended for large datasets)

## 🤝 Contributing

### Development Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

### Code Style

- TypeScript for type safety
- ESLint for code quality
- Prettier for formatting
- React best practices
- Performance-first approach

## 📝 License

This chart engine is part of the USDCOP Trading Dashboard project. See the main project license for details.

## 📞 Support

For technical support or questions about the chart engine:

1. Check the demo component for usage examples
2. Review the TypeScript interfaces for API details
3. Monitor browser console for performance warnings
4. Use the built-in performance monitor for optimization

---

**ChartPro** - Where institutional-grade meets modern web technology. 📈