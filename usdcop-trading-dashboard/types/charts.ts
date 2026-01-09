/**
 * Chart Types
 * ============
 *
 * Tipos para gráficos, visualizaciones, TradingView, etc.
 */

import { OHLCVData, TechnicalIndicators, Position } from './trading';
import { DeepPartial } from './common';

// === CHART TYPES ===

/**
 * Tipo de gráfico
 */
export enum ChartType {
  CANDLESTICK = 'candlestick',
  LINE = 'line',
  AREA = 'area',
  BAR = 'bar',
  HISTOGRAM = 'histogram',
  BASELINE = 'baseline',
  HEIKIN_ASHI = 'heikin_ashi',
}

/**
 * Tema del gráfico
 */
export enum ChartTheme {
  DARK = 'dark',
  LIGHT = 'light',
  CUSTOM = 'custom',
}

/**
 * Timeframe del gráfico
 */
export enum ChartTimeframe {
  ONE_MIN = '1m',
  FIVE_MIN = '5m',
  FIFTEEN_MIN = '15m',
  THIRTY_MIN = '30m',
  ONE_HOUR = '1h',
  FOUR_HOUR = '4h',
  ONE_DAY = '1d',
  ONE_WEEK = '1w',
  ONE_MONTH = '1M',
}

// === CHART DATA ===

/**
 * Datos base para gráfico
 */
export interface ChartData extends OHLCVData {
  indicators?: TechnicalIndicators;
}

/**
 * Serie de datos para gráfico
 */
export interface ChartSeries<T = any> {
  id: string;
  name: string;
  type: ChartType;
  data: T[];
  color?: string;
  visible?: boolean;
  options?: Record<string, any>;
}

/**
 * Datos de línea
 */
export interface LineData {
  time: number;
  value: number;
}

/**
 * Datos de área
 */
export interface AreaData extends LineData {
  topColor?: string;
  bottomColor?: string;
}

/**
 * Datos de histograma
 */
export interface HistogramData {
  time: number;
  value: number;
  color?: string;
}

// === CHART CONFIGURATION ===

/**
 * Configuración de layout
 */
export interface LayoutConfig {
  background?: {
    type: 'solid' | 'gradient';
    color?: string;
    topColor?: string;
    bottomColor?: string;
  };
  textColor?: string;
  fontSize?: number;
  fontFamily?: string;
}

/**
 * Configuración de grid
 */
export interface GridConfig {
  vertLines?: {
    color?: string;
    style?: number;
    visible?: boolean;
  };
  horzLines?: {
    color?: string;
    style?: number;
    visible?: boolean;
  };
}

/**
 * Configuración de crosshair
 */
export interface CrosshairConfig {
  mode?: 'normal' | 'magnet';
  vertLine?: {
    color?: string;
    width?: number;
    style?: number;
    visible?: boolean;
    labelVisible?: boolean;
    labelBackgroundColor?: string;
  };
  horzLine?: {
    color?: string;
    width?: number;
    style?: number;
    visible?: boolean;
    labelVisible?: boolean;
    labelBackgroundColor?: string;
  };
}

/**
 * Configuración de escala de precio
 */
export interface PriceScaleConfig {
  mode?: 'normal' | 'logarithmic' | 'percentage' | 'indexed';
  autoScale?: boolean;
  invertScale?: boolean;
  alignLabels?: boolean;
  borderVisible?: boolean;
  borderColor?: string;
  textColor?: string;
  visible?: boolean;
  scaleMargins?: {
    top?: number;
    bottom?: number;
  };
}

/**
 * Configuración de escala de tiempo
 */
export interface TimeScaleConfig {
  rightOffset?: number;
  barSpacing?: number;
  fixLeftEdge?: boolean;
  lockVisibleTimeRangeOnResize?: boolean;
  rightBarStaysOnScroll?: boolean;
  borderVisible?: boolean;
  borderColor?: string;
  visible?: boolean;
  timeVisible?: boolean;
  secondsVisible?: boolean;
}

/**
 * Configuración completa del gráfico
 */
export interface ChartConfig {
  theme: ChartTheme;
  layout?: LayoutConfig;
  grid?: GridConfig;
  crosshair?: CrosshairConfig;
  rightPriceScale?: PriceScaleConfig;
  leftPriceScale?: PriceScaleConfig;
  timeScale?: TimeScaleConfig;
  watermark?: {
    visible?: boolean;
    text?: string;
    color?: string;
    fontSize?: number;
  };
  handleScroll?: {
    mouseWheel?: boolean;
    pressedMouseMove?: boolean;
    horzTouchDrag?: boolean;
    vertTouchDrag?: boolean;
  };
  handleScale?: {
    axisPressedMouseMove?: {
      time?: boolean;
      price?: boolean;
    };
    mouseWheel?: boolean;
    pinch?: boolean;
  };
}

// === INDICATORS ===

/**
 * Tipo de indicador
 */
export enum IndicatorType {
  // Trend
  EMA = 'ema',
  SMA = 'sma',
  WMA = 'wma',
  VWAP = 'vwap',

  // Oscillators
  RSI = 'rsi',
  MACD = 'macd',
  STOCHASTIC = 'stochastic',
  CCI = 'cci',

  // Volatility
  BOLLINGER_BANDS = 'bollinger_bands',
  ATR = 'atr',
  KELTNER_CHANNELS = 'keltner_channels',

  // Volume
  VOLUME = 'volume',
  OBV = 'obv',
  VOLUME_PROFILE = 'volume_profile',

  // Custom
  CUSTOM = 'custom',
}

/**
 * Configuración de indicador
 */
export interface IndicatorConfig {
  type: IndicatorType;
  enabled: boolean;
  color?: string;
  lineWidth?: number;
  params?: Record<string, any>;
  overlay?: boolean; // Si se superpone en el gráfico principal
}

/**
 * Indicador renderizado
 */
export interface RenderedIndicator {
  id: string;
  type: IndicatorType;
  config: IndicatorConfig;
  series: ChartSeries[];
  visible: boolean;
}

// === DRAWING TOOLS ===

/**
 * Tipo de herramienta de dibujo
 */
export enum DrawingToolType {
  TREND_LINE = 'trend_line',
  HORIZONTAL_LINE = 'horizontal_line',
  VERTICAL_LINE = 'vertical_line',
  RECTANGLE = 'rectangle',
  FIBONACCI = 'fibonacci',
  TEXT = 'text',
  ARROW = 'arrow',
  PRICE_LABEL = 'price_label',
}

/**
 * Herramienta de dibujo
 */
export interface DrawingTool {
  id: string;
  type: DrawingToolType;
  points: Array<{ time: number; price: number }>;
  color?: string;
  lineWidth?: number;
  style?: 'solid' | 'dashed' | 'dotted';
  text?: string;
  locked?: boolean;
  visible?: boolean;
}

// === ANNOTATIONS ===

/**
 * Marcador en el gráfico
 */
export interface ChartMarker {
  time: number;
  position: 'aboveBar' | 'belowBar' | 'inBar';
  color: string;
  shape: 'circle' | 'square' | 'arrowUp' | 'arrowDown';
  text?: string;
  size?: number;
}

/**
 * Anotación de precio
 */
export interface PriceAnnotation {
  price: number;
  time?: number;
  text: string;
  color?: string;
  backgroundColor?: string;
  borderColor?: string;
}

/**
 * Anotación de tiempo
 */
export interface TimeAnnotation {
  time: number;
  text: string;
  color?: string;
}

// === POSITIONS & TRADES ON CHART ===

/**
 * Posición en el gráfico
 */
export interface ChartPosition {
  id: string;
  entryTime: number;
  entryPrice: number;
  exitTime?: number;
  exitPrice?: number;
  side: 'long' | 'short';
  size: number;
  pnl?: number;
  stopLoss?: number;
  takeProfit?: number;
  markers?: ChartMarker[];
}

/**
 * Trade en el gráfico
 */
export interface ChartTrade {
  id: string;
  time: number;
  price: number;
  type: 'buy' | 'sell';
  size: number;
  marker?: ChartMarker;
}

// === VOLUME PROFILE ===

/**
 * Datos de perfil de volumen
 */
export interface VolumeProfileData {
  price: number;
  volume: number;
  buyVolume: number;
  sellVolume: number;
  delta: number;
}

/**
 * Configuración de perfil de volumen
 */
export interface VolumeProfileConfig {
  visible: boolean;
  orientation: 'horizontal' | 'vertical';
  width?: number;
  color?: string;
  buyColor?: string;
  sellColor?: string;
  showPOC?: boolean; // Point of Control
  showVAH?: boolean; // Value Area High
  showVAL?: boolean; // Value Area Low
}

// === CHART INTERACTIONS ===

/**
 * Evento de click en el gráfico
 */
export interface ChartClickEvent {
  time: number;
  price: number;
  seriesId?: string;
  point?: {
    x: number;
    y: number;
  };
}

/**
 * Evento de hover en el gráfico
 */
export interface ChartHoverEvent extends ChartClickEvent {
  dataIndex?: number;
}

/**
 * Rango visible del gráfico
 */
export interface VisibleRange {
  from: number;
  to: number;
}

/**
 * Opciones de zoom
 */
export interface ZoomOptions {
  level: number;
  center?: number;
  animated?: boolean;
}

// === PERFORMANCE ===

/**
 * Configuración de performance
 */
export interface PerformanceConfig {
  enableWebGL?: boolean;
  maxDataPoints?: number;
  updateFrequency?: number;
  throttleUpdates?: boolean;
  batchUpdates?: boolean;
  lazyLoading?: boolean;
  virtualScrolling?: boolean;
}

// === EXPORT ===

/**
 * Opciones de exportación
 */
export interface ChartExportOptions {
  format: 'png' | 'svg' | 'pdf';
  width?: number;
  height?: number;
  quality?: number;
  includeWatermark?: boolean;
  backgroundColor?: string;
}

// === CHART INSTANCE ===

/**
 * Métodos del chart instance
 */
export interface IChartInstance {
  // Data
  setData(data: ChartData[]): void;
  updateData(data: ChartData): void;
  clearData(): void;

  // Series
  addSeries(series: ChartSeries): void;
  removeSeries(seriesId: string): void;
  updateSeries(seriesId: string, data: unknown[]): void;

  // Indicators
  addIndicator(config: IndicatorConfig): void;
  removeIndicator(indicatorId: string): void;
  toggleIndicator(indicatorId: string, visible: boolean): void;

  // Drawing tools
  addDrawing(tool: DrawingTool): void;
  removeDrawing(toolId: string): void;
  clearDrawings(): void;

  // Markers & Annotations
  addMarker(marker: ChartMarker): void;
  removeMarker(time: number): void;
  addPriceAnnotation(annotation: PriceAnnotation): void;

  // View control
  fitContent(): void;
  scrollToTime(time: number): void;
  setVisibleRange(range: VisibleRange): void;
  zoom(options: ZoomOptions): void;

  // Events
  onClick(handler: (event: ChartClickEvent) => void): void;
  onHover(handler: (event: ChartHoverEvent) => void): void;
  onVisibleRangeChange(handler: (range: VisibleRange) => void): void;

  // Export
  export(options: ChartExportOptions): Promise<Blob>;
  screenshot(): Promise<string>; // Base64

  // Cleanup
  destroy(): void;
}

// === CHART DIMENSIONS ===

/**
 * Dimensiones del gráfico
 */
export interface ChartDimensions {
  width: number | string;
  height: number | string;
  candleWidth?: number;
}

/**
 * Dimensiones responsive
 */
export interface ResponsiveDimensions {
  mobile: ChartDimensions;
  tablet: ChartDimensions;
  desktop: ChartDimensions;
  fullscreen: ChartDimensions;
}
