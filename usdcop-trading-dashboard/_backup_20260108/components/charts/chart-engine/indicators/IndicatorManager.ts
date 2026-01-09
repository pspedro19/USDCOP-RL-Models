/**
 * Indicator Manager for ChartPro
 * Extensible plugin architecture for technical indicators
 */

import { IChartApi, ISeriesApi, CandlestickData, LineData, HistogramData, Time, DeepPartial, LineSeriesOptions, HistogramSeriesOptions } from 'lightweight-charts';
import * as TechnicalIndicators from 'technicalindicators';

export interface IndicatorConfig {
  id: string;
  name: string;
  type: IndicatorType;
  parameters: Record<string, any>;
  style: IndicatorStyle;
  enabled: boolean;
  overlay: boolean; // true for price overlays, false for separate pane
  paneHeight?: number; // height percentage for separate pane indicators
}

export interface IndicatorStyle {
  color: string;
  width: number;
  style: 'solid' | 'dashed' | 'dotted';
  opacity: number;
  displayName?: string;
  showLastValue?: boolean;
  precision?: number;
}

export type IndicatorType =
  | 'sma' | 'ema' | 'wma' | 'dema' | 'tema' | 'trima'
  | 'rsi' | 'macd' | 'stoch' | 'cci' | 'williams'
  | 'bollinger' | 'atr' | 'adx' | 'obv' | 'mfi'
  | 'ichimoku' | 'parabolic_sar' | 'stochastic_rsi'
  | 'awesome_oscillator' | 'momentum' | 'roc'
  | 'pivot_points' | 'fibonacci_retracement'
  | 'vwap' | 'volume_profile' | 'market_profile';

export interface IndicatorPlugin {
  id: string;
  name: string;
  description: string;
  parameters: IndicatorParameter[];
  calculate: (data: CandlestickData[], params: Record<string, any>) => IndicatorResult;
  defaultStyle: IndicatorStyle;
  isOverlay: boolean;
}

export interface IndicatorParameter {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'color' | 'select';
  defaultValue: any;
  min?: number;
  max?: number;
  options?: string[]; // for select type
  description: string;
}

export interface IndicatorResult {
  main?: LineData[];
  signal?: LineData[];
  histogram?: HistogramData[];
  bands?: {
    upper: LineData[];
    lower: LineData[];
    middle?: LineData[];
  };
  levels?: {
    name: string;
    value: number;
    color: string;
  }[];
}

export class IndicatorManager {
  private chart: IChartApi;
  private indicators: Map<string, IndicatorConfig> = new Map();
  private series: Map<string, ISeriesApi<any>> = new Map();
  private plugins: Map<string, IndicatorPlugin> = new Map();
  private indicatorPanes: Map<string, { api: IChartApi; height: number }> = new Map();

  constructor(chart: IChartApi) {
    this.chart = chart;
    this.registerBuiltInIndicators();
  }

  private registerBuiltInIndicators(): void {
    // Moving Averages
    this.registerPlugin({
      id: 'sma',
      name: 'Simple Moving Average',
      description: 'Simple Moving Average calculation',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 20,
          min: 1,
          max: 200,
          description: 'Period for SMA calculation'
        }
      ],
      calculate: (data: CandlestickData[], params: Record<string, any>) => {
        const closes = data.map(d => d.close);
        const smaValues = TechnicalIndicators.SMA.calculate({
          period: params.period,
          values: closes
        });

        const result: LineData[] = [];
        for (let i = 0; i < smaValues.length; i++) {
          result.push({
            time: data[i + params.period - 1].time,
            value: smaValues[i]
          });
        }

        return { main: result };
      },
      defaultStyle: {
        color: '#74b9ff',
        width: 2,
        style: 'solid',
        opacity: 1,
        showLastValue: true,
        precision: 4
      },
      isOverlay: true
    });

    this.registerPlugin({
      id: 'ema',
      name: 'Exponential Moving Average',
      description: 'Exponential Moving Average calculation',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 20,
          min: 1,
          max: 200,
          description: 'Period for EMA calculation'
        }
      ],
      calculate: (data: CandlestickData[], params: Record<string, any>) => {
        const closes = data.map(d => d.close);
        const emaValues = TechnicalIndicators.EMA.calculate({
          period: params.period,
          values: closes
        });

        const result: LineData[] = [];
        for (let i = 0; i < emaValues.length; i++) {
          result.push({
            time: data[i + params.period - 1].time,
            value: emaValues[i]
          });
        }

        return { main: result };
      },
      defaultStyle: {
        color: '#fdcb6e',
        width: 2,
        style: 'solid',
        opacity: 1,
        showLastValue: true,
        precision: 4
      },
      isOverlay: true
    });

    // RSI
    this.registerPlugin({
      id: 'rsi',
      name: 'Relative Strength Index',
      description: 'RSI oscillator (0-100)',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 14,
          min: 2,
          max: 50,
          description: 'Period for RSI calculation'
        }
      ],
      calculate: (data: CandlestickData[], params: Record<string, any>) => {
        const closes = data.map(d => d.close);
        const rsiValues = TechnicalIndicators.RSI.calculate({
          period: params.period,
          values: closes
        });

        const result: LineData[] = [];
        for (let i = 0; i < rsiValues.length; i++) {
          result.push({
            time: data[i + params.period].time,
            value: rsiValues[i]
          });
        }

        return {
          main: result,
          levels: [
            { name: 'Overbought', value: 70, color: '#ff6b6b' },
            { name: 'Oversold', value: 30, color: '#51cf66' }
          ]
        };
      },
      defaultStyle: {
        color: '#fd79a8',
        width: 2,
        style: 'solid',
        opacity: 1,
        showLastValue: true,
        precision: 2
      },
      isOverlay: false
    });

    // MACD
    this.registerPlugin({
      id: 'macd',
      name: 'MACD',
      description: 'Moving Average Convergence Divergence',
      parameters: [
        {
          name: 'fastPeriod',
          type: 'number',
          defaultValue: 12,
          min: 1,
          max: 50,
          description: 'Fast EMA period'
        },
        {
          name: 'slowPeriod',
          type: 'number',
          defaultValue: 26,
          min: 1,
          max: 100,
          description: 'Slow EMA period'
        },
        {
          name: 'signalPeriod',
          type: 'number',
          defaultValue: 9,
          min: 1,
          max: 50,
          description: 'Signal line period'
        }
      ],
      calculate: (data: CandlestickData[], params: Record<string, any>) => {
        const closes = data.map(d => d.close);
        const macdData = TechnicalIndicators.MACD.calculate({
          fastPeriod: params.fastPeriod,
          slowPeriod: params.slowPeriod,
          signalPeriod: params.signalPeriod,
          values: closes,
          SimpleMAOscillator: false,
          SimpleMASignal: false
        });

        const macdLine: LineData[] = [];
        const signalLine: LineData[] = [];
        const histogram: HistogramData[] = [];

        const startIndex = params.slowPeriod + params.signalPeriod - 2;

        for (let i = 0; i < macdData.length; i++) {
          const time = data[i + startIndex].time;
          const macdValue = macdData[i];

          macdLine.push({
            time,
            value: macdValue.MACD || 0
          });

          signalLine.push({
            time,
            value: macdValue.signal || 0
          });

          histogram.push({
            time,
            value: macdValue.histogram || 0,
            color: (macdValue.histogram || 0) >= 0 ? '#26a69a' : '#ef5350'
          });
        }

        return {
          main: macdLine,
          signal: signalLine,
          histogram
        };
      },
      defaultStyle: {
        color: '#55a3ff',
        width: 2,
        style: 'solid',
        opacity: 1,
        showLastValue: true,
        precision: 6
      },
      isOverlay: false
    });

    // Bollinger Bands
    this.registerPlugin({
      id: 'bollinger',
      name: 'Bollinger Bands',
      description: 'Bollinger Bands with standard deviation',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 20,
          min: 2,
          max: 100,
          description: 'Period for moving average'
        },
        {
          name: 'stdDev',
          type: 'number',
          defaultValue: 2,
          min: 0.1,
          max: 5,
          description: 'Standard deviation multiplier'
        }
      ],
      calculate: (data: CandlestickData[], params: Record<string, any>) => {
        const closes = data.map(d => d.close);
        const bbData = TechnicalIndicators.BollingerBands.calculate({
          period: params.period,
          stdDev: params.stdDev,
          values: closes
        });

        const upperBand: LineData[] = [];
        const lowerBand: LineData[] = [];
        const middleBand: LineData[] = [];

        for (let i = 0; i < bbData.length; i++) {
          const time = data[i + params.period - 1].time;
          const bb = bbData[i];

          upperBand.push({
            time,
            value: bb.upper
          });

          lowerBand.push({
            time,
            value: bb.lower
          });

          middleBand.push({
            time,
            value: bb.middle
          });
        }

        return {
          bands: {
            upper: upperBand,
            lower: lowerBand,
            middle: middleBand
          }
        };
      },
      defaultStyle: {
        color: '#00cec9',
        width: 1,
        style: 'solid',
        opacity: 0.8,
        showLastValue: true,
        precision: 4
      },
      isOverlay: true
    });

    // ATR
    this.registerPlugin({
      id: 'atr',
      name: 'Average True Range',
      description: 'Volatility indicator',
      parameters: [
        {
          name: 'period',
          type: 'number',
          defaultValue: 14,
          min: 1,
          max: 100,
          description: 'Period for ATR calculation'
        }
      ],
      calculate: (data: CandlestickData[], params: Record<string, any>) => {
        const atrData = TechnicalIndicators.ATR.calculate({
          period: params.period,
          high: data.map(d => d.high),
          low: data.map(d => d.low),
          close: data.map(d => d.close)
        });

        const result: LineData[] = [];
        for (let i = 0; i < atrData.length; i++) {
          result.push({
            time: data[i + params.period - 1].time,
            value: atrData[i]
          });
        }

        return { main: result };
      },
      defaultStyle: {
        color: '#a29bfe',
        width: 2,
        style: 'solid',
        opacity: 1,
        showLastValue: true,
        precision: 4
      },
      isOverlay: false
    });
  }

  public registerPlugin(plugin: IndicatorPlugin): void {
    this.plugins.set(plugin.id, plugin);
  }

  public addIndicator(config: Partial<IndicatorConfig>): string {
    const plugin = this.plugins.get(config.type!);
    if (!plugin) {
      throw new Error(`Indicator plugin not found: ${config.type}`);
    }

    const id = config.id || `${config.type}_${Date.now()}`;
    const fullConfig: IndicatorConfig = {
      id,
      name: config.name || plugin.name,
      type: config.type!,
      parameters: { ...this.getDefaultParameters(plugin), ...config.parameters },
      style: { ...plugin.defaultStyle, ...config.style },
      enabled: config.enabled !== false,
      overlay: config.overlay !== undefined ? config.overlay : plugin.isOverlay,
      paneHeight: config.paneHeight || 25
    };

    this.indicators.set(id, fullConfig);

    if (fullConfig.enabled) {
      this.calculateAndRenderIndicator(id);
    }

    return id;
  }

  public removeIndicator(id: string): void {
    const config = this.indicators.get(id);
    if (!config) return;

    // Remove series
    const series = this.series.get(id);
    if (series) {
      this.chart.removeSeries(series);
      this.series.delete(id);
    }

    // Remove additional series for complex indicators
    const additionalKeys = Array.from(this.series.keys()).filter(key => key.startsWith(`${id}_`));
    additionalKeys.forEach(key => {
      const additionalSeries = this.series.get(key);
      if (additionalSeries) {
        this.chart.removeSeries(additionalSeries);
        this.series.delete(key);
      }
    });

    this.indicators.delete(id);
  }

  public updateIndicator(id: string, data: LineData[]): void {
    const series = this.series.get(id);
    if (series) {
      series.setData(data);
    }
  }

  public updateIndicatorConfig(id: string, config: Partial<IndicatorConfig>): void {
    const currentConfig = this.indicators.get(id);
    if (!currentConfig) return;

    const newConfig = { ...currentConfig, ...config };
    this.indicators.set(id, newConfig);

    // Remove and recreate series if style changed
    if (config.style || config.parameters) {
      this.removeIndicator(id);
      this.indicators.set(id, newConfig);
      if (newConfig.enabled) {
        this.calculateAndRenderIndicator(id);
      }
    }
  }

  public toggleIndicator(id: string): void {
    const config = this.indicators.get(id);
    if (!config) return;

    config.enabled = !config.enabled;

    if (config.enabled) {
      this.calculateAndRenderIndicator(id);
    } else {
      const series = this.series.get(id);
      if (series) {
        this.chart.removeSeries(series);
        this.series.delete(id);
      }
    }
  }

  public calculateIndicator(id: string, data: CandlestickData[]): IndicatorResult | null {
    const config = this.indicators.get(id);
    if (!config) return null;

    const plugin = this.plugins.get(config.type);
    if (!plugin) return null;

    try {
      return plugin.calculate(data, config.parameters);
    } catch (error) {
      console.error(`Error calculating indicator ${id}:`, error);
      return null;
    }
  }

  private calculateAndRenderIndicator(id: string): void {
    const config = this.indicators.get(id);
    if (!config) return;

    const plugin = this.plugins.get(config.type);
    if (!plugin) return;

    // This would need to get actual chart data
    // For now, using mock data
    const mockData: CandlestickData[] = []; // Get from chart

    const result = this.calculateIndicator(id, mockData);
    if (!result) return;

    this.renderIndicatorResult(id, config, result);
  }

  private renderIndicatorResult(id: string, config: IndicatorConfig, result: IndicatorResult): void {
    // Main line
    if (result.main) {
      const series = this.chart.addLineSeries({
        color: config.style.color,
        lineWidth: config.style.width,
        lineStyle: this.getLineStyle(config.style.style),
        priceLineVisible: config.style.showLastValue || false,
        lastValueVisible: config.style.showLastValue || false,
        title: config.style.displayName || config.name,
        priceFormat: {
          type: 'price',
          precision: config.style.precision || 4,
          minMove: 1 / Math.pow(10, config.style.precision || 4)
        }
      });

      series.setData(result.main);
      this.series.set(id, series);
    }

    // Signal line (for MACD, etc.)
    if (result.signal) {
      const signalSeries = this.chart.addLineSeries({
        color: this.adjustColor(config.style.color, -0.3),
        lineWidth: config.style.width,
        lineStyle: this.getLineStyle('dashed'),
        priceLineVisible: false,
        lastValueVisible: config.style.showLastValue || false,
        title: `${config.name} Signal`
      });

      signalSeries.setData(result.signal);
      this.series.set(`${id}_signal`, signalSeries);
    }

    // Histogram (for MACD, etc.)
    if (result.histogram) {
      const histogramSeries = this.chart.addHistogramSeries({
        priceLineVisible: false,
        lastValueVisible: false,
        title: `${config.name} Histogram`
      });

      histogramSeries.setData(result.histogram);
      this.series.set(`${id}_histogram`, histogramSeries);
    }

    // Bands (for Bollinger Bands, etc.)
    if (result.bands) {
      if (result.bands.upper) {
        const upperSeries = this.chart.addLineSeries({
          color: config.style.color,
          lineWidth: 1,
          lineStyle: this.getLineStyle('solid'),
          priceLineVisible: false,
          lastValueVisible: false,
          title: `${config.name} Upper`
        });

        upperSeries.setData(result.bands.upper);
        this.series.set(`${id}_upper`, upperSeries);
      }

      if (result.bands.lower) {
        const lowerSeries = this.chart.addLineSeries({
          color: config.style.color,
          lineWidth: 1,
          lineStyle: this.getLineStyle('solid'),
          priceLineVisible: false,
          lastValueVisible: false,
          title: `${config.name} Lower`
        });

        lowerSeries.setData(result.bands.lower);
        this.series.set(`${id}_lower`, lowerSeries);
      }

      if (result.bands.middle) {
        const middleSeries = this.chart.addLineSeries({
          color: this.adjustColor(config.style.color, 0.2),
          lineWidth: 1,
          lineStyle: this.getLineStyle('dashed'),
          priceLineVisible: false,
          lastValueVisible: false,
          title: `${config.name} Middle`
        });

        middleSeries.setData(result.bands.middle);
        this.series.set(`${id}_middle`, middleSeries);
      }
    }
  }

  private getDefaultParameters(plugin: IndicatorPlugin): Record<string, any> {
    const params: Record<string, any> = {};
    plugin.parameters.forEach(param => {
      params[param.name] = param.defaultValue;
    });
    return params;
  }

  private getLineStyle(style: 'solid' | 'dashed' | 'dotted'): number {
    switch (style) {
      case 'dashed':
        return 1;
      case 'dotted':
        return 2;
      default:
        return 0;
    }
  }

  private adjustColor(color: string, factor: number): string {
    // Simple color adjustment (would be more sophisticated in production)
    return color;
  }

  // Public methods
  public getIndicators(): IndicatorConfig[] {
    return Array.from(this.indicators.values());
  }

  public getAvailablePlugins(): IndicatorPlugin[] {
    return Array.from(this.plugins.values());
  }

  public getIndicatorConfig(id: string): IndicatorConfig | undefined {
    return this.indicators.get(id);
  }

  public exportIndicators(): IndicatorConfig[] {
    return this.getIndicators();
  }

  public importIndicators(configs: IndicatorConfig[]): void {
    configs.forEach(config => {
      this.indicators.set(config.id, config);
      if (config.enabled) {
        this.calculateAndRenderIndicator(config.id);
      }
    });
  }

  public destroy(): void {
    // Remove all series
    this.series.forEach(series => {
      this.chart.removeSeries(series);
    });

    this.indicators.clear();
    this.series.clear();
    this.plugins.clear();
  }
}

export default IndicatorManager;