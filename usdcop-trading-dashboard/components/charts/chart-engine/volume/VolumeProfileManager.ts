/**
 * Volume Profile Manager for ChartPro
 * Advanced volume analysis with Point of Control (POC) and Value Area
 */

import { IChartApi, ISeriesApi, CandlestickData, HistogramData, Time } from 'lightweight-charts';
import * as echarts from 'echarts';

export interface VolumeProfileLevel {
  price: number;
  volume: number;
  percentage: number;
  isPOC: boolean;
  isValueAreaHigh: boolean;
  isValueAreaLow: boolean;
  isInValueArea: boolean;
}

export interface VolumeProfileData {
  levels: VolumeProfileLevel[];
  poc: number; // Point of Control
  valueAreaHigh: number;
  valueAreaLow: number;
  totalVolume: number;
  valueAreaVolume: number;
  valueAreaPercentage: number;
}

export interface VolumeProfileConfig {
  numberOfLevels: number;
  valueAreaPercentage: number; // Default 70%
  showPOC: boolean;
  showValueArea: boolean;
  showVolumeNumbers: boolean;
  position: 'left' | 'right';
  width: number; // Width as percentage of chart
  opacity: number;
  colors: {
    poc: string;
    valueArea: string;
    normalVolume: string;
    background: string;
    text: string;
  };
}

export class VolumeProfileManager {
  private chart: IChartApi;
  private container: HTMLElement;
  private echartsInstance: echarts.EChartsOption | null = null;
  private profileContainer: HTMLDivElement | null = null;
  private currentData: VolumeProfileData | null = null;
  private config: VolumeProfileConfig;
  private isVisible = false;

  private defaultConfig: VolumeProfileConfig = {
    numberOfLevels: 50,
    valueAreaPercentage: 70,
    showPOC: true,
    showValueArea: true,
    showVolumeNumbers: true,
    position: 'right',
    width: 20, // 20% of chart width
    opacity: 0.8,
    colors: {
      poc: '#ff6b6b',
      valueArea: 'rgba(74, 144, 226, 0.3)',
      normalVolume: 'rgba(116, 185, 255, 0.6)',
      background: 'rgba(0, 0, 0, 0.1)',
      text: '#ffffff'
    }
  };

  constructor(chart: IChartApi, container: HTMLElement, config?: Partial<VolumeProfileConfig>) {
    this.chart = chart;
    this.container = container;
    this.config = { ...this.defaultConfig, ...config };

    this.createProfileContainer();
    this.setupEventListeners();
  }

  private createProfileContainer(): void {
    this.profileContainer = document.createElement('div');
    this.profileContainer.style.position = 'absolute';
    this.profileContainer.style.top = '0';
    this.profileContainer.style.height = '100%';
    this.profileContainer.style.width = `${this.config.width}%`;
    this.profileContainer.style.zIndex = '5';
    this.profileContainer.style.pointerEvents = 'none';
    this.profileContainer.style.display = 'none';

    if (this.config.position === 'right') {
      this.profileContainer.style.right = '0';
    } else {
      this.profileContainer.style.left = '0';
    }

    this.container.appendChild(this.profileContainer);

    // Initialize ECharts
    this.echartsInstance = echarts.init(this.profileContainer, 'dark');
  }

  private setupEventListeners(): void {
    // Update on chart size changes
    const resizeObserver = new ResizeObserver(() => {
      if (this.echartsInstance && this.isVisible) {
        this.echartsInstance.resize();
      }
    });

    resizeObserver.observe(this.container);

    // Update on visible range changes
    this.chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      if (this.isVisible) {
        this.updateVolumeProfile();
      }
    });
  }

  public calculateVolumeProfile(candleData: CandlestickData[], volumeData?: HistogramData[]): VolumeProfileData {
    if (!candleData.length) {
      return {
        levels: [],
        poc: 0,
        valueAreaHigh: 0,
        valueAreaLow: 0,
        totalVolume: 0,
        valueAreaVolume: 0,
        valueAreaPercentage: 0
      };
    }

    // Find price range
    const prices = candleData.flatMap(candle => [candle.high, candle.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    const levelSize = priceRange / this.config.numberOfLevels;

    // Initialize volume levels
    const volumeLevels: Map<number, number> = new Map();

    // Calculate volume for each price level
    candleData.forEach((candle, index) => {
      const volume = volumeData?.[index]?.value ||
                    (candle.high - candle.low) * 1000; // Estimate volume if not provided

      // Distribute volume across price levels within the candle
      const candleRange = candle.high - candle.low;
      const numLevelsInCandle = Math.max(1, Math.ceil(candleRange / levelSize));

      for (let i = 0; i < numLevelsInCandle; i++) {
        const levelPrice = candle.low + (candleRange * i / numLevelsInCandle);
        const levelIndex = Math.floor((levelPrice - minPrice) / levelSize);
        const levelKey = minPrice + (levelIndex * levelSize);

        const existingVolume = volumeLevels.get(levelKey) || 0;
        volumeLevels.set(levelKey, existingVolume + (volume / numLevelsInCandle));
      }
    });

    // Convert to array and sort by price
    const levels = Array.from(volumeLevels.entries())
      .map(([price, volume]) => ({ price, volume }))
      .sort((a, b) => a.price - b.price);

    const totalVolume = levels.reduce((sum, level) => sum + level.volume, 0);

    // Find Point of Control (highest volume level)
    const poc = levels.reduce((max, level) =>
      level.volume > max.volume ? level : max
    );

    // Calculate Value Area (default 70% of volume)
    const targetVolumeArea = totalVolume * (this.config.valueAreaPercentage / 100);
    const valueArea = this.calculateValueArea(levels, poc, targetVolumeArea);

    // Create final level data
    const profileLevels: VolumeProfileLevel[] = levels.map(level => ({
      price: level.price,
      volume: level.volume,
      percentage: (level.volume / totalVolume) * 100,
      isPOC: level.price === poc.price,
      isValueAreaHigh: level.price === valueArea.high,
      isValueAreaLow: level.price === valueArea.low,
      isInValueArea: level.price >= valueArea.low && level.price <= valueArea.high
    }));

    return {
      levels: profileLevels,
      poc: poc.price,
      valueAreaHigh: valueArea.high,
      valueAreaLow: valueArea.low,
      totalVolume,
      valueAreaVolume: valueArea.volume,
      valueAreaPercentage: (valueArea.volume / totalVolume) * 100
    };
  }

  private calculateValueArea(
    levels: Array<{ price: number; volume: number }>,
    poc: { price: number; volume: number },
    targetVolume: number
  ): { high: number; low: number; volume: number } {
    // Start from POC and expand up and down until we reach target volume
    const pocIndex = levels.findIndex(level => level.price === poc.price);

    let currentVolume = poc.volume;
    let highIndex = pocIndex;
    let lowIndex = pocIndex;

    while (currentVolume < targetVolume && (highIndex < levels.length - 1 || lowIndex > 0)) {
      const upVolume = highIndex < levels.length - 1 ? levels[highIndex + 1].volume : 0;
      const downVolume = lowIndex > 0 ? levels[lowIndex - 1].volume : 0;

      if (upVolume >= downVolume && highIndex < levels.length - 1) {
        highIndex++;
        currentVolume += upVolume;
      } else if (lowIndex > 0) {
        lowIndex--;
        currentVolume += downVolume;
      } else {
        break;
      }
    }

    return {
      high: levels[highIndex].price,
      low: levels[lowIndex].price,
      volume: currentVolume
    };
  }

  private renderVolumeProfile(data: VolumeProfileData): void {
    if (!this.echartsInstance || !data.levels.length) return;

    const maxVolume = Math.max(...data.levels.map(level => level.volume));

    const seriesData = data.levels.map(level => [
      level.volume / maxVolume * 100, // Normalize to percentage for horizontal bar
      level.price,
      level.volume,
      level.isPOC,
      level.isInValueArea
    ]);

    const option: echarts.EChartsOption = {
      animation: false,
      grid: {
        left: '5%',
        right: '5%',
        top: '0%',
        bottom: '0%',
        containLabel: false
      },
      xAxis: {
        type: 'value',
        show: false,
        min: 0,
        max: 100
      },
      yAxis: {
        type: 'value',
        show: false,
        min: Math.min(...data.levels.map(l => l.price)),
        max: Math.max(...data.levels.map(l => l.price))
      },
      series: [
        {
          type: 'bar',
          data: seriesData,
          itemStyle: {
            color: (params: any) => {
              const [, , , isPOC, isInValueArea] = params.value;
              if (isPOC) return this.config.colors.poc;
              if (isInValueArea) return this.config.colors.valueArea;
              return this.config.colors.normalVolume;
            },
            opacity: this.config.opacity
          },
          label: this.config.showVolumeNumbers ? {
            show: true,
            position: this.config.position === 'right' ? 'left' : 'right',
            formatter: (params: any) => {
              const volume = params.value[2];
              return this.formatVolume(volume);
            },
            color: this.config.colors.text,
            fontSize: 10
          } : undefined,
          barMaxWidth: 20,
          coordinateSystem: 'cartesian2d'
        }
      ],
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const [percentage, price, volume, isPOC, isInValueArea] = params.value;
          let tooltip = `Price: ${price.toFixed(4)}<br/>`;
          tooltip += `Volume: ${this.formatVolume(volume)}<br/>`;
          tooltip += `Percentage: ${percentage.toFixed(2)}%<br/>`;

          if (isPOC) tooltip += '<br/><strong>Point of Control</strong>';
          if (isInValueArea) tooltip += '<br/>In Value Area';

          return tooltip;
        },
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        borderColor: '#333',
        textStyle: {
          color: '#fff'
        }
      }
    };

    this.echartsInstance.setOption(option, true);

    // Add POC and Value Area lines
    if (this.config.showPOC || this.config.showValueArea) {
      this.addPriceLines(data);
    }
  }

  private addPriceLines(data: VolumeProfileData): void {
    // This would integrate with the main chart to show horizontal lines
    // for POC and Value Area boundaries

    if (this.config.showPOC) {
      // Add POC line to main chart
      this.addHorizontalLine(data.poc, this.config.colors.poc, 'POC');
    }

    if (this.config.showValueArea) {
      // Add Value Area boundaries
      this.addHorizontalLine(data.valueAreaHigh, this.config.colors.valueArea, 'VA High');
      this.addHorizontalLine(data.valueAreaLow, this.config.colors.valueArea, 'VA Low');
    }
  }

  private addHorizontalLine(price: number, color: string, label: string): void {
    // This would need integration with the main TradingView chart
    // to add horizontal price lines
    console.log(`Adding horizontal line at ${price} with color ${color} and label ${label}`);
  }

  private formatVolume(volume: number): string {
    if (volume >= 1e9) {
      return `${(volume / 1e9).toFixed(2)}B`;
    } else if (volume >= 1e6) {
      return `${(volume / 1e6).toFixed(2)}M`;
    } else if (volume >= 1e3) {
      return `${(volume / 1e3).toFixed(2)}K`;
    }
    return volume.toFixed(0);
  }

  private updateVolumeProfile(): void {
    if (!this.isVisible) return;

    // Get visible data from the main chart
    const visibleRange = this.chart.timeScale().getVisibleRange();
    if (!visibleRange) return;

    // This would need to be implemented to get the actual candle and volume data
    // for the visible range from the main chart

    // For now, we'll use mock data
    // In a real implementation, you'd get this from the chart series

    // const candleData = this.getVisibleCandleData(visibleRange);
    // const volumeData = this.getVisibleVolumeData(visibleRange);

    // const profileData = this.calculateVolumeProfile(candleData, volumeData);
    // this.currentData = profileData;
    // this.renderVolumeProfile(profileData);
  }

  // Public methods
  public show(): void {
    if (this.profileContainer) {
      this.profileContainer.style.display = 'block';
      this.isVisible = true;
      this.updateVolumeProfile();
    }
  }

  public hide(): void {
    if (this.profileContainer) {
      this.profileContainer.style.display = 'none';
      this.isVisible = false;
    }
  }

  public toggle(): void {
    if (this.isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  public updateConfig(newConfig: Partial<VolumeProfileConfig>): void {
    this.config = { ...this.config, ...newConfig };

    if (this.profileContainer) {
      this.profileContainer.style.width = `${this.config.width}%`;

      if (this.config.position === 'right') {
        this.profileContainer.style.right = '0';
        this.profileContainer.style.left = 'auto';
      } else {
        this.profileContainer.style.left = '0';
        this.profileContainer.style.right = 'auto';
      }
    }

    if (this.isVisible && this.currentData) {
      this.renderVolumeProfile(this.currentData);
    }
  }

  public setData(candleData: CandlestickData[], volumeData?: HistogramData[]): void {
    const profileData = this.calculateVolumeProfile(candleData, volumeData);
    this.currentData = profileData;

    if (this.isVisible) {
      this.renderVolumeProfile(profileData);
    }
  }

  public getCurrentData(): VolumeProfileData | null {
    return this.currentData;
  }

  public exportImage(): Promise<string> {
    return new Promise((resolve, reject) => {
      if (!this.echartsInstance) {
        reject(new Error('ECharts instance not available'));
        return;
      }

      try {
        const imageData = this.echartsInstance.getDataURL({
          type: 'png',
          backgroundColor: 'transparent',
          pixelRatio: 2
        });
        resolve(imageData);
      } catch (error) {
        reject(error);
      }
    });
  }

  public destroy(): void {
    if (this.echartsInstance) {
      this.echartsInstance.dispose();
      this.echartsInstance = null;
    }

    if (this.profileContainer && this.profileContainer.parentNode) {
      this.profileContainer.parentNode.removeChild(this.profileContainer);
      this.profileContainer = null;
    }

    this.currentData = null;
  }
}

export default VolumeProfileManager;