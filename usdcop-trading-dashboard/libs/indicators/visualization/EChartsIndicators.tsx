/**
 * ECharts Indicator Visualizations
 * ===============================
 *
 * Professional indicator visualizations using Apache ECharts
 * with advanced overlays and interactive features.
 */

'use client';

import React, { useRef, useEffect, useMemo } from 'react';
import * as echarts from 'echarts';
import { CandleData, VolumeProfile, CorrelationMatrix, ChartVisualization } from '../types';

interface EChartsIndicatorProps {
  data: any[];
  type: 'line' | 'area' | 'histogram' | 'scatter' | 'heatmap' | 'volume_profile';
  config: {
    colors: string[];
    opacity?: number;
    thickness?: number;
    smooth?: boolean;
    fill?: boolean;
    title?: string;
    yAxis?: {
      min?: number;
      max?: number;
      splitLine?: boolean;
    };
  };
  overlays?: {
    levels: number[];
    zones: { min: number; max: number; color: string; label?: string }[];
    annotations: { x: number; y: number; text: string }[];
  };
  height?: number;
  onDataZoom?: (params: any) => void;
  onBrush?: (params: any) => void;
}

export const EChartsIndicator: React.FC<EChartsIndicatorProps> = ({
  data,
  type,
  config,
  overlays,
  height = 300,
  onDataZoom,
  onBrush
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts>();

  const chartOptions = useMemo(() => {
    const baseOptions: echarts.EChartsOption = {
      title: {
        text: config.title,
        textStyle: {
          fontSize: 14,
          fontWeight: 'normal',
          color: '#374151'
        }
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        borderColor: '#333',
        textStyle: { color: '#fff' },
        formatter: (params: any) => {
          if (!Array.isArray(params)) params = [params];

          let tooltip = `<div style="font-size: 12px;">`;
          tooltip += `<div style="margin-bottom: 4px;">${new Date(params[0].axisValue).toLocaleString()}</div>`;

          params.forEach((param: any) => {
            const marker = `<span style="display:inline-block;margin-right:5px;border-radius:10px;width:10px;height:10px;background-color:${param.color};"></span>`;
            tooltip += `<div>${marker}${param.seriesName}: ${typeof param.value === 'number' ? param.value.toFixed(4) : param.value}</div>`;
          });

          tooltip += `</div>`;
          return tooltip;
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'time',
        boundaryGap: false,
        axisLine: { lineStyle: { color: '#d1d5db' } },
        axisTick: { lineStyle: { color: '#d1d5db' } },
        axisLabel: { color: '#6b7280' },
        splitLine: { show: false }
      },
      yAxis: {
        type: 'value',
        scale: true,
        min: config.yAxis?.min,
        max: config.yAxis?.max,
        axisLine: { lineStyle: { color: '#d1d5db' } },
        axisTick: { lineStyle: { color: '#d1d5db' } },
        axisLabel: { color: '#6b7280' },
        splitLine: {
          show: config.yAxis?.splitLine !== false,
          lineStyle: { color: '#f3f4f6', type: 'dashed' }
        }
      },
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0],
          filterMode: 'filter'
        },
        {
          type: 'slider',
          xAxisIndex: [0],
          filterMode: 'filter',
          height: 20,
          bottom: 0
        }
      ],
      brush: {
        toolbox: ['rect', 'polygon', 'clear'],
        xAxisIndex: 0
      },
      series: []
    };

    // Generate series based on chart type
    switch (type) {
      case 'line':
        baseOptions.series = data.map((series, index) => ({
          name: series.name || `Series ${index + 1}`,
          type: 'line',
          data: series.data.map((point: any) => [point.timestamp, point.value]),
          lineStyle: {
            color: config.colors[index % config.colors.length],
            width: config.thickness || 2
          },
          smooth: config.smooth || false,
          symbol: 'none',
          emphasis: { focus: 'series' }
        }));
        break;

      case 'area':
        baseOptions.series = data.map((series, index) => ({
          name: series.name || `Series ${index + 1}`,
          type: 'line',
          data: series.data.map((point: any) => [point.timestamp, point.value]),
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: config.colors[index % config.colors.length] + '80' },
                { offset: 1, color: config.colors[index % config.colors.length] + '20' }
              ]
            }
          },
          lineStyle: {
            color: config.colors[index % config.colors.length],
            width: config.thickness || 2
          },
          smooth: config.smooth || false,
          symbol: 'none'
        }));
        break;

      case 'histogram':
        baseOptions.series = [{
          name: 'Histogram',
          type: 'bar',
          data: data.map((point: any) => [point.timestamp, point.value]),
          itemStyle: {
            color: (params: any) => {
              const value = params.value[1];
              return value >= 0 ? config.colors[0] : config.colors[1] || '#ef4444';
            }
          },
          barWidth: '80%'
        }];
        break;

      case 'scatter':
        baseOptions.series = [{
          name: 'Scatter',
          type: 'scatter',
          data: data.map((point: any) => [point.timestamp, point.value]),
          symbolSize: config.thickness || 6,
          itemStyle: {
            color: config.colors[0] || '#3b82f6'
          }
        }];
        break;

      case 'heatmap':
        // Heatmap implementation for correlation matrix
        const heatmapData = data.flatMap((row: any, i: number) =>
          row.map((value: number, j: number) => [i, j, value])
        );

        baseOptions.xAxis = {
          type: 'category',
          data: data.map((_: any, i: number) => `Asset ${i + 1}`),
          splitArea: { show: true }
        };

        baseOptions.yAxis = {
          type: 'category',
          data: data.map((_: any, i: number) => `Asset ${i + 1}`),
          splitArea: { show: true }
        };

        baseOptions.series = [{
          name: 'Correlation',
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: (params: any) => params.value[2].toFixed(2)
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }];

        baseOptions.visualMap = {
          min: -1,
          max: 1,
          calculable: true,
          orient: 'horizontal',
          left: 'center',
          bottom: '5%',
          inRange: {
            color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
          }
        };
        break;

      case 'volume_profile':
        // Volume profile visualization
        const profileData = data as VolumeProfile;

        baseOptions.xAxis = {
          type: 'value',
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { show: false },
          splitLine: { show: false }
        };

        baseOptions.yAxis = {
          type: 'value',
          axisLine: { lineStyle: { color: '#d1d5db' } },
          axisTick: { lineStyle: { color: '#d1d5db' } },
          axisLabel: { color: '#6b7280' },
          splitLine: { show: false }
        };

        baseOptions.series = [
          {
            name: 'Volume Profile',
            type: 'bar',
            data: profileData.levels.map(level => [level.volume, level.price]),
            barGap: 0,
            itemStyle: {
              color: (params: any) => {
                const price = params.value[1];
                if (price === profileData.poc) return '#fbbf24'; // POC in amber
                if (price >= profileData.valueAreaLow && price <= profileData.valueAreaHigh) {
                  return '#60a5fa'; // Value area in blue
                }
                return '#9ca3af'; // Other levels in gray
              }
            }
          }
        ];

        // Add POC and Value Area markings
        baseOptions.graphic = [
          {
            type: 'line',
            shape: {
              x1: 0, y1: profileData.poc,
              x2: Math.max(...profileData.levels.map(l => l.volume)), y2: profileData.poc
            },
            style: {
              stroke: '#f59e0b',
              lineWidth: 2,
              lineDash: [5, 5]
            }
          },
          {
            type: 'rect',
            shape: {
              x: 0,
              y: profileData.valueAreaLow,
              width: Math.max(...profileData.levels.map(l => l.volume)),
              height: profileData.valueAreaHigh - profileData.valueAreaLow
            },
            style: {
              fill: 'rgba(96, 165, 250, 0.1)',
              stroke: '#3b82f6',
              lineWidth: 1
            }
          }
        ];
        break;
    }

    // Add overlays
    if (overlays) {
      const markLines: any[] = [];
      const markAreas: any[] = [];

      // Add level lines
      overlays.levels.forEach(level => {
        markLines.push({
          yAxis: level,
          lineStyle: {
            color: '#6b7280',
            type: 'dashed'
          },
          label: {
            formatter: level.toFixed(4),
            position: 'end'
          }
        });
      });

      // Add zones
      overlays.zones.forEach(zone => {
        markAreas.push({
          yAxis: [zone.min, zone.max],
          itemStyle: {
            color: zone.color + '20'
          },
          label: {
            show: !!zone.label,
            formatter: zone.label
          }
        });
      });

      // Add marklines and areas to first series
      if (baseOptions.series && baseOptions.series.length > 0) {
        (baseOptions.series[0] as any).markLine = {
          data: markLines,
          silent: true
        };
        (baseOptions.series[0] as any).markArea = {
          data: markAreas,
          silent: true
        };
      }
    }

    return baseOptions;
  }, [data, type, config, overlays]);

  useEffect(() => {
    if (!chartRef.current) return;

    // Initialize chart
    chartInstance.current = echarts.init(chartRef.current, 'light', {
      renderer: 'canvas',
      useDirtyRect: true
    });

    // Add event listeners
    if (onDataZoom) {
      chartInstance.current.on('dataZoom', onDataZoom);
    }
    if (onBrush) {
      chartInstance.current.on('brush', onBrush);
    }

    return () => {
      chartInstance.current?.dispose();
    };
  }, [onDataZoom, onBrush]);

  useEffect(() => {
    if (chartInstance.current) {
      chartInstance.current.setOption(chartOptions, true);
    }
  }, [chartOptions]);

  useEffect(() => {
    const handleResize = () => {
      chartInstance.current?.resize();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div
      ref={chartRef}
      style={{ width: '100%', height: `${height}px` }}
      className="echarts-indicator"
    />
  );
};

// Specialized Volume Profile Chart
interface VolumeProfileChartProps {
  data: VolumeProfile;
  height?: number;
  showDelta?: boolean;
  colorScheme?: 'default' | 'heatmap' | 'institutional';
}

export const VolumeProfileChart: React.FC<VolumeProfileChartProps> = ({
  data,
  height = 400,
  showDelta = true,
  colorScheme = 'default'
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts>();

  const chartOptions = useMemo(() => {
    const colors = {
      default: { volume: '#3b82f6', poc: '#f59e0b', valueArea: '#10b981' },
      heatmap: { volume: '#8b5cf6', poc: '#ef4444', valueArea: '#06b6d4' },
      institutional: { volume: '#1f2937', poc: '#dc2626', valueArea: '#059669' }
    };

    const scheme = colors[colorScheme];

    return {
      title: {
        text: 'Volume Profile Analysis',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'bold' }
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const level = data.levels.find(l => l.price === params.value[1]);
          if (!level) return '';

          return `
            <div style="font-size: 12px;">
              <div><strong>Price:</strong> ${level.price.toFixed(4)}</div>
              <div><strong>Volume:</strong> ${level.volume.toLocaleString()}</div>
              <div><strong>% of Total:</strong> ${level.percentOfTotal.toFixed(2)}%</div>
              ${showDelta ? `<div><strong>Delta:</strong> ${level.delta.toFixed(0)}</div>` : ''}
              <div><strong>Buy Volume:</strong> ${level.buyVolume.toLocaleString()}</div>
              <div><strong>Sell Volume:</strong> ${level.sellVolume.toLocaleString()}</div>
            </div>
          `;
        }
      },
      grid: {
        left: '10%',
        right: '10%',
        top: '15%',
        bottom: '10%'
      },
      xAxis: {
        type: 'value',
        name: 'Volume',
        nameLocation: 'middle',
        nameGap: 30,
        axisLabel: {
          formatter: (value: number) => (value / 1000).toFixed(0) + 'K'
        }
      },
      yAxis: {
        type: 'value',
        name: 'Price',
        nameLocation: 'middle',
        nameGap: 50,
        axisLabel: {
          formatter: (value: number) => value.toFixed(4)
        }
      },
      series: [
        {
          name: 'Volume Profile',
          type: 'bar',
          data: data.levels.map(level => [level.volume, level.price]),
          barGap: 0,
          itemStyle: {
            color: (params: any) => {
              const price = params.value[1];
              if (price === data.poc) return scheme.poc;
              if (price >= data.valueAreaLow && price <= data.valueAreaHigh) {
                return scheme.valueArea;
              }
              return scheme.volume + '80';
            }
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ],
      graphic: [
        // POC line
        {
          type: 'line',
          shape: {
            x1: '10%', y1: data.poc,
            x2: '90%', y2: data.poc
          },
          style: {
            stroke: scheme.poc,
            lineWidth: 3,
            lineDash: [10, 5]
          }
        },
        // Value Area High line
        {
          type: 'line',
          shape: {
            x1: '10%', y1: data.valueAreaHigh,
            x2: '90%', y2: data.valueAreaHigh
          },
          style: {
            stroke: scheme.valueArea,
            lineWidth: 2,
            lineDash: [5, 5]
          }
        },
        // Value Area Low line
        {
          type: 'line',
          shape: {
            x1: '10%', y1: data.valueAreaLow,
            x2: '90%', y2: data.valueAreaLow
          },
          style: {
            stroke: scheme.valueArea,
            lineWidth: 2,
            lineDash: [5, 5]
          }
        },
        // POC label
        {
          type: 'text',
          left: '92%',
          top: data.poc,
          style: {
            text: `POC: ${data.poc.toFixed(4)}`,
            fontSize: 12,
            fontWeight: 'bold',
            fill: scheme.poc
          }
        },
        // Value Area labels
        {
          type: 'text',
          left: '92%',
          top: data.valueAreaHigh,
          style: {
            text: `VAH: ${data.valueAreaHigh.toFixed(4)}`,
            fontSize: 10,
            fill: scheme.valueArea
          }
        },
        {
          type: 'text',
          left: '92%',
          top: data.valueAreaLow,
          style: {
            text: `VAL: ${data.valueAreaLow.toFixed(4)}`,
            fontSize: 10,
            fill: scheme.valueArea
          }
        }
      ]
    };
  }, [data, showDelta, colorScheme]);

  useEffect(() => {
    if (!chartRef.current) return;

    chartInstance.current = echarts.init(chartRef.current, 'light');
    chartInstance.current.setOption(chartOptions);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, [chartOptions]);

  return (
    <div
      ref={chartRef}
      style={{ width: '100%', height: `${height}px` }}
      className="volume-profile-chart"
    />
  );
};

// Correlation Heatmap Chart
interface CorrelationHeatmapProps {
  data: CorrelationMatrix;
  height?: number;
  showValues?: boolean;
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  data,
  height = 400,
  showValues = true
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts>();

  const chartOptions = useMemo(() => {
    const heatmapData = data.matrix.flatMap((row, i) =>
      row.map((value, j) => [i, j, value])
    );

    return {
      title: {
        text: 'Correlation Matrix',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'bold' }
      },
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const [i, j, value] = params.value;
          return `${data.assets[i]} vs ${data.assets[j]}<br/>Correlation: ${value.toFixed(3)}`;
        }
      },
      grid: {
        height: '70%',
        top: '10%'
      },
      xAxis: {
        type: 'category',
        data: data.assets,
        splitArea: { show: true },
        axisLabel: { rotate: 45 }
      },
      yAxis: {
        type: 'category',
        data: data.assets,
        splitArea: { show: true }
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        inRange: {
          color: [
            '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
            '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027'
          ]
        }
      },
      series: [{
        name: 'Correlation',
        type: 'heatmap',
        data: heatmapData,
        label: {
          show: showValues,
          formatter: (params: any) => params.value[2].toFixed(2),
          fontSize: 10
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }]
    };
  }, [data, showValues]);

  useEffect(() => {
    if (!chartRef.current) return;

    chartInstance.current = echarts.init(chartRef.current, 'light');
    chartInstance.current.setOption(chartOptions);

    const handleResize = () => chartInstance.current?.resize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chartInstance.current?.dispose();
    };
  }, [chartOptions]);

  return (
    <div
      ref={chartRef}
      style={{ width: '100%', height: `${height}px` }}
      className="correlation-heatmap"
    />
  );
};