/**
 * D3.js Indicator Visualizations
 * =============================
 *
 * Custom D3.js visualizations for advanced technical analysis
 * with interactive features and real-time updates.
 */

'use client';

import React, { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';
import { CandleData, VolumeProfile, OrderFlow, MarketMicrostructure } from '../types';

interface D3IndicatorProps {
  width?: number;
  height?: number;
  margin?: { top: number; right: number; bottom: number; left: number };
}

// Order Flow Delta Chart
interface OrderFlowChartProps extends D3IndicatorProps {
  data: OrderFlow[];
  showImbalance?: boolean;
  colorScheme?: 'default' | 'institutional' | 'dark';
}

export const OrderFlowChart: React.FC<OrderFlowChartProps> = ({
  data,
  width = 800,
  height = 300,
  margin = { top: 20, right: 30, bottom: 40, left: 50 },
  showImbalance = true,
  colorScheme = 'default'
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  const colors = useMemo(() => {
    const schemes = {
      default: {
        positive: '#10b981',
        negative: '#ef4444',
        neutral: '#6b7280',
        background: '#ffffff',
        grid: '#f3f4f6'
      },
      institutional: {
        positive: '#059669',
        negative: '#dc2626',
        neutral: '#4b5563',
        background: '#f9fafb',
        grid: '#e5e7eb'
      },
      dark: {
        positive: '#34d399',
        negative: '#f87171',
        neutral: '#9ca3af',
        background: '#1f2937',
        grid: '#374151'
      }
    };
    return schemes[colorScheme];
  }, [colorScheme]);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => new Date(d.timestamp * 1000)) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(data, d => d.imbalance) as [number, number])
      .nice()
      .range([innerHeight, 0]);

    const volumeScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => Math.abs(d.netVolume)) || 1])
      .range([2, 20]);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add grid
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickSize(-innerHeight)
        .tickFormat(() => '')
      )
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(yScale)
        .tickSize(-innerWidth)
        .tickFormat(() => '')
      )
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    // Zero line
    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .style('stroke', colors.neutral)
      .style('stroke-width', 2)
      .style('stroke-dasharray', '5,5');

    // Order flow bars
    const bars = g.selectAll('.order-flow-bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'order-flow-bar')
      .attr('x', d => xScale(new Date(d.timestamp * 1000)) - 2)
      .attr('y', d => d.imbalance >= 0 ? yScale(d.imbalance) : yScale(0))
      .attr('width', 4)
      .attr('height', d => Math.abs(yScale(d.imbalance) - yScale(0)))
      .style('fill', d => d.imbalance >= 0 ? colors.positive : colors.negative)
      .style('opacity', 0.8);

    // Add volume circles if showing imbalance
    if (showImbalance) {
      g.selectAll('.volume-circle')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'volume-circle')
        .attr('cx', d => xScale(new Date(d.timestamp * 1000)))
        .attr('cy', d => yScale(d.imbalance))
        .attr('r', d => volumeScale(Math.abs(d.netVolume)))
        .style('fill', 'none')
        .style('stroke', d => d.imbalance >= 0 ? colors.positive : colors.negative)
        .style('stroke-width', 2)
        .style('opacity', 0.6);
    }

    // Add buy/sell pressure lines
    const line = d3.line<OrderFlow>()
      .x(d => xScale(new Date(d.timestamp * 1000)))
      .y(d => yScale(d.buyPressure - d.sellPressure))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(data)
      .attr('class', 'pressure-line')
      .attr('d', line)
      .style('fill', 'none')
      .style('stroke', colors.neutral)
      .style('stroke-width', 2)
      .style('opacity', 0.7);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickFormat(d3.timeFormat('%H:%M')))
      .style('color', colors.neutral);

    g.append('g')
      .call(d3.axisLeft(yScale))
      .style('color', colors.neutral);

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('z-index', '1000');

    // Add hover effects
    bars.on('mouseover', function(event, d) {
      d3.select(this).style('opacity', 1);
      tooltip.style('visibility', 'visible')
        .html(`
          <div>Time: ${new Date(d.timestamp * 1000).toLocaleTimeString()}</div>
          <div>Imbalance: ${d.imbalance.toFixed(2)}</div>
          <div>Buy Pressure: ${(d.buyPressure * 100).toFixed(1)}%</div>
          <div>Sell Pressure: ${(d.sellPressure * 100).toFixed(1)}%</div>
          <div>Net Volume: ${d.netVolume.toLocaleString()}</div>
          <div>VWAP: ${d.vwap.toFixed(4)}</div>
        `);
    })
    .on('mousemove', function(event) {
      tooltip.style('top', (event.pageY - 10) + 'px')
        .style('left', (event.pageX + 10) + 'px');
    })
    .on('mouseout', function() {
      d3.select(this).style('opacity', 0.8);
      tooltip.style('visibility', 'hidden');
    });

    // Cleanup tooltip on component unmount
    return () => {
      tooltip.remove();
    };
  }, [data, width, height, margin, colors, showImbalance]);

  return (
    <div className="order-flow-chart">
      <svg ref={svgRef} width={width} height={height} />
    </div>
  );
};

// Market Microstructure Visualization
interface MicrostructureChartProps extends D3IndicatorProps {
  data: MarketMicrostructure[];
  metrics?: ('spread' | 'depth' | 'liquidity' | 'volatility' | 'efficiency' | 'toxicity')[];
}

export const MicrostructureChart: React.FC<MicrostructureChartProps> = ({
  data,
  width = 800,
  height = 400,
  margin = { top: 20, right: 80, bottom: 40, left: 60 },
  metrics = ['spread', 'liquidity', 'volatility']
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  const colors = useMemo(() => ({
    spread: '#3b82f6',
    depth: '#10b981',
    liquidity: '#8b5cf6',
    volatility: '#f59e0b',
    efficiency: '#06b6d4',
    toxicity: '#ef4444'
  }), []);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => new Date(d.timestamp * 1000)) as [Date, Date])
      .range([0, innerWidth]);

    // Create separate y-scales for different metrics (normalized)
    const yScales = metrics.reduce((acc, metric) => {
      const values = data.map(d => d[metric as keyof MarketMicrostructure] as number);
      acc[metric] = d3.scaleLinear()
        .domain(d3.extent(values) as [number, number])
        .nice()
        .range([innerHeight, 0]);
      return acc;
    }, {} as Record<string, d3.ScaleLinear<number, number>>);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add grid
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickSize(-innerHeight)
        .tickFormat(() => '')
      )
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0.3);

    // Draw lines for each metric
    metrics.forEach((metric, index) => {
      const line = d3.line<MarketMicrostructure>()
        .x(d => xScale(new Date(d.timestamp * 1000)))
        .y(d => yScales[metric](d[metric as keyof MarketMicrostructure] as number))
        .curve(d3.curveMonotoneX)
        .defined(d => !isNaN(d[metric as keyof MarketMicrostructure] as number));

      g.append('path')
        .datum(data)
        .attr('class', `line-${metric}`)
        .attr('d', line)
        .style('fill', 'none')
        .style('stroke', colors[metric as keyof typeof colors])
        .style('stroke-width', 2)
        .style('opacity', 0.8);

      // Add metric label
      g.append('text')
        .attr('x', innerWidth + 10)
        .attr('y', 20 + index * 20)
        .style('fill', colors[metric as keyof typeof colors])
        .style('font-size', '12px')
        .style('font-weight', 'bold')
        .text(metric.charAt(0).toUpperCase() + metric.slice(1));
    });

    // Add dots for better visibility
    metrics.forEach(metric => {
      g.selectAll(`.dot-${metric}`)
        .data(data.filter(d => !isNaN(d[metric as keyof MarketMicrostructure] as number)))
        .enter()
        .append('circle')
        .attr('class', `dot-${metric}`)
        .attr('cx', d => xScale(new Date(d.timestamp * 1000)))
        .attr('cy', d => yScales[metric](d[metric as keyof MarketMicrostructure] as number))
        .attr('r', 3)
        .style('fill', colors[metric as keyof typeof colors])
        .style('opacity', 0.7);
    });

    // X-axis
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale)
        .tickFormat(d3.timeFormat('%H:%M')));

    // Y-axis (using the first metric's scale)
    const primaryMetric = metrics[0];
    g.append('g')
      .call(d3.axisLeft(yScales[primaryMetric]));

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text('Market Microstructure Analysis');

    // Tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'microstructure-tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none')
      .style('z-index', '1000');

    // Add invisible overlay for mouse tracking
    const overlay = g.append('rect')
      .attr('class', 'overlay')
      .attr('width', innerWidth)
      .attr('height', innerHeight)
      .style('fill', 'none')
      .style('pointer-events', 'all');

    // Bisector for finding nearest data point
    const bisect = d3.bisector((d: MarketMicrostructure) => new Date(d.timestamp * 1000)).left;

    overlay.on('mouseover', () => tooltip.style('visibility', 'visible'))
      .on('mouseout', () => tooltip.style('visibility', 'hidden'))
      .on('mousemove', function(event) {
        const [mouseX] = d3.pointer(event, this);
        const x0 = xScale.invert(mouseX);
        const i = bisect(data, x0, 1);
        const d0 = data[i - 1];
        const d1 = data[i];
        const d = x0.getTime() - new Date(d0.timestamp * 1000).getTime() > new Date(d1.timestamp * 1000).getTime() - x0.getTime() ? d1 : d0;

        if (d) {
          const tooltipContent = `
            <div><strong>Time:</strong> ${new Date(d.timestamp * 1000).toLocaleTimeString()}</div>
            ${metrics.map(metric =>
              `<div><strong>${metric}:</strong> ${(d[metric as keyof MarketMicrostructure] as number).toFixed(4)}</div>`
            ).join('')}
          `;

          tooltip.html(tooltipContent)
            .style('top', (event.pageY - 10) + 'px')
            .style('left', (event.pageX + 10) + 'px');
        }
      });

    // Cleanup
    return () => {
      tooltip.remove();
    };
  }, [data, width, height, margin, metrics, colors]);

  return (
    <div className="microstructure-chart">
      <svg ref={svgRef} width={width} height={height} />
    </div>
  );
};

// Volume Delta Ladder
interface VolumeDeltaLadderProps extends D3IndicatorProps {
  data: CandleData[];
  levels?: number;
  timeWindow?: number; // minutes
}

export const VolumeDeltaLadder: React.FC<VolumeDeltaLadderProps> = ({
  data,
  width = 300,
  height = 600,
  margin = { top: 20, right: 20, bottom: 20, left: 60 },
  levels = 20,
  timeWindow = 60
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  // Process data to create ladder levels
  const ladderData = useMemo(() => {
    if (!data.length) return [];

    // Get recent data based on time window
    const now = Date.now() / 1000;
    const windowStart = now - (timeWindow * 60);
    const recentData = data.filter(d => d.timestamp >= windowStart);

    if (!recentData.length) return [];

    // Calculate price levels
    const minPrice = Math.min(...recentData.map(d => d.low));
    const maxPrice = Math.max(...recentData.map(d => d.high));
    const priceStep = (maxPrice - minPrice) / levels;

    // Initialize levels
    const levelMap = new Map<number, { buyVolume: number; sellVolume: number; trades: number }>();

    for (let i = 0; i <= levels; i++) {
      const price = minPrice + (i * priceStep);
      levelMap.set(price, { buyVolume: 0, sellVolume: 0, trades: 0 });
    }

    // Distribute volume to levels
    recentData.forEach(candle => {
      const range = candle.high - candle.low;
      const volume = candle.volume || 0;

      // Estimate buy/sell based on candle color and position
      const isBullish = candle.close > candle.open;
      const bodySize = Math.abs(candle.close - candle.open);
      const bodyRatio = range > 0 ? bodySize / range : 0;

      for (let i = 0; i <= levels; i++) {
        const levelPrice = minPrice + (i * priceStep);

        if (levelPrice >= candle.low && levelPrice <= candle.high) {
          const level = levelMap.get(levelPrice);
          if (level) {
            const volumeWeight = 1 / Math.max(1, Math.abs(levelPrice - ((candle.high + candle.low) / 2)));
            const distributedVolume = volume * volumeWeight;

            // Estimate buy/sell split
            let buyRatio = 0.5;
            if (isBullish) {
              const pricePosition = range > 0 ? (levelPrice - candle.low) / range : 0.5;
              buyRatio = 0.3 + 0.4 * pricePosition + 0.3 * bodyRatio;
            } else {
              const pricePosition = range > 0 ? (candle.high - levelPrice) / range : 0.5;
              buyRatio = 0.3 + 0.4 * pricePosition - 0.3 * bodyRatio;
            }

            buyRatio = Math.max(0.1, Math.min(0.9, buyRatio));

            level.buyVolume += distributedVolume * buyRatio;
            level.sellVolume += distributedVolume * (1 - buyRatio);
            level.trades += 1;
          }
        }
      }
    });

    return Array.from(levelMap.entries()).map(([price, data]) => ({
      price,
      buyVolume: data.buyVolume,
      sellVolume: data.sellVolume,
      delta: data.buyVolume - data.sellVolume,
      totalVolume: data.buyVolume + data.sellVolume,
      trades: data.trades
    })).filter(d => d.totalVolume > 0);
  }, [data, levels, timeWindow]);

  useEffect(() => {
    if (!svgRef.current || !ladderData.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Scales
    const yScale = d3.scaleBand()
      .domain(ladderData.map(d => d.price.toString()))
      .range([innerHeight, 0])
      .padding(0.1);

    const volumeScale = d3.scaleLinear()
      .domain([0, d3.max(ladderData, d => d.totalVolume) || 1])
      .range([0, innerWidth / 2]);

    const deltaScale = d3.scaleLinear()
      .domain(d3.extent(ladderData, d => d.delta) as [number, number])
      .range([-innerWidth / 4, innerWidth / 4]);

    // Create main group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add center line
    g.append('line')
      .attr('x1', innerWidth / 2)
      .attr('x2', innerWidth / 2)
      .attr('y1', 0)
      .attr('y2', innerHeight)
      .style('stroke', '#6b7280')
      .style('stroke-width', 1);

    // Add buy volume bars (right side)
    g.selectAll('.buy-bar')
      .data(ladderData)
      .enter()
      .append('rect')
      .attr('class', 'buy-bar')
      .attr('x', innerWidth / 2)
      .attr('y', d => yScale(d.price.toString()) || 0)
      .attr('width', d => volumeScale(d.buyVolume))
      .attr('height', yScale.bandwidth())
      .style('fill', '#10b981')
      .style('opacity', 0.8);

    // Add sell volume bars (left side)
    g.selectAll('.sell-bar')
      .data(ladderData)
      .enter()
      .append('rect')
      .attr('class', 'sell-bar')
      .attr('x', d => innerWidth / 2 - volumeScale(d.sellVolume))
      .attr('y', d => yScale(d.price.toString()) || 0)
      .attr('width', d => volumeScale(d.sellVolume))
      .attr('height', yScale.bandwidth())
      .style('fill', '#ef4444')
      .style('opacity', 0.8);

    // Add delta indicators
    g.selectAll('.delta-indicator')
      .data(ladderData)
      .enter()
      .append('circle')
      .attr('class', 'delta-indicator')
      .attr('cx', innerWidth / 2)
      .attr('cy', d => (yScale(d.price.toString()) || 0) + yScale.bandwidth() / 2)
      .attr('r', d => Math.abs(deltaScale(d.delta) - deltaScale(0)) / 2)
      .style('fill', d => d.delta > 0 ? '#10b981' : '#ef4444')
      .style('opacity', 0.6);

    // Add price labels
    g.selectAll('.price-label')
      .data(ladderData)
      .enter()
      .append('text')
      .attr('class', 'price-label')
      .attr('x', -5)
      .attr('y', d => (yScale(d.price.toString()) || 0) + yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .style('text-anchor', 'end')
      .style('font-size', '10px')
      .style('fill', '#374151')
      .text(d => d.price.toFixed(4));

    // Add volume labels
    g.selectAll('.volume-label')
      .data(ladderData)
      .enter()
      .append('text')
      .attr('class', 'volume-label')
      .attr('x', innerWidth + 5)
      .attr('y', d => (yScale(d.price.toString()) || 0) + yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .style('font-size', '9px')
      .style('fill', '#6b7280')
      .text(d => d.totalVolume.toFixed(0));

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .text('Volume Delta Ladder');

    // Legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width - 80}, 30)`);

    legend.append('rect')
      .attr('width', 15)
      .attr('height', 10)
      .style('fill', '#10b981');

    legend.append('text')
      .attr('x', 20)
      .attr('y', 9)
      .style('font-size', '10px')
      .text('Buy');

    legend.append('rect')
      .attr('y', 15)
      .attr('width', 15)
      .attr('height', 10)
      .style('fill', '#ef4444');

    legend.append('text')
      .attr('x', 20)
      .attr('y', 24)
      .style('font-size', '10px')
      .text('Sell');
  }, [ladderData, width, height, margin]);

  return (
    <div className="volume-delta-ladder">
      <svg ref={svgRef} width={width} height={height} />
    </div>
  );
};