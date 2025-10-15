/**
 * Drawing Tools Manager for ChartPro
 * Provides professional drawing tools with fabric.js integration
 */

import { IChartApi, Time, UTCTimestamp } from 'lightweight-charts';
import { fabric } from 'fabric';

export interface DrawingTool {
  id: string;
  type: DrawingToolType;
  name: string;
  icon: string;
  cursor: string;
}

export type DrawingToolType =
  | 'crosshair'
  | 'trendline'
  | 'horizontal_line'
  | 'vertical_line'
  | 'rectangle'
  | 'ellipse'
  | 'fibonacci'
  | 'text'
  | 'arrow'
  | 'parallel_channel'
  | 'pitchfork'
  | 'gann_fan'
  | 'elliott_wave';

export interface DrawingObject {
  id: string;
  type: DrawingToolType;
  points: Array<{ time: Time; price: number }>;
  style: DrawingStyle;
  locked: boolean;
  visible: boolean;
  created: Date;
  modified: Date;
}

export interface DrawingStyle {
  color: string;
  width: number;
  style: 'solid' | 'dashed' | 'dotted';
  opacity: number;
  fillColor?: string;
  fillOpacity?: number;
  fontSize?: number;
  fontFamily?: string;
  text?: string;
}

export class DrawingToolsManager {
  private chart: IChartApi;
  private container: HTMLElement;
  private fabricCanvas: fabric.Canvas;
  private activeTool: DrawingToolType = 'crosshair';
  private isDrawing = false;
  private currentDrawing: fabric.Object | null = null;
  private drawings: Map<string, DrawingObject> = new Map();
  private eventListeners: Map<string, Function> = new Map();

  // Default styles
  private defaultStyles: Record<DrawingToolType, Partial<DrawingStyle>> = {
    crosshair: {},
    trendline: {
      color: '#3b82f6',
      width: 2,
      style: 'solid',
      opacity: 1
    },
    horizontal_line: {
      color: '#10b981',
      width: 1,
      style: 'solid',
      opacity: 1
    },
    vertical_line: {
      color: '#10b981',
      width: 1,
      style: 'solid',
      opacity: 1
    },
    rectangle: {
      color: '#f59e0b',
      width: 2,
      style: 'solid',
      opacity: 1,
      fillColor: '#f59e0b',
      fillOpacity: 0.1
    },
    ellipse: {
      color: '#8b5cf6',
      width: 2,
      style: 'solid',
      opacity: 1,
      fillColor: '#8b5cf6',
      fillOpacity: 0.1
    },
    fibonacci: {
      color: '#f97316',
      width: 1,
      style: 'solid',
      opacity: 0.8
    },
    text: {
      color: '#ffffff',
      fontSize: 14,
      fontFamily: 'Inter, sans-serif',
      opacity: 1
    },
    arrow: {
      color: '#ef4444',
      width: 2,
      style: 'solid',
      opacity: 1
    },
    parallel_channel: {
      color: '#06b6d4',
      width: 1,
      style: 'solid',
      opacity: 0.8
    },
    pitchfork: {
      color: '#84cc16',
      width: 1,
      style: 'solid',
      opacity: 0.8
    },
    gann_fan: {
      color: '#f472b6',
      width: 1,
      style: 'solid',
      opacity: 0.6
    },
    elliott_wave: {
      color: '#a78bfa',
      width: 2,
      style: 'solid',
      opacity: 1,
      fontSize: 12,
      fontFamily: 'Inter, sans-serif'
    }
  };

  constructor(chart: IChartApi, container: HTMLElement) {
    this.chart = chart;
    this.container = container;

    this.initializeFabricCanvas();
    this.setupEventListeners();
  }

  private initializeFabricCanvas(): void {
    // Create overlay canvas for drawings
    const canvasEl = document.createElement('canvas');
    canvasEl.style.position = 'absolute';
    canvasEl.style.top = '0';
    canvasEl.style.left = '0';
    canvasEl.style.pointerEvents = 'none';
    canvasEl.style.zIndex = '10';

    this.container.appendChild(canvasEl);

    this.fabricCanvas = new fabric.Canvas(canvasEl, {
      selection: false,
      preserveObjectStacking: true,
      renderOnAddRemove: true,
      skipTargetFind: false
    });

    // Sync canvas size with chart container
    this.resizeCanvas();
  }

  private setupEventListeners(): void {
    const container = this.container;

    // Mouse events for drawing
    const mouseDownHandler = (event: MouseEvent) => this.handleMouseDown(event);
    const mouseMoveHandler = (event: MouseEvent) => this.handleMouseMove(event);
    const mouseUpHandler = (event: MouseEvent) => this.handleMouseUp(event);
    const keyDownHandler = (event: KeyboardEvent) => this.handleKeyDown(event);

    container.addEventListener('mousedown', mouseDownHandler);
    container.addEventListener('mousemove', mouseMoveHandler);
    container.addEventListener('mouseup', mouseUpHandler);
    document.addEventListener('keydown', keyDownHandler);

    // Store listeners for cleanup
    this.eventListeners.set('mousedown', mouseDownHandler);
    this.eventListeners.set('mousemove', mouseMoveHandler);
    this.eventListeners.set('mouseup', mouseUpHandler);
    this.eventListeners.set('keydown', keyDownHandler);

    // Resize handler
    const resizeHandler = () => this.resizeCanvas();
    window.addEventListener('resize', resizeHandler);
    this.eventListeners.set('resize', resizeHandler);

    // Chart time scale changes
    this.chart.timeScale().subscribeVisibleTimeRangeChange(() => {
      this.updateDrawingsPosition();
    });
  }

  private resizeCanvas(): void {
    if (!this.fabricCanvas) return;

    const rect = this.container.getBoundingClientRect();
    this.fabricCanvas.setDimensions({
      width: rect.width,
      height: rect.height
    });
  }

  private handleMouseDown(event: MouseEvent): void {
    if (this.activeTool === 'crosshair') return;

    event.preventDefault();
    event.stopPropagation();

    const point = this.getChartCoordinates(event);
    if (!point) return;

    this.isDrawing = true;
    this.startDrawing(point, event);
  }

  private handleMouseMove(event: MouseEvent): void {
    if (!this.isDrawing || this.activeTool === 'crosshair') return;

    const point = this.getChartCoordinates(event);
    if (!point) return;

    this.updateDrawing(point, event);
  }

  private handleMouseUp(event: MouseEvent): void {
    if (!this.isDrawing || this.activeTool === 'crosshair') return;

    const point = this.getChartCoordinates(event);
    if (!point) return;

    this.finishDrawing(point, event);
    this.isDrawing = false;
    this.currentDrawing = null;
  }

  private handleKeyDown(event: KeyboardEvent): void {
    switch (event.key) {
      case 'Delete':
      case 'Backspace':
        this.deleteSelectedDrawings();
        break;
      case 'Escape':
        this.cancelCurrentDrawing();
        break;
      case 'z':
        if (event.ctrlKey || event.metaKey) {
          this.undo();
        }
        break;
    }
  }

  private getChartCoordinates(event: MouseEvent): { time: Time; price: number; x: number; y: number } | null {
    const rect = this.container.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    try {
      const timeScale = this.chart.timeScale();
      const priceScale = this.chart.priceScale('right');

      const time = timeScale.coordinateToTime(x);
      const price = priceScale.coordinateToPrice(y);

      if (time === null || price === null) return null;

      return { time, price, x, y };
    } catch (error) {
      console.warn('Failed to get chart coordinates:', error);
      return null;
    }
  }

  private startDrawing(point: { time: Time; price: number; x: number; y: number }, event: MouseEvent): void {
    const style = this.getDefaultStyle(this.activeTool);

    switch (this.activeTool) {
      case 'trendline':
        this.startTrendline(point, style);
        break;
      case 'horizontal_line':
        this.startHorizontalLine(point, style);
        break;
      case 'vertical_line':
        this.startVerticalLine(point, style);
        break;
      case 'rectangle':
        this.startRectangle(point, style);
        break;
      case 'ellipse':
        this.startEllipse(point, style);
        break;
      case 'fibonacci':
        this.startFibonacci(point, style);
        break;
      case 'text':
        this.startText(point, style, event);
        break;
      case 'arrow':
        this.startArrow(point, style);
        break;
    }
  }

  private startTrendline(point: { x: number; y: number }, style: DrawingStyle): void {
    const line = new fabric.Line([point.x, point.y, point.x, point.y], {
      stroke: style.color,
      strokeWidth: style.width,
      strokeDashArray: this.getStrokeDashArray(style.style),
      opacity: style.opacity,
      selectable: true,
      evented: true
    });

    this.fabricCanvas.add(line);
    this.currentDrawing = line;
  }

  private startHorizontalLine(point: { x: number; y: number }, style: DrawingStyle): void {
    const line = new fabric.Line([0, point.y, this.fabricCanvas.width!, point.y], {
      stroke: style.color,
      strokeWidth: style.width,
      strokeDashArray: this.getStrokeDashArray(style.style),
      opacity: style.opacity,
      selectable: true,
      evented: true
    });

    this.fabricCanvas.add(line);
    this.currentDrawing = line;
  }

  private startVerticalLine(point: { x: number; y: number }, style: DrawingStyle): void {
    const line = new fabric.Line([point.x, 0, point.x, this.fabricCanvas.height!], {
      stroke: style.color,
      strokeWidth: style.width,
      strokeDashArray: this.getStrokeDashArray(style.style),
      opacity: style.opacity,
      selectable: true,
      evented: true
    });

    this.fabricCanvas.add(line);
    this.currentDrawing = line;
  }

  private startRectangle(point: { x: number; y: number }, style: DrawingStyle): void {
    const rect = new fabric.Rect({
      left: point.x,
      top: point.y,
      width: 0,
      height: 0,
      stroke: style.color,
      strokeWidth: style.width,
      fill: style.fillColor || 'transparent',
      opacity: style.opacity,
      selectable: true,
      evented: true
    });

    this.fabricCanvas.add(rect);
    this.currentDrawing = rect;
  }

  private startEllipse(point: { x: number; y: number }, style: DrawingStyle): void {
    const ellipse = new fabric.Ellipse({
      left: point.x,
      top: point.y,
      rx: 0,
      ry: 0,
      stroke: style.color,
      strokeWidth: style.width,
      fill: style.fillColor || 'transparent',
      opacity: style.opacity,
      selectable: true,
      evented: true
    });

    this.fabricCanvas.add(ellipse);
    this.currentDrawing = ellipse;
  }

  private startFibonacci(point: { x: number; y: number }, style: DrawingStyle): void {
    // Fibonacci retracement levels
    const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
    const group = new fabric.Group([], {
      selectable: true,
      evented: true
    });

    // Store starting point for updates
    (group as any).startPoint = point;
    (group as any).fibLevels = levels;
    (group as any).style = style;

    this.fabricCanvas.add(group);
    this.currentDrawing = group;
  }

  private startText(point: { x: number; y: number }, style: DrawingStyle, event: MouseEvent): void {
    const text = prompt('Enter text:');
    if (!text) return;

    const textObj = new fabric.Text(text, {
      left: point.x,
      top: point.y,
      fill: style.color,
      fontSize: style.fontSize || 14,
      fontFamily: style.fontFamily || 'Inter, sans-serif',
      opacity: style.opacity,
      selectable: true,
      evented: true
    });

    this.fabricCanvas.add(textObj);
    this.fabricCanvas.renderAll();
  }

  private startArrow(point: { x: number; y: number }, style: DrawingStyle): void {
    const line = new fabric.Line([point.x, point.y, point.x, point.y], {
      stroke: style.color,
      strokeWidth: style.width,
      opacity: style.opacity,
      selectable: true,
      evented: true
    });

    // Add arrowhead
    const triangle = new fabric.Triangle({
      width: 10,
      height: 10,
      fill: style.color,
      left: point.x,
      top: point.y,
      opacity: style.opacity
    });

    const group = new fabric.Group([line, triangle], {
      selectable: true,
      evented: true
    });

    this.fabricCanvas.add(group);
    this.currentDrawing = group;
  }

  private updateDrawing(point: { x: number; y: number }, event: MouseEvent): void {
    if (!this.currentDrawing) return;

    switch (this.activeTool) {
      case 'trendline':
        this.updateTrendline(point);
        break;
      case 'rectangle':
        this.updateRectangle(point);
        break;
      case 'ellipse':
        this.updateEllipse(point);
        break;
      case 'fibonacci':
        this.updateFibonacci(point);
        break;
      case 'arrow':
        this.updateArrow(point);
        break;
    }

    this.fabricCanvas.renderAll();
  }

  private updateTrendline(point: { x: number; y: number }): void {
    const line = this.currentDrawing as fabric.Line;
    line.set({ x2: point.x, y2: point.y });
  }

  private updateRectangle(point: { x: number; y: number }): void {
    const rect = this.currentDrawing as fabric.Rect;
    const startX = rect.left!;
    const startY = rect.top!;

    rect.set({
      width: Math.abs(point.x - startX),
      height: Math.abs(point.y - startY),
      left: Math.min(startX, point.x),
      top: Math.min(startY, point.y)
    });
  }

  private updateEllipse(point: { x: number; y: number }): void {
    const ellipse = this.currentDrawing as fabric.Ellipse;
    const startX = ellipse.left!;
    const startY = ellipse.top!;

    ellipse.set({
      rx: Math.abs(point.x - startX) / 2,
      ry: Math.abs(point.y - startY) / 2,
      left: Math.min(startX, point.x),
      top: Math.min(startY, point.y)
    });
  }

  private updateFibonacci(point: { x: number; y: number }): void {
    const group = this.currentDrawing as fabric.Group;
    const startPoint = (group as any).startPoint;
    const levels = (group as any).fibLevels;
    const style = (group as any).style;

    // Clear existing objects
    group.removeAll();

    const deltaY = point.y - startPoint.y;

    levels.forEach(level => {
      const y = startPoint.y + deltaY * level;
      const line = new fabric.Line([startPoint.x, y, point.x, y], {
        stroke: style.color,
        strokeWidth: style.width,
        opacity: style.opacity * (1 - level * 0.3),
        strokeDashArray: level === 0 || level === 1 ? [] : [5, 5]
      });

      const text = new fabric.Text(`${(level * 100).toFixed(1)}%`, {
        left: Math.max(startPoint.x, point.x) + 5,
        top: y - 8,
        fill: style.color,
        fontSize: 12,
        opacity: style.opacity
      });

      group.addWithUpdate(line);
      group.addWithUpdate(text);
    });
  }

  private updateArrow(point: { x: number; y: number }): void {
    const group = this.currentDrawing as fabric.Group;
    const objects = group.getObjects();
    const line = objects[0] as fabric.Line;
    const triangle = objects[1] as fabric.Triangle;

    line.set({ x2: point.x, y2: point.y });

    // Update arrowhead position and rotation
    const angle = Math.atan2(point.y - line.y1!, point.x - line.x1!) * 180 / Math.PI;
    triangle.set({
      left: point.x,
      top: point.y,
      angle: angle + 90
    });
  }

  private finishDrawing(point: { x: number; y: number }, event: MouseEvent): void {
    if (!this.currentDrawing) return;

    // Generate unique ID for the drawing
    const id = `drawing_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Convert fabric coordinates back to chart coordinates
    const chartPoint = this.getChartCoordinates(event);
    if (!chartPoint) return;

    // Create drawing object record
    const drawingObject: DrawingObject = {
      id,
      type: this.activeTool,
      points: [{ time: chartPoint.time, price: chartPoint.price }],
      style: this.getDefaultStyle(this.activeTool),
      locked: false,
      visible: true,
      created: new Date(),
      modified: new Date()
    };

    this.drawings.set(id, drawingObject);

    // Add metadata to fabric object
    (this.currentDrawing as any).drawingId = id;

    this.fabricCanvas.renderAll();
  }

  private getDefaultStyle(toolType: DrawingToolType): DrawingStyle {
    const defaultStyle = this.defaultStyles[toolType] || {};
    return {
      color: '#3b82f6',
      width: 2,
      style: 'solid',
      opacity: 1,
      ...defaultStyle
    };
  }

  private getStrokeDashArray(style: 'solid' | 'dashed' | 'dotted'): number[] {
    switch (style) {
      case 'dashed':
        return [10, 5];
      case 'dotted':
        return [2, 2];
      default:
        return [];
    }
  }

  private updateDrawingsPosition(): void {
    // Update all drawings when chart view changes
    this.fabricCanvas.getObjects().forEach(obj => {
      // Update positions based on time/price coordinates
      // This would require storing original chart coordinates
      // and recalculating screen positions
    });
    this.fabricCanvas.renderAll();
  }

  private deleteSelectedDrawings(): void {
    const activeObjects = this.fabricCanvas.getActiveObjects();
    activeObjects.forEach(obj => {
      const drawingId = (obj as any).drawingId;
      if (drawingId) {
        this.drawings.delete(drawingId);
      }
      this.fabricCanvas.remove(obj);
    });
    this.fabricCanvas.renderAll();
  }

  private cancelCurrentDrawing(): void {
    if (this.currentDrawing) {
      this.fabricCanvas.remove(this.currentDrawing);
      this.currentDrawing = null;
      this.isDrawing = false;
      this.fabricCanvas.renderAll();
    }
  }

  private undo(): void {
    const objects = this.fabricCanvas.getObjects();
    if (objects.length > 0) {
      const lastObject = objects[objects.length - 1];
      const drawingId = (lastObject as any).drawingId;
      if (drawingId) {
        this.drawings.delete(drawingId);
      }
      this.fabricCanvas.remove(lastObject);
      this.fabricCanvas.renderAll();
    }
  }

  // Public methods
  public setActiveTool(tool: DrawingToolType): void {
    this.activeTool = tool;

    // Update cursor and canvas interaction mode
    if (tool === 'crosshair') {
      this.fabricCanvas.selection = false;
      this.fabricCanvas.defaultCursor = 'crosshair';
    } else {
      this.fabricCanvas.selection = true;
      this.fabricCanvas.defaultCursor = 'default';
    }

    // Enable/disable pointer events
    const canvasEl = this.fabricCanvas.getElement();
    canvasEl.style.pointerEvents = tool === 'crosshair' ? 'none' : 'auto';
  }

  public clearAllDrawings(): void {
    this.fabricCanvas.clear();
    this.drawings.clear();
  }

  public exportDrawings(): DrawingObject[] {
    return Array.from(this.drawings.values());
  }

  public importDrawings(drawings: DrawingObject[]): void {
    this.clearAllDrawings();

    drawings.forEach(drawing => {
      this.drawings.set(drawing.id, drawing);
      // Recreate fabric objects from drawing data
      // This would require implementing serialization/deserialization
    });
  }

  public getDrawingById(id: string): DrawingObject | undefined {
    return this.drawings.get(id);
  }

  public updateDrawingStyle(id: string, style: Partial<DrawingStyle>): void {
    const drawing = this.drawings.get(id);
    if (drawing) {
      drawing.style = { ...drawing.style, ...style };
      drawing.modified = new Date();

      // Update corresponding fabric object
      const fabricObj = this.fabricCanvas.getObjects().find(obj => (obj as any).drawingId === id);
      if (fabricObj) {
        fabricObj.set({
          stroke: style.color || drawing.style.color,
          strokeWidth: style.width || drawing.style.width,
          opacity: style.opacity || drawing.style.opacity,
          // Update other style properties as needed
        });
        this.fabricCanvas.renderAll();
      }
    }
  }

  public lockDrawing(id: string, locked: boolean): void {
    const drawing = this.drawings.get(id);
    if (drawing) {
      drawing.locked = locked;
      drawing.modified = new Date();

      // Update fabric object
      const fabricObj = this.fabricCanvas.getObjects().find(obj => (obj as any).drawingId === id);
      if (fabricObj) {
        fabricObj.set({
          selectable: !locked,
          evented: !locked
        });
        this.fabricCanvas.renderAll();
      }
    }
  }

  public toggleDrawingVisibility(id: string): void {
    const drawing = this.drawings.get(id);
    if (drawing) {
      drawing.visible = !drawing.visible;
      drawing.modified = new Date();

      // Update fabric object
      const fabricObj = this.fabricCanvas.getObjects().find(obj => (obj as any).drawingId === id);
      if (fabricObj) {
        fabricObj.set({ visible: drawing.visible });
        this.fabricCanvas.renderAll();
      }
    }
  }

  public destroy(): void {
    // Remove event listeners
    this.eventListeners.forEach((listener, event) => {
      if (event === 'resize') {
        window.removeEventListener(event, listener as EventListener);
      } else if (event === 'keydown') {
        document.removeEventListener(event, listener as EventListener);
      } else {
        this.container.removeEventListener(event, listener as EventListener);
      }
    });

    // Dispose fabric canvas
    this.fabricCanvas.dispose();

    // Remove canvas element
    const canvasEl = this.fabricCanvas.getElement();
    if (canvasEl && canvasEl.parentNode) {
      canvasEl.parentNode.removeChild(canvasEl);
    }

    // Clear data
    this.drawings.clear();
    this.eventListeners.clear();
  }
}

export default DrawingToolsManager;