/**
 * Core Export Handler Interface
 * ==============================
 *
 * Base interface for all export handlers (CSV, Excel, PDF)
 * Enables extensible export functionality
 */

export interface ExportOptions {
  format: 'pdf' | 'excel' | 'csv';
  data: any;
  filename: string;
}

export interface ReportConfig {
  title: string;
  subtitle?: string;
  author?: string;
  company?: string;
  watermark?: string;
}

export interface ChartData {
  element: HTMLElement;
  title: string;
  width?: number;
  height?: number;
}

export interface TableData {
  title: string;
  headers: string[];
  rows: string[][];
}

export interface MetricData {
  label: string;
  value: string | number;
}

export interface ExcelSheetData {
  name: string;
  data: any[];
}

/**
 * IExportHandler Interface
 *
 * All export handlers must implement this interface
 * to ensure consistent export behavior
 */
export interface IExportHandler {
  /**
   * Export data in the specific format
   */
  export(data: any, filename: string, config?: ReportConfig): Promise<void>;

  /**
   * Get the file extension for this handler
   */
  getExtension(): string;

  /**
   * Get the MIME type for this handler
   */
  getMimeType(): string;

  /**
   * Validate if the data can be exported
   */
  validateData(data: any): boolean;
}
