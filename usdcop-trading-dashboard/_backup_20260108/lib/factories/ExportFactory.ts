/**
 * Export Factory
 * ===============
 *
 * Factory for creating export handlers based on format
 * Implements the Factory Pattern for export handler instantiation
 *
 * Usage:
 * ```typescript
 * const handler = ExportFactory.create('csv');
 * await handler.export(data, 'report');
 * ```
 */

import type { IExportHandler } from '@/lib/core/interfaces';
import { CsvExportHandler } from '@/lib/services/export/CsvExportHandler';
import { ExcelExportHandler } from '@/lib/services/export/ExcelExportHandler';
import { PdfExportHandler } from '@/lib/services/export/PdfExportHandler';

export type ExportFormat = 'csv' | 'excel' | 'pdf';

/**
 * ExportFactory
 *
 * Creates instances of IExportHandler implementations
 * based on the specified format
 */
export class ExportFactory {
  /**
   * Create an export handler instance
   *
   * @param format - The format of the export handler to create
   * @returns An instance of IExportHandler
   * @throws Error if the format is unknown
   */
  static create(format: ExportFormat): IExportHandler {
    switch (format) {
      case 'csv':
        return new CsvExportHandler();

      case 'excel':
        return new ExcelExportHandler();

      case 'pdf':
        return new PdfExportHandler();

      default:
        throw new Error(`Unknown export format: ${format}`);
    }
  }

  /**
   * Export data using the specified format
   *
   * Convenience method that creates a handler and exports in one call
   *
   * @param format - The export format
   * @param data - The data to export
   * @param filename - The filename (without extension)
   * @param config - Optional report configuration
   */
  static async export(
    format: ExportFormat,
    data: any,
    filename: string,
    config?: any
  ): Promise<void> {
    const handler = this.create(format);
    await handler.export(data, filename, config);
  }

  /**
   * Get available export formats
   *
   * @returns Array of available export formats
   */
  static getAvailableFormats(): ExportFormat[] {
    return ['csv', 'excel', 'pdf'];
  }

  /**
   * Check if a format is valid
   *
   * @param format - The format to check
   * @returns True if the format is valid
   */
  static isValidFormat(format: string): format is ExportFormat {
    return this.getAvailableFormats().includes(format as ExportFormat);
  }

  /**
   * Get handler information
   *
   * @param format - The export format
   * @returns Object with extension and MIME type
   */
  static getHandlerInfo(format: ExportFormat): { extension: string; mimeType: string } {
    const handler = this.create(format);
    return {
      extension: handler.getExtension(),
      mimeType: handler.getMimeType(),
    };
  }

  /**
   * Validate data for a specific format
   *
   * @param format - The export format
   * @param data - The data to validate
   * @returns True if the data is valid for the format
   */
  static validateData(format: ExportFormat, data: any): boolean {
    const handler = this.create(format);
    return handler.validateData(data);
  }
}
