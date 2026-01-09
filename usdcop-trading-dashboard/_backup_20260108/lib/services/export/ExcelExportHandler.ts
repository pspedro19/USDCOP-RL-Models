/**
 * Excel Export Handler
 * =====================
 *
 * Handles Excel file exports
 * Implements IExportHandler interface
 *
 * Note: This is a basic implementation. For production,
 * consider using libraries like 'xlsx' or 'exceljs'
 */

import type { IExportHandler, ReportConfig, ExcelSheetData } from '@/lib/core/interfaces';

export class ExcelExportHandler implements IExportHandler {
  /**
   * Export data to Excel
   */
  async export(data: any, filename: string, config?: ReportConfig): Promise<void> {
    if (!this.validateData(data)) {
      throw new Error('Invalid data for Excel export');
    }

    console.log('[ExcelExportHandler] Exporting to Excel:', { filename, config });

    // For now, we'll export as CSV since proper Excel export requires external libraries
    // In production, you would use a library like 'xlsx' or 'exceljs'
    const csvContent = this.convertToCSV(data);
    this.downloadFile(csvContent, filename);

    console.warn(
      '[ExcelExportHandler] Using CSV format. Install xlsx library for proper Excel export.'
    );
  }

  /**
   * Get file extension
   */
  getExtension(): string {
    return 'xlsx';
  }

  /**
   * Get MIME type
   */
  getMimeType(): string {
    return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
  }

  /**
   * Validate data
   */
  validateData(data: any): boolean {
    // Support both array and ExcelSheetData format
    if (Array.isArray(data)) {
      return data.length > 0;
    }

    // Check if it's ExcelSheetData format
    if (typeof data === 'object' && 'sheets' in data) {
      return Array.isArray(data.sheets) && data.sheets.length > 0;
    }

    return false;
  }

  /**
   * Convert data to CSV format (temporary solution)
   */
  private convertToCSV(data: any): string {
    let arrayData: any[];

    // Handle ExcelSheetData format
    if (typeof data === 'object' && 'sheets' in data) {
      const sheets = data.sheets as ExcelSheetData[];
      // For now, just export the first sheet
      arrayData = sheets[0]?.data || [];
    } else {
      arrayData = data;
    }

    if (!arrayData || arrayData.length === 0) return '';

    const headers = Object.keys(arrayData[0]);
    const csvRows = [
      headers.join(','),
      ...arrayData.map((row) =>
        headers
          .map((header) => {
            const value = row[header];
            if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
              return `"${value.replace(/"/g, '""')}"`;
            }
            return value;
          })
          .join(',')
      ),
    ];

    return csvRows.join('\n');
  }

  /**
   * Trigger file download
   */
  private downloadFile(content: string, filename: string): void {
    if (typeof window === 'undefined') {
      console.error('[ExcelExportHandler] File download only available in browser');
      return;
    }

    // Using CSV MIME type for now since we're generating CSV
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    // Use .csv extension for now
    link.download = `${filename}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Clean up
    URL.revokeObjectURL(url);
  }

  /**
   * Export multiple sheets to Excel (placeholder)
   */
  async exportSheets(sheets: ExcelSheetData[], filename: string): Promise<void> {
    console.log('[ExcelExportHandler] Multi-sheet export not yet implemented');
    console.log('[ExcelExportHandler] Exporting first sheet only');

    if (sheets.length > 0) {
      await this.export(sheets[0].data, filename);
    }
  }
}
