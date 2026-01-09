/**
 * CSV Export Handler
 * ==================
 *
 * Handles CSV file exports
 * Implements IExportHandler interface
 */

import type { IExportHandler, ReportConfig } from '@/lib/core/interfaces';

export class CsvExportHandler implements IExportHandler {
  /**
   * Export data to CSV
   */
  async export(data: any, filename: string, config?: ReportConfig): Promise<void> {
    if (!this.validateData(data)) {
      throw new Error('Invalid data for CSV export');
    }

    const csvContent = this.arrayToCSV(data);
    this.downloadFile(csvContent, filename);
  }

  /**
   * Get file extension
   */
  getExtension(): string {
    return 'csv';
  }

  /**
   * Get MIME type
   */
  getMimeType(): string {
    return 'text/csv';
  }

  /**
   * Validate data
   */
  validateData(data: any): boolean {
    return Array.isArray(data) && data.length > 0;
  }

  /**
   * Convert array to CSV format
   */
  private arrayToCSV(data: any[]): string {
    if (!data || data.length === 0) return '';

    const headers = Object.keys(data[0]);
    const csvRows = [
      headers.join(','),
      ...data.map((row) =>
        headers
          .map((header) => {
            const value = row[header];
            // Handle values with commas by wrapping in quotes
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
      console.error('[CsvExportHandler] File download only available in browser');
      return;
    }

    const blob = new Blob([content], { type: this.getMimeType() });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    link.download = `${filename}.${this.getExtension()}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Clean up
    URL.revokeObjectURL(url);
  }
}
