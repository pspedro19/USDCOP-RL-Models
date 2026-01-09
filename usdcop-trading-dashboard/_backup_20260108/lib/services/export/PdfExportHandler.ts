/**
 * PDF Export Handler
 * ===================
 *
 * Handles PDF file exports
 * Implements IExportHandler interface
 *
 * Note: This is a basic implementation. For production,
 * consider using libraries like 'jsPDF' or 'pdfmake'
 */

import type {
  IExportHandler,
  ReportConfig,
  MetricData,
  TableData,
  ChartData,
} from '@/lib/core/interfaces';

export interface PdfExportData {
  metrics?: MetricData[];
  tables?: TableData[];
  charts?: ChartData[];
}

export class PdfExportHandler implements IExportHandler {
  /**
   * Export data to PDF
   */
  async export(data: any, filename: string, config?: ReportConfig): Promise<void> {
    if (!this.validateData(data)) {
      throw new Error('Invalid data for PDF export');
    }

    console.log('[PdfExportHandler] Exporting to PDF:', { filename, config });

    // For now, we'll generate an HTML representation
    // In production, you would use a library like 'jsPDF' or 'pdfmake'
    const htmlContent = this.generateHTML(data, config);
    this.downloadAsHTML(htmlContent, filename);

    console.warn(
      '[PdfExportHandler] Using HTML format. Install jsPDF or pdfmake for proper PDF export.'
    );
  }

  /**
   * Get file extension
   */
  getExtension(): string {
    return 'pdf';
  }

  /**
   * Get MIME type
   */
  getMimeType(): string {
    return 'application/pdf';
  }

  /**
   * Validate data
   */
  validateData(data: any): boolean {
    if (typeof data !== 'object' || data === null) return false;

    // Check if data has at least one of: metrics, tables, or charts
    return (
      (Array.isArray(data.metrics) && data.metrics.length > 0) ||
      (Array.isArray(data.tables) && data.tables.length > 0) ||
      (Array.isArray(data.charts) && data.charts.length > 0)
    );
  }

  /**
   * Generate HTML representation of the report
   */
  private generateHTML(data: PdfExportData, config?: ReportConfig): string {
    const title = config?.title || 'Report';
    const subtitle = config?.subtitle || '';
    const author = config?.author || 'Trading System';
    const company = config?.company || 'Professional Trading Platform';
    const watermark = config?.watermark || '';

    let html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${title}</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 40px;
      position: relative;
    }
    .watermark {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) rotate(-45deg);
      font-size: 100px;
      color: rgba(0, 0, 0, 0.1);
      z-index: -1;
      pointer-events: none;
    }
    .header {
      text-align: center;
      margin-bottom: 40px;
      border-bottom: 2px solid #333;
      padding-bottom: 20px;
    }
    h1 {
      margin: 0;
      color: #333;
    }
    h2 {
      margin: 10px 0;
      color: #666;
      font-weight: normal;
    }
    .meta {
      color: #999;
      font-size: 14px;
      margin-top: 10px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }
    .metric {
      background: #f5f5f5;
      padding: 20px;
      border-radius: 8px;
    }
    .metric-label {
      font-size: 14px;
      color: #666;
      margin-bottom: 5px;
    }
    .metric-value {
      font-size: 24px;
      font-weight: bold;
      color: #333;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 40px;
    }
    th, td {
      padding: 12px;
      text-align: left;
      border-bottom: 1px solid #ddd;
    }
    th {
      background-color: #f5f5f5;
      font-weight: bold;
    }
    .table-title {
      font-size: 18px;
      font-weight: bold;
      margin-bottom: 10px;
    }
  </style>
</head>
<body>
`;

    if (watermark) {
      html += `  <div class="watermark">${watermark}</div>\n`;
    }

    html += `
  <div class="header">
    <h1>${title}</h1>
    ${subtitle ? `<h2>${subtitle}</h2>` : ''}
    <div class="meta">
      <div>${author} | ${company}</div>
      <div>${new Date().toLocaleString()}</div>
    </div>
  </div>
`;

    // Add metrics
    if (data.metrics && data.metrics.length > 0) {
      html += '  <div class="metrics">\n';
      data.metrics.forEach((metric) => {
        html += `
    <div class="metric">
      <div class="metric-label">${metric.label}</div>
      <div class="metric-value">${metric.value}</div>
    </div>
`;
      });
      html += '  </div>\n';
    }

    // Add tables
    if (data.tables && data.tables.length > 0) {
      data.tables.forEach((table) => {
        html += `
  <div class="table-title">${table.title}</div>
  <table>
    <thead>
      <tr>
`;
        table.headers.forEach((header) => {
          html += `        <th>${header}</th>\n`;
        });
        html += `
      </tr>
    </thead>
    <tbody>
`;
        table.rows.forEach((row) => {
          html += '      <tr>\n';
          row.forEach((cell) => {
            html += `        <td>${cell}</td>\n`;
          });
          html += '      </tr>\n';
        });
        html += `
    </tbody>
  </table>
`;
      });
    }

    html += `
</body>
</html>
`;

    return html;
  }

  /**
   * Download as HTML (temporary solution)
   */
  private downloadAsHTML(content: string, filename: string): void {
    if (typeof window === 'undefined') {
      console.error('[PdfExportHandler] File download only available in browser');
      return;
    }

    const blob = new Blob([content], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    // Use .html extension for now
    link.download = `${filename}.html`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Clean up
    URL.revokeObjectURL(url);
  }
}
