'use client';

/**
 * Advanced Export Capabilities Component
 * Professional export tools for charts, analysis reports, and trading data
 * Features: PDF reports, Excel exports, PNG/SVG charts, CSV data, custom templates
 */

import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Download,
  FileImage,
  FileText,
  FileSpreadsheet,
  Save,
  Upload,
  Share2,
  Settings,
  Calendar,
  Clock,
  Target,
  BarChart3,
  TrendingUp,
  Printer,
  Mail,
  Link,
  Copy,
  Check,
  Eye,
  EyeOff,
  RefreshCw,
  Zap
} from 'lucide-react';
import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';
import 'jspdf-autotable';
import * as XLSX from 'xlsx';

interface ExportData {
  charts: any[];
  indicators: any[];
  patterns: any[];
  trades: any[];
  analysis: any;
  metadata: {
    symbol: string;
    dateRange: { start: string; end: string };
    timeframe: string;
    dataPoints: number;
  };
}

interface ExportOptions {
  format: 'pdf' | 'excel' | 'csv' | 'png' | 'svg' | 'json';
  includeCharts: boolean;
  includeIndicators: boolean;
  includePatterns: boolean;
  includeTrades: boolean;
  includeAnalysis: boolean;
  template: 'professional' | 'summary' | 'detailed' | 'custom';
  quality: 'high' | 'medium' | 'low';
  orientation: 'portrait' | 'landscape';
  colorScheme: 'color' | 'grayscale';
}

interface AdvancedExportProps {
  data: ExportData;
  chartRef?: React.RefObject<HTMLElement>;
  onExportStart?: () => void;
  onExportComplete?: (result: { success: boolean; filename: string; size: number }) => void;
}

export const AdvancedExportCapabilities: React.FC<AdvancedExportProps> = ({
  data,
  chartRef,
  onExportStart,
  onExportComplete
}) => {
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const [showPreview, setShowPreview] = useState(false);
  const [exportHistory, setExportHistory] = useState<any[]>([]);
  const [copied, setCopied] = useState(false);
  
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    format: 'pdf',
    includeCharts: true,
    includeIndicators: true,
    includePatterns: true,
    includeTrades: true,
    includeAnalysis: true,
    template: 'professional',
    quality: 'high',
    orientation: 'portrait',
    colorScheme: 'color'
  });

  const previewRef = useRef<HTMLDivElement>(null);

  // Export handlers
  const exportToPDF = useCallback(async () => {
    try {
      const pdf = new jsPDF({
        orientation: exportOptions.orientation,
        unit: 'mm',
        format: 'a4'
      });

      // Add title page
      pdf.setFontSize(24);
      pdf.text('Trading Analysis Report', 20, 30);
      pdf.setFontSize(12);
      pdf.text(`Symbol: ${data.metadata.symbol}`, 20, 50);
      pdf.text(`Period: ${data.metadata.dateRange.start} - ${data.metadata.dateRange.end}`, 20, 60);
      pdf.text(`Generated: ${new Date().toLocaleString()}`, 20, 70);

      let yPosition = 90;

      // Add chart if available and requested
      if (exportOptions.includeCharts && chartRef?.current) {
        setExportProgress(20);
        const canvas = await html2canvas(chartRef.current, {
          backgroundColor: '#0f172a',
          scale: exportOptions.quality === 'high' ? 2 : 1
        });
        const imgData = canvas.toDataURL('image/png');
        
        pdf.addPage();
        pdf.setFontSize(16);
        pdf.text('Price Chart', 20, 20);
        pdf.addImage(imgData, 'PNG', 20, 30, 170, 100);
        yPosition = 140;
      }

      // Add technical indicators
      if (exportOptions.includeIndicators && data.indicators.length > 0) {
        setExportProgress(40);
        pdf.addPage();
        pdf.setFontSize(16);
        pdf.text('Technical Indicators', 20, 20);
        
        const indicators = data.indicators.map(ind => [
          ind.name,
          ind.value?.toFixed(2) || 'N/A',
          ind.signal || 'Neutral',
          ind.strength || 'N/A'
        ]);

        (pdf as any).autoTable({
          startY: 30,
          head: [['Indicator', 'Value', 'Signal', 'Strength']],
          body: indicators,
          theme: 'striped',
          styles: { fontSize: 10 }
        });
      }

      // Add pattern analysis
      if (exportOptions.includePatterns && data.patterns.length > 0) {
        setExportProgress(60);
        pdf.addPage();
        pdf.setFontSize(16);
        pdf.text('Candlestick Patterns', 20, 20);
        
        const patterns = data.patterns.map(pattern => [
          pattern.name,
          pattern.type,
          pattern.strength,
          pattern.reliability + '%',
          new Date(pattern.datetime).toLocaleDateString()
        ]);

        (pdf as any).autoTable({
          startY: 30,
          head: [['Pattern', 'Type', 'Strength', 'Reliability', 'Date']],
          body: patterns,
          theme: 'striped',
          styles: { fontSize: 10 }
        });
      }

      // Add trade history
      if (exportOptions.includeTrades && data.trades.length > 0) {
        setExportProgress(80);
        pdf.addPage();
        pdf.setFontSize(16);
        pdf.text('Trade History', 20, 20);
        
        const trades = data.trades.map(trade => [
          new Date(trade.timestamp).toLocaleDateString(),
          trade.side,
          trade.price.toFixed(2),
          trade.size.toString(),
          (trade.price * trade.size).toFixed(2)
        ]);

        (pdf as any).autoTable({
          startY: 30,
          head: [['Date', 'Side', 'Price', 'Size', 'Value']],
          body: trades,
          theme: 'striped',
          styles: { fontSize: 10 }
        });
      }

      // Add analysis summary
      if (exportOptions.includeAnalysis && data.analysis) {
        setExportProgress(90);
        pdf.addPage();
        pdf.setFontSize(16);
        pdf.text('Analysis Summary', 20, 20);
        pdf.setFontSize(12);
        let y = 40;
        
        const analysisText = [
          `Overall Trend: ${data.analysis.trend || 'N/A'}`,
          `Confidence: ${data.analysis.confidence || 'N/A'}%`,
          `Support Level: $${data.analysis.support?.toFixed(2) || 'N/A'}`,
          `Resistance Level: $${data.analysis.resistance?.toFixed(2) || 'N/A'}`,
          `Volatility: ${data.analysis.volatility || 'N/A'}`,
          `Volume Profile: ${data.analysis.volumeProfile || 'N/A'}`
        ];

        analysisText.forEach(text => {
          pdf.text(text, 20, y);
          y += 10;
        });
      }

      setExportProgress(100);

      const filename = `trading-report-${data.metadata.symbol}-${new Date().toISOString().split('T')[0]}.pdf`;
      pdf.save(filename);

      return { success: true, filename, size: pdf.output('blob').size };
    } catch (error) {
      console.error('PDF export failed:', error);
      return { success: false, filename: '', size: 0 };
    }
  }, [data, exportOptions, chartRef]);

  const exportToExcel = useCallback(async () => {
    try {
      const wb = XLSX.utils.book_new();

      // Create summary sheet
      const summaryData = [
        ['Trading Analysis Report'],
        ['Symbol', data.metadata.symbol],
        ['Date Range', `${data.metadata.dateRange.start} - ${data.metadata.dateRange.end}`],
        ['Timeframe', data.metadata.timeframe],
        ['Data Points', data.metadata.dataPoints],
        ['Generated', new Date().toLocaleString()],
        []
      ];

      if (exportOptions.includeAnalysis && data.analysis) {
        summaryData.push(
          ['Analysis Summary'],
          ['Overall Trend', data.analysis.trend || 'N/A'],
          ['Confidence', data.analysis.confidence + '%' || 'N/A'],
          ['Support Level', data.analysis.support?.toFixed(2) || 'N/A'],
          ['Resistance Level', data.analysis.resistance?.toFixed(2) || 'N/A']
        );
      }

      const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
      XLSX.utils.book_append_sheet(wb, summarySheet, 'Summary');

      // Add indicators sheet
      if (exportOptions.includeIndicators && data.indicators.length > 0) {
        const indicatorsSheet = XLSX.utils.json_to_sheet(data.indicators);
        XLSX.utils.book_append_sheet(wb, indicatorsSheet, 'Indicators');
      }

      // Add patterns sheet
      if (exportOptions.includePatterns && data.patterns.length > 0) {
        const patternsSheet = XLSX.utils.json_to_sheet(data.patterns);
        XLSX.utils.book_append_sheet(wb, patternsSheet, 'Patterns');
      }

      // Add trades sheet
      if (exportOptions.includeTrades && data.trades.length > 0) {
        const tradesSheet = XLSX.utils.json_to_sheet(data.trades);
        XLSX.utils.book_append_sheet(wb, tradesSheet, 'Trades');
      }

      const filename = `trading-data-${data.metadata.symbol}-${new Date().toISOString().split('T')[0]}.xlsx`;
      XLSX.writeFile(wb, filename);

      return { success: true, filename, size: 0 }; // Excel size calculation is complex
    } catch (error) {
      console.error('Excel export failed:', error);
      return { success: false, filename: '', size: 0 };
    }
  }, [data, exportOptions]);

  const exportToCSV = useCallback(async () => {
    try {
      let csvContent = `Trading Data Export - ${data.metadata.symbol}\n`;
      csvContent += `Generated: ${new Date().toLocaleString()}\n\n`;

      if (exportOptions.includeIndicators && data.indicators.length > 0) {
        csvContent += 'Technical Indicators\n';
        csvContent += 'Name,Value,Signal,Strength\n';
        data.indicators.forEach(ind => {
          csvContent += `${ind.name},${ind.value || ''},${ind.signal || ''},${ind.strength || ''}\n`;
        });
        csvContent += '\n';
      }

      if (exportOptions.includePatterns && data.patterns.length > 0) {
        csvContent += 'Candlestick Patterns\n';
        csvContent += 'Name,Type,Strength,Reliability,Date\n';
        data.patterns.forEach(pattern => {
          csvContent += `${pattern.name},${pattern.type},${pattern.strength},${pattern.reliability},${pattern.datetime}\n`;
        });
        csvContent += '\n';
      }

      if (exportOptions.includeTrades && data.trades.length > 0) {
        csvContent += 'Trade History\n';
        csvContent += 'Timestamp,Side,Price,Size,Value\n';
        data.trades.forEach(trade => {
          csvContent += `${trade.timestamp},${trade.side},${trade.price},${trade.size},${trade.price * trade.size}\n`;
        });
      }

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      const url = URL.createObjectURL(blob);
      const filename = `trading-data-${data.metadata.symbol}-${new Date().toISOString().split('T')[0]}.csv`;
      
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      return { success: true, filename, size: blob.size };
    } catch (error) {
      console.error('CSV export failed:', error);
      return { success: false, filename: '', size: 0 };
    }
  }, [data, exportOptions]);

  const exportToPNG = useCallback(async () => {
    try {
      if (!chartRef?.current) {
        throw new Error('Chart reference not available');
      }

      const canvas = await html2canvas(chartRef.current, {
        backgroundColor: exportOptions.colorScheme === 'grayscale' ? '#ffffff' : '#0f172a',
        scale: exportOptions.quality === 'high' ? 2 : exportOptions.quality === 'medium' ? 1.5 : 1,
        useCORS: true,
        allowTaint: false
      });

      // Convert to grayscale if requested
      if (exportOptions.colorScheme === 'grayscale') {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const data = imageData.data;
          
          for (let i = 0; i < data.length; i += 4) {
            const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
            data[i] = gray;     // red
            data[i + 1] = gray; // green
            data[i + 2] = gray; // blue
          }
          
          ctx.putImageData(imageData, 0, 0);
        }
      }

      canvas.toBlob((blob) => {
        if (blob) {
          const link = document.createElement('a');
          const url = URL.createObjectURL(blob);
          const filename = `chart-${data.metadata.symbol}-${new Date().toISOString().split('T')[0]}.png`;
          
          link.setAttribute('href', url);
          link.setAttribute('download', filename);
          link.style.visibility = 'hidden';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
        }
      }, 'image/png');

      return { success: true, filename: `chart-${data.metadata.symbol}.png`, size: 0 };
    } catch (error) {
      console.error('PNG export failed:', error);
      return { success: false, filename: '', size: 0 };
    }
  }, [chartRef, data.metadata.symbol, exportOptions]);

  // Main export handler
  const handleExport = useCallback(async () => {
    setIsExporting(true);
    setExportProgress(0);
    onExportStart?.();

    let result;

    try {
      switch (exportOptions.format) {
        case 'pdf':
          result = await exportToPDF();
          break;
        case 'excel':
          result = await exportToExcel();
          break;
        case 'csv':
          result = await exportToCSV();
          break;
        case 'png':
          result = await exportToPNG();
          break;
        case 'json':
          const jsonData = JSON.stringify(data, null, 2);
          const blob = new Blob([jsonData], { type: 'application/json' });
          const link = document.createElement('a');
          const url = URL.createObjectURL(blob);
          const filename = `trading-data-${data.metadata.symbol}.json`;
          
          link.setAttribute('href', url);
          link.setAttribute('download', filename);
          link.click();
          
          result = { success: true, filename, size: blob.size };
          break;
        default:
          throw new Error('Unsupported format');
      }

      // Add to export history
      if (result.success) {
        const exportRecord = {
          id: Date.now().toString(),
          filename: result.filename,
          format: exportOptions.format,
          size: result.size,
          timestamp: new Date().toISOString(),
          options: { ...exportOptions }
        };
        setExportHistory(prev => [exportRecord, ...prev.slice(0, 9)]); // Keep last 10 exports
      }

      onExportComplete?.(result);
    } catch (error) {
      console.error('Export failed:', error);
      result = { success: false, filename: '', size: 0 };
      onExportComplete?.(result);
    } finally {
      setIsExporting(false);
      setExportProgress(0);
    }
  }, [exportOptions, exportToPDF, exportToExcel, exportToCSV, exportToPNG, data, onExportStart, onExportComplete]);

  // Copy to clipboard
  const copyToClipboard = useCallback(async () => {
    try {
      const textData = JSON.stringify(data, null, 2);
      await navigator.clipboard.writeText(textData);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Copy failed:', error);
    }
  }, [data]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="space-y-6"
    >
      {/* Export Options */}
      <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <motion.h3
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent"
            >
              Export & Sharing
            </motion.h3>
            <Badge className="bg-slate-800 text-slate-300 border-slate-600">
              {data.metadata.dataPoints.toLocaleString()} data points
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowPreview(!showPreview)}
              className={`${showPreview ? 'text-cyan-400' : 'text-slate-400'} hover:text-white`}
            >
              {showPreview ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Format Selection */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-white">Export Format</h4>
            <div className="grid grid-cols-3 gap-2">
              {[
                { format: 'pdf', icon: FileText, label: 'PDF Report' },
                { format: 'excel', icon: FileSpreadsheet, label: 'Excel' },
                { format: 'csv', icon: FileText, label: 'CSV' },
                { format: 'png', icon: FileImage, label: 'PNG Image' },
                { format: 'svg', icon: FileImage, label: 'SVG Vector' },
                { format: 'json', icon: FileText, label: 'JSON Data' }
              ].map(({ format, icon: Icon, label }) => (
                <motion.button
                  key={format}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setExportOptions(prev => ({ ...prev, format: format as any }))}
                  className={`p-3 rounded-lg text-xs font-medium transition-all duration-200 flex flex-col items-center gap-2 ${
                    exportOptions.format === format
                      ? 'bg-cyan-500 text-white'
                      : 'bg-slate-800/50 text-slate-400 border border-slate-700/30 hover:border-slate-600/50 hover:text-slate-300'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {label}
                </motion.button>
              ))}
            </div>
          </div>

          {/* Content Selection */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-white">Include Content</h4>
            <div className="space-y-2">
              {[
                { key: 'includeCharts', label: 'Charts & Visualizations', icon: BarChart3 },
                { key: 'includeIndicators', label: 'Technical Indicators', icon: TrendingUp },
                { key: 'includePatterns', label: 'Pattern Analysis', icon: Target },
                { key: 'includeTrades', label: 'Trade History', icon: Clock },
                { key: 'includeAnalysis', label: 'Market Analysis', icon: FileText }
              ].map(({ key, label, icon: Icon }) => (
                <label key={key} className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={exportOptions[key as keyof ExportOptions] as boolean}
                    onChange={(e) => setExportOptions(prev => ({ ...prev, [key]: e.target.checked }))}
                    className="rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-slate-900"
                  />
                  <Icon className="w-4 h-4 text-slate-400" />
                  <span className="text-slate-300">{label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Advanced Options */}
        <div className="mt-6 pt-6 border-t border-slate-700/50">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="text-sm text-slate-400 mb-2 block">Template</label>
              <select
                value={exportOptions.template}
                onChange={(e) => setExportOptions(prev => ({ ...prev, template: e.target.value as any }))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
              >
                <option value="professional">Professional</option>
                <option value="summary">Summary</option>
                <option value="detailed">Detailed</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            <div>
              <label className="text-sm text-slate-400 mb-2 block">Quality</label>
              <select
                value={exportOptions.quality}
                onChange={(e) => setExportOptions(prev => ({ ...prev, quality: e.target.value as any }))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
              >
                <option value="high">High (2x)</option>
                <option value="medium">Medium (1.5x)</option>
                <option value="low">Low (1x)</option>
              </select>
            </div>

            <div>
              <label className="text-sm text-slate-400 mb-2 block">Orientation</label>
              <select
                value={exportOptions.orientation}
                onChange={(e) => setExportOptions(prev => ({ ...prev, orientation: e.target.value as any }))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
              >
                <option value="portrait">Portrait</option>
                <option value="landscape">Landscape</option>
              </select>
            </div>

            <div>
              <label className="text-sm text-slate-400 mb-2 block">Colors</label>
              <select
                value={exportOptions.colorScheme}
                onChange={(e) => setExportOptions(prev => ({ ...prev, colorScheme: e.target.value as any }))}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
              >
                <option value="color">Full Color</option>
                <option value="grayscale">Grayscale</option>
              </select>
            </div>
          </div>
        </div>

        {/* Export Actions */}
        <div className="mt-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button
              onClick={copyToClipboard}
              variant="ghost"
              size="sm"
              className="text-slate-400 hover:text-white"
            >
              {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              {copied ? 'Copied!' : 'Copy Data'}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="text-slate-400 hover:text-white"
            >
              <Share2 className="w-4 h-4 mr-2" />
              Share
            </Button>
          </div>

          <Button
            onClick={handleExport}
            disabled={isExporting}
            className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400 text-white font-semibold px-6"
          >
            {isExporting ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Exporting... {exportProgress}%
              </>
            ) : (
              <>
                <Download className="w-4 h-4 mr-2" />
                Export {exportOptions.format.toUpperCase()}
              </>
            )}
          </Button>
        </div>

        {/* Export Progress */}
        <AnimatePresence>
          {isExporting && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 pt-4 border-t border-slate-700/50"
            >
              <div className="flex items-center gap-3">
                <div className="flex-1 bg-slate-800 rounded-full h-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${exportProgress}%` }}
                    className="bg-gradient-to-r from-cyan-500 to-purple-500 h-full rounded-full"
                  />
                </div>
                <span className="text-sm text-slate-400 font-mono">
                  {exportProgress}%
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* Export History */}
      {exportHistory.length > 0 && (
        <Card className="bg-slate-900/70 backdrop-blur-xl border-slate-700/50 p-6">
          <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-slate-400" />
            Recent Exports
          </h4>
          
          <div className="space-y-2">
            {exportHistory.map((export_) => (
              <div
                key={export_.id}
                className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full bg-emerald-400`} />
                  <div>
                    <div className="text-white font-medium text-sm">{export_.filename}</div>
                    <div className="text-slate-400 text-xs">
                      {new Date(export_.timestamp).toLocaleString()} • {formatFileSize(export_.size)}
                    </div>
                  </div>
                </div>
                
                <Badge className="bg-slate-700 text-slate-300 border-slate-600">
                  {export_.format.toUpperCase()}
                </Badge>
              </div>
            ))}
          </div>
        </Card>
      )}
    </motion.div>
  );
};

export default AdvancedExportCapabilities;