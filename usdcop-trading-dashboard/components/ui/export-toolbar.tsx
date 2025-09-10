/**
 * Professional Export Toolbar Component
 * 
 * Provides consistent export functionality across all dashboard views
 */

'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { 
  FileText, 
  FileSpreadsheet, 
  Download, 
  Printer,
  Share2,
  ChevronDown,
  Clock
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface ExportOption {
  id: string;
  label: string;
  icon: React.ReactNode;
  action: () => void | Promise<void>;
  description: string;
  format: string;
}

interface ExportToolbarProps {
  onExportPDF?: () => void | Promise<void>;
  onExportCSV?: () => void | Promise<void>;
  onExportExcel?: () => void | Promise<void>;
  onPrint?: () => void;
  onShare?: () => void;
  title?: string;
  disabled?: boolean;
  className?: string;
}

export function ExportToolbar({
  onExportPDF,
  onExportCSV,
  onExportExcel,
  onPrint,
  onShare,
  title = 'Export Options',
  disabled = false,
  className = ''
}: ExportToolbarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState<string | null>(null);
  
  const exportOptions: ExportOption[] = [
    {
      id: 'pdf',
      label: 'PDF Report',
      icon: <FileText className="w-4 h-4" />,
      action: onExportPDF || (() => {}),
      description: 'Complete report with charts and metrics',
      format: 'PDF'
    },
    {
      id: 'csv',
      label: 'CSV Data',
      icon: <FileSpreadsheet className="w-4 h-4" />,
      action: onExportCSV || (() => {}),
      description: 'Raw data for analysis',
      format: 'CSV'
    },
    {
      id: 'excel',
      label: 'Excel Workbook',
      icon: <FileSpreadsheet className="w-4 h-4" />,
      action: onExportExcel || (() => {}),
      description: 'Multi-sheet Excel file',
      format: 'XLSX'
    }
  ].filter(option => option.action !== (() => {}));
  
  const handleExport = async (option: ExportOption) => {
    if (disabled || isExporting) return;
    
    try {
      setIsExporting(option.id);
      await option.action();
    } catch (error) {
      console.error(`Export failed for ${option.format}:`, error);
    } finally {
      setIsExporting(null);
      setIsOpen(false);
    }
  };
  
  const handlePrint = () => {
    if (onPrint) {
      onPrint();
    } else {
      window.print();
    }
    setIsOpen(false);
  };
  
  const handleShare = () => {
    if (onShare) {
      onShare();
    } else {
      if (navigator.share) {
        navigator.share({
          title: title,
          text: `Check out this ${title.toLowerCase()} from our trading dashboard`,
          url: window.location.href
        });
      }
    }
    setIsOpen(false);
  };
  
  return (
    <div className={`relative ${className}`}>
      <Button
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled}
        className="terminal-button flex items-center gap-2 px-4 py-2"
        variant="outline"
      >
        <Download className="w-4 h-4" />
        Export
        <ChevronDown 
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} 
        />
      </Button>
      
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <div 
              className="fixed inset-0 z-40"
              onClick={() => setIsOpen(false)}
            />
            
            {/* Export Menu */}
            <motion.div
              initial={{ opacity: 0, y: -10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -10, scale: 0.95 }}
              transition={{ duration: 0.15 }}
              className="absolute top-full right-0 mt-2 z-50"
            >
              <Card className="terminal-card w-80 p-4 shadow-2xl">
                <div className="space-y-3">
                  <h3 className="font-semibold text-terminal-accent mb-3">
                    {title}
                  </h3>
                  
                  {/* Export Options */}
                  <div className="space-y-2">
                    {exportOptions.map((option) => (
                      <motion.button
                        key={option.id}
                        onClick={() => handleExport(option)}
                        disabled={isExporting !== null}
                        className={`
                          w-full flex items-center gap-3 p-3 rounded-lg
                          bg-terminal-surface hover:bg-terminal-surface-variant
                          border border-terminal-border hover:border-terminal-accent
                          transition-all duration-200 group
                          ${isExporting === option.id ? 'opacity-50' : ''}
                        `}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <div className="flex-shrink-0 p-2 rounded-lg bg-terminal-accent/10 text-terminal-accent group-hover:bg-terminal-accent group-hover:text-terminal-bg transition-colors">
                          {isExporting === option.id ? (
                            <Clock className="w-4 h-4 animate-spin" />
                          ) : (
                            option.icon
                          )}
                        </div>
                        
                        <div className="flex-1 text-left">
                          <div className="font-medium text-terminal-text">
                            {option.label}
                          </div>
                          <div className="text-sm text-terminal-text-muted">
                            {option.description}
                          </div>
                        </div>
                        
                        <div className="text-xs font-mono bg-terminal-accent/20 text-terminal-accent px-2 py-1 rounded">
                          {option.format}
                        </div>
                      </motion.button>
                    ))}
                  </div>
                  
                  {/* Additional Actions */}
                  <div className="border-t border-terminal-border pt-3 mt-3">
                    <div className="grid grid-cols-2 gap-2">
                      <Button
                        onClick={handlePrint}
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-2"
                      >
                        <Printer className="w-4 h-4" />
                        Print
                      </Button>
                      
                      <Button
                        onClick={handleShare}
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-2"
                      >
                        <Share2 className="w-4 h-4" />
                        Share
                      </Button>
                    </div>
                  </div>
                  
                  {/* Export Status */}
                  {isExporting && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="bg-terminal-accent/10 border border-terminal-accent/20 rounded-lg p-3 mt-3"
                    >
                      <div className="flex items-center gap-2 text-terminal-accent">
                        <Clock className="w-4 h-4 animate-spin" />
                        <span className="text-sm font-medium">
                          Generating {exportOptions.find(opt => opt.id === isExporting)?.format} export...
                        </span>
                      </div>
                    </motion.div>
                  )}
                </div>
              </Card>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
}

// Compact version for space-constrained layouts
export function CompactExportToolbar({ 
  onExportPDF, 
  onExportCSV,
  disabled = false 
}: Pick<ExportToolbarProps, 'onExportPDF' | 'onExportCSV' | 'disabled'>) {
  const [isExporting, setIsExporting] = useState<string | null>(null);
  
  const handleExport = async (format: string, action: (() => void | Promise<void>) | undefined) => {
    if (!action || disabled || isExporting) return;
    
    try {
      setIsExporting(format);
      await action();
    } catch (error) {
      console.error(`Export failed for ${format}:`, error);
    } finally {
      setIsExporting(null);
    }
  };
  
  return (
    <div className="flex items-center gap-1">
      {onExportPDF && (
        <Button
          onClick={() => handleExport('PDF', onExportPDF)}
          disabled={disabled || isExporting !== null}
          size="sm"
          variant="outline"
          className="flex items-center gap-1 px-2 py-1"
        >
          {isExporting === 'PDF' ? (
            <Clock className="w-3 h-3 animate-spin" />
          ) : (
            <FileText className="w-3 h-3" />
          )}
          PDF
        </Button>
      )}
      
      {onExportCSV && (
        <Button
          onClick={() => handleExport('CSV', onExportCSV)}
          disabled={disabled || isExporting !== null}
          size="sm"
          variant="outline"
          className="flex items-center gap-1 px-2 py-1"
        >
          {isExporting === 'CSV' ? (
            <Clock className="w-3 h-3 animate-spin" />
          ) : (
            <FileSpreadsheet className="w-3 h-3" />
          )}
          CSV
        </Button>
      )}
    </div>
  );
}