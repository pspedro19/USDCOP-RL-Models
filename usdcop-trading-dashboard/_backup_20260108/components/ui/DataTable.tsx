"use client";

import * as React from "react";
import { motion, AnimatePresence, MotionProps } from "framer-motion";
import { FixedSizeList as List } from "react-window";
import { cn } from "@/lib/utils";
import {
  ChevronUp,
  ChevronDown,
  Search,
  Filter,
  Download,
  MoreHorizontal,
  ArrowUpDown,
  Eye,
  Edit,
  Trash2,
  TrendingUp,
  TrendingDown,
  Activity,
  RefreshCw
} from "lucide-react";

export interface ColumnDef<T = any> {
  id: string;
  header: string;
  accessorKey?: keyof T;
  accessor?: (row: T) => any;
  cell?: (value: any, row: T, index: number) => React.ReactNode;
  sortable?: boolean;
  filterable?: boolean;
  width?: number;
  minWidth?: number;
  maxWidth?: number;
  align?: 'left' | 'center' | 'right';
  type?: 'text' | 'number' | 'currency' | 'percentage' | 'date' | 'status' | 'action';
  format?: (value: any) => string;
  className?: string;
}

export interface DataTableProps<T = any>
  extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>,
    MotionProps {
  data: T[];
  columns: ColumnDef<T>[];
  variant?: 'default' | 'compact' | 'professional' | 'trading' | 'analytics';
  virtualized?: boolean;
  rowHeight?: number;
  maxHeight?: number;
  sortable?: boolean;
  filterable?: boolean;
  searchable?: boolean;
  selectable?: boolean;
  pagination?: boolean;
  pageSize?: number;
  showHeader?: boolean;
  showFooter?: boolean;
  striped?: boolean;
  hoverable?: boolean;
  animated?: boolean;
  realTime?: boolean;
  loading?: boolean;
  empty?: React.ReactNode;
  onRowClick?: (row: T, index: number) => void;
  onRowSelect?: (selectedRows: T[]) => void;
  onSort?: (column: string, direction: 'asc' | 'desc') => void;
  onFilter?: (filters: Record<string, any>) => void;
  onSearch?: (query: string) => void;
  onExport?: () => void;
  className?: string;
}

const DataTable = <T extends Record<string, any>>({
  data,
  columns,
  variant = 'default',
  virtualized = false,
  rowHeight = 48,
  maxHeight = 400,
  sortable = true,
  filterable = false,
  searchable = true,
  selectable = false,
  pagination = false,
  pageSize = 50,
  showHeader = true,
  showFooter = false,
  striped = true,
  hoverable = true,
  animated = true,
  realTime = false,
  loading = false,
  empty,
  onRowClick,
  onRowSelect,
  onSort,
  onFilter,
  onSearch,
  onExport,
  className,
  ...props
}: DataTableProps<T>) => {
  const [sortConfig, setSortConfig] = React.useState<{
    key: string;
    direction: 'asc' | 'desc';
  } | null>(null);
  const [selectedRows, setSelectedRows] = React.useState<Set<number>>(new Set());
  const [searchQuery, setSearchQuery] = React.useState('');
  const [filters, setFilters] = React.useState<Record<string, any>>({});
  const [currentPage, setCurrentPage] = React.useState(1);
  const [updatedRows, setUpdatedRows] = React.useState<Set<number>>(new Set());

  React.useEffect(() => {
    if (realTime) {
      // Mark rows as updated for animation (simplified logic)
      const updated = new Set<number>();
      data.forEach((_, index) => {
        if (Math.random() > 0.95) { // 5% chance for demo
          updated.add(index);
        }
      });
      setUpdatedRows(updated);

      const timer = setTimeout(() => setUpdatedRows(new Set()), 1000);
      return () => clearTimeout(timer);
    }
  }, [data, realTime]);

  const handleSort = (columnId: string) => {
    if (!sortable) return;

    const direction = sortConfig?.key === columnId && sortConfig.direction === 'asc' ? 'desc' : 'asc';
    setSortConfig({ key: columnId, direction });
    onSort?.(columnId, direction);
  };

  const handleRowSelect = (index: number) => {
    if (!selectable) return;

    const newSelected = new Set(selectedRows);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedRows(newSelected);
    onRowSelect?.(Array.from(newSelected).map(i => data[i]));
  };

  const handleSelectAll = () => {
    if (selectedRows.size === data.length) {
      setSelectedRows(new Set());
      onRowSelect?.([]);
    } else {
      const allRows = new Set(data.map((_, index) => index));
      setSelectedRows(allRows);
      onRowSelect?.(data);
    }
  };

  const formatCellValue = (column: ColumnDef<T>, value: any, row: T, index: number) => {
    if (column.cell) {
      return column.cell(value, row, index);
    }

    if (column.format) {
      return column.format(value);
    }

    switch (column.type) {
      case 'currency':
        return typeof value === 'number' ? `$${value.toFixed(2)}` : value;
      case 'percentage':
        return typeof value === 'number' ? `${value.toFixed(2)}%` : value;
      case 'date':
        return value instanceof Date ? value.toLocaleDateString() : value;
      case 'status':
        return (
          <span
            className={cn(
              "px-2 py-1 rounded-full text-xs font-medium",
              value === 'active' && "bg-green-500/20 text-green-400",
              value === 'inactive' && "bg-slate-500/20 text-slate-400",
              value === 'pending' && "bg-amber-500/20 text-amber-400",
              value === 'error' && "bg-red-500/20 text-red-400"
            )}
          >
            {value}
          </span>
        );
      default:
        return value?.toString() || '';
    }
  };

  const getCellValue = (row: T, column: ColumnDef<T>) => {
    if (column.accessor) {
      return column.accessor(row);
    }
    if (column.accessorKey) {
      return row[column.accessorKey];
    }
    return '';
  };

  const HeaderCell = ({ column }: { column: ColumnDef<T> }) => {
    const isSorted = sortConfig?.key === column.id;
    const sortDirection = sortConfig?.direction;

    return (
      <motion.th
        className={cn(
          "px-4 py-3 text-left border-b border-slate-600/30",
          "bg-gradient-to-r from-slate-800/50 to-transparent",
          variant === 'professional' && "font-mono text-xs uppercase tracking-wider",
          column.sortable && sortable && "cursor-pointer hover:bg-slate-700/30",
          column.align === 'center' && "text-center",
          column.align === 'right' && "text-right",
          column.className
        )}
        style={{
          width: column.width,
          minWidth: column.minWidth,
          maxWidth: column.maxWidth
        }}
        onClick={() => column.sortable && handleSort(column.id)}
        whileHover={column.sortable && sortable ? { backgroundColor: 'rgba(51, 65, 85, 0.3)' } : undefined}
      >
        <div className="flex items-center space-x-2">
          <span className={cn(
            "font-medium text-slate-300",
            variant === 'professional' && "text-cyan-400 text-glow"
          )}>
            {column.header}
          </span>

          {column.sortable && sortable && (
            <motion.div
              className="flex flex-col"
              whileHover={{ scale: 1.1 }}
            >
              {!isSorted && <ArrowUpDown className="w-3 h-3 text-slate-500" />}
              {isSorted && sortDirection === 'asc' && <ChevronUp className="w-3 h-3 text-cyan-400" />}
              {isSorted && sortDirection === 'desc' && <ChevronDown className="w-3 h-3 text-cyan-400" />}
            </motion.div>
          )}
        </div>
      </motion.th>
    );
  };

  const DataRow = ({ row, index, style }: { row: T; index: number; style?: React.CSSProperties }) => {
    const isSelected = selectedRows.has(index);
    const isUpdated = updatedRows.has(index);

    const rowVariants = {
      initial: { opacity: 0, x: -20 },
      animate: {
        opacity: 1,
        x: 0,
        transition: {
          duration: 0.3,
          delay: animated ? index * 0.02 : 0,
          ease: [0.4, 0, 0.2, 1]
        }
      },
      hover: {
        backgroundColor: 'rgba(51, 65, 85, 0.3)',
        scale: 1.01,
        transition: { duration: 0.2 }
      },
      tap: {
        scale: 0.99,
        transition: { duration: 0.1 }
      },
      updated: {
        backgroundColor: [
          'rgba(6, 182, 212, 0.1)',
          'rgba(6, 182, 212, 0.3)',
          'rgba(6, 182, 212, 0.1)'
        ],
        transition: { duration: 0.8, ease: [0.4, 0, 0.2, 1] }
      }
    };

    return (
      <motion.tr
        style={style}
        variants={rowVariants}
        initial={animated ? "initial" : undefined}
        animate={animated ? (isUpdated ? "updated" : "animate") : undefined}
        whileHover={animated && hoverable ? "hover" : undefined}
        whileTap={animated ? "tap" : undefined}
        onClick={() => onRowClick?.(row, index)}
        className={cn(
          "group transition-colors duration-200",
          striped && index % 2 === 0 && "bg-slate-900/30",
          isSelected && "bg-cyan-500/20 border-l-4 border-l-cyan-400",
          onRowClick && "cursor-pointer",
          variant === 'professional' && "font-mono text-sm"
        )}
      >
        {selectable && (
          <td className="px-4 py-3 w-12">
            <motion.input
              type="checkbox"
              checked={isSelected}
              onChange={() => handleRowSelect(index)}
              className="w-4 h-4 text-cyan-400 bg-transparent border-slate-600 rounded focus:ring-cyan-400"
              onClick={(e) => e.stopPropagation()}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            />
          </td>
        )}

        {columns.map((column) => {
          const value = getCellValue(row, column);
          const formattedValue = formatCellValue(column, value, row, index);

          return (
            <motion.td
              key={column.id}
              className={cn(
                "px-4 py-3 border-b border-slate-600/20",
                column.align === 'center' && "text-center",
                column.align === 'right' && "text-right",
                column.className
              )}
              style={{
                width: column.width,
                minWidth: column.minWidth,
                maxWidth: column.maxWidth
              }}
              whileHover={column.type === 'number' || column.type === 'currency' ? {
                color: '#06B6D4',
                transition: { duration: 0.2 }
              } : undefined}
            >
              <div className="flex items-center space-x-2">
                {formattedValue}

                {/* Trend indicators for numeric values */}
                {(column.type === 'number' || column.type === 'currency') && typeof value === 'number' && (
                  <motion.div
                    className={cn(
                      "opacity-0 group-hover:opacity-100 transition-opacity",
                      value > 0 ? "text-market-up" : value < 0 ? "text-market-down" : "text-slate-400"
                    )}
                    whileHover={{ scale: 1.2 }}
                  >
                    {value > 0 && <TrendingUp className="w-3 h-3" />}
                    {value < 0 && <TrendingDown className="w-3 h-3" />}
                    {value === 0 && <Activity className="w-3 h-3" />}
                  </motion.div>
                )}
              </div>
            </motion.td>
          );
        })}

        {/* Actions column */}
        <td className="px-4 py-3 w-12">
          <motion.div
            className="opacity-0 group-hover:opacity-100 transition-opacity"
            whileHover={{ scale: 1.1 }}
          >
            <MoreHorizontal className="w-4 h-4 text-slate-400 cursor-pointer" />
          </motion.div>
        </td>
      </motion.tr>
    );
  };

  const VirtualizedRow = ({ index, style }: { index: number; style: React.CSSProperties }) => {
    return <DataRow row={data[index]} index={index} style={style} />;
  };

  const containerVariants = {
    initial: { opacity: 0, y: 20 },
    animate: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
        staggerChildren: animated ? 0.05 : 0,
        delayChildren: 0.1
      }
    }
  };

  if (loading) {
    return (
      <motion.div
        className={cn(
          "relative backdrop-blur-lg border border-slate-600/30 rounded-2xl overflow-hidden",
          "bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/90",
          className
        )}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="p-8 text-center">
          <motion.div
            className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full mx-auto mb-4"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <p className="text-slate-400">Loading data...</p>
        </div>
      </motion.div>
    );
  }

  if (data.length === 0) {
    return (
      <motion.div
        className={cn(
          "relative backdrop-blur-lg border border-slate-600/30 rounded-2xl overflow-hidden",
          "bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/90",
          className
        )}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="p-8 text-center">
          {empty || (
            <>
              <div className="w-12 h-12 bg-slate-700/50 rounded-xl flex items-center justify-center mx-auto mb-4">
                <Eye className="w-6 h-6 text-slate-400" />
              </div>
              <p className="text-slate-400 font-medium">No data available</p>
              <p className="text-slate-500 text-sm mt-1">Check back later or adjust your filters</p>
            </>
          )}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial={animated ? "initial" : undefined}
      animate={animated ? "animate" : undefined}
      className={cn(
        "relative backdrop-blur-lg border border-slate-600/30 rounded-2xl overflow-hidden",
        "bg-gradient-to-br from-slate-900/80 via-slate-800/60 to-slate-900/90",
        variant === 'professional' && "shadow-glass-lg",
        className
      )}
      {...props}
    >
      {/* Toolbar */}
      {(searchable || filterable || onExport) && (
        <motion.div
          className="flex items-center justify-between p-4 border-b border-slate-600/30"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="flex items-center space-x-4">
            {searchable && (
              <motion.div
                className="relative"
                whileFocus={{ scale: 1.02 }}
              >
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    onSearch?.(e.target.value);
                  }}
                  className={cn(
                    "pl-10 pr-4 py-2 bg-slate-800/50 border border-slate-600/30 rounded-xl",
                    "text-slate-200 placeholder-slate-400 focus:border-cyan-400/50 focus:outline-none",
                    variant === 'professional' && "font-mono text-sm"
                  )}
                />
              </motion.div>
            )}

            {filterable && (
              <motion.button
                className="flex items-center space-x-2 px-3 py-2 bg-slate-800/50 border border-slate-600/30 rounded-xl hover:border-cyan-400/50 transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Filter className="w-4 h-4 text-slate-400" />
                <span className="text-sm text-slate-300">Filters</span>
              </motion.button>
            )}
          </div>

          <div className="flex items-center space-x-2">
            {realTime && (
              <motion.div
                className="flex items-center space-x-2"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <div className="w-2 h-2 rounded-full bg-status-live shadow-status-live" />
                <span className="text-xs text-status-live font-medium">LIVE</span>
              </motion.div>
            )}

            {onExport && (
              <motion.button
                onClick={onExport}
                className="flex items-center space-x-2 px-3 py-2 bg-cyan-500/20 border border-cyan-400/30 rounded-xl hover:bg-cyan-500/30 transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Download className="w-4 h-4 text-cyan-400" />
                <span className="text-sm text-cyan-400">Export</span>
              </motion.button>
            )}
          </div>
        </motion.div>
      )}

      {/* Table */}
      <div className="relative overflow-hidden">
        {virtualized ? (
          <div>
            {/* Header */}
            {showHeader && (
              <table className="w-full">
                <thead>
                  <tr>
                    {selectable && (
                      <th className="px-4 py-3 w-12 text-left border-b border-slate-600/30">
                        <input
                          type="checkbox"
                          checked={selectedRows.size === data.length && data.length > 0}
                          onChange={handleSelectAll}
                          className="w-4 h-4 text-cyan-400 bg-transparent border-slate-600 rounded focus:ring-cyan-400"
                        />
                      </th>
                    )}
                    {columns.map((column) => (
                      <HeaderCell key={column.id} column={column} />
                    ))}
                    <th className="px-4 py-3 w-12 border-b border-slate-600/30"></th>
                  </tr>
                </thead>
              </table>
            )}

            {/* Virtualized Body */}
            <List
              height={maxHeight}
              itemCount={data.length}
              itemSize={rowHeight}
              itemData={data}
            >
              {VirtualizedRow}
            </List>
          </div>
        ) : (
          <table className="w-full">
            {/* Header */}
            {showHeader && (
              <thead>
                <tr>
                  {selectable && (
                    <th className="px-4 py-3 w-12 text-left border-b border-slate-600/30">
                      <motion.input
                        type="checkbox"
                        checked={selectedRows.size === data.length && data.length > 0}
                        onChange={handleSelectAll}
                        className="w-4 h-4 text-cyan-400 bg-transparent border-slate-600 rounded focus:ring-cyan-400"
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                      />
                    </th>
                  )}
                  {columns.map((column) => (
                    <HeaderCell key={column.id} column={column} />
                  ))}
                  <th className="px-4 py-3 w-12 border-b border-slate-600/30"></th>
                </tr>
              </thead>
            )}

            {/* Body */}
            <tbody className={maxHeight ? `max-h-[${maxHeight}px] overflow-y-auto` : ''}>
              <AnimatePresence mode="popLayout">
                {data.map((row, index) => (
                  <DataRow key={index} row={row} index={index} />
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        )}
      </div>

      {/* Footer */}
      {showFooter && (
        <motion.div
          className="flex items-center justify-between p-4 border-t border-slate-600/30 bg-slate-800/50"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="text-sm text-slate-400">
            {selectedRows.size > 0 && `${selectedRows.size} of `}
            {data.length} rows
          </div>

          {pagination && (
            <div className="flex items-center space-x-2">
              {/* Pagination controls would go here */}
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
};

DataTable.displayName = "DataTable";

export { DataTable };