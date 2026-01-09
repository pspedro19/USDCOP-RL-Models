/**
 * Training Period Configuration
 * =============================
 * Defines the date boundaries for different training phases.
 * Based on 70/15/15 train/val/test split of OHLCV data.
 */

// Exact dates from database calculation
export const TRAINING_PERIODS = {
  // Training period: Model learned from this data
  TRAIN_START: new Date('2020-01-02T00:00:00Z'),
  TRAIN_END: new Date('2024-02-26T00:00:00Z'),
  
  // Validation period: Used for hyperparameter tuning
  VAL_START: new Date('2024-02-26T00:00:00Z'),
  VAL_END: new Date('2025-01-13T00:00:00Z'),
  
  // Test period: Final evaluation, model never saw this during training
  TEST_START: new Date('2025-01-13T00:00:00Z'),
  TEST_END: new Date('2026-01-08T00:00:00Z'), // Current date
  
  // Out-of-sample: Live/paper trading after deployment
  // For now, TEST and OOS are the same since we just deployed
};

// Helper to determine data type based on timestamp
export function getDataType(timestamp: Date | string): 'train' | 'validation' | 'test' | 'oos' {
  const ts = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  
  if (ts < TRAINING_PERIODS.TRAIN_END) {
    return 'train';
  } else if (ts < TRAINING_PERIODS.VAL_END) {
    return 'validation';
  } else if (ts < TRAINING_PERIODS.TEST_END) {
    return 'test';
  } else {
    return 'oos';
  }
}

// Period colors for visualization
export const PERIOD_COLORS = {
  train: {
    bg: 'rgba(59, 130, 246, 0.05)',      // Blue very subtle
    border: 'rgba(59, 130, 246, 0.3)',
    text: '#3B82F6',
    label: 'TRAIN'
  },
  validation: {
    bg: 'rgba(168, 85, 247, 0.05)',       // Purple very subtle
    border: 'rgba(168, 85, 247, 0.3)',
    text: '#A855F7',
    label: 'VAL'
  },
  test: {
    bg: 'rgba(34, 197, 94, 0.05)',        // Green very subtle
    border: 'rgba(34, 197, 94, 0.3)',
    text: '#22C55E',
    label: 'TEST'
  },
  oos: {
    bg: 'rgba(249, 115, 22, 0.05)',       // Orange very subtle
    border: 'rgba(249, 115, 22, 0.3)',
    text: '#F97316',
    label: 'OOS'
  }
};

// Grid line colors (very subtle)
export const GRID_COLORS = {
  day: 'rgba(148, 163, 184, 0.1)',        // Slate-400 very subtle
  week: 'rgba(148, 163, 184, 0.2)',       // Slate-400 more visible
  month: 'rgba(148, 163, 184, 0.3)',      // Slate-400 most visible
};

// Default chart date range: from validation start to present
export const DEFAULT_CHART_RANGE = {
  start: TRAINING_PERIODS.VAL_START,
  end: new Date()
};
