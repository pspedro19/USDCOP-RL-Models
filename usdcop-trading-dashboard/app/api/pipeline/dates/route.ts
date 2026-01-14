/**
 * Pipeline Dates API
 * ==================
 * Returns official training/validation/test date ranges from pipeline config.
 *
 * Source: config/trading_config.yaml (SSOT)
 * Contract: All dates should match the ML pipeline official configuration.
 */

import { NextResponse } from 'next/server';
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

export interface PipelineDates {
  model_version: string;
  dates: {
    // Data availability (historical data range)
    data_start: string;
    data_end: string;

    // Training period (model was trained on this data)
    training_start: string;
    training_end: string;

    // Validation period (hyperparameter tuning)
    validation_start: string;
    validation_end: string;

    // Test period (out-of-sample evaluation)
    test_start: string;
    test_end: string;
  };
  metadata: {
    config_source: string;
    last_updated: string;
  };
}

// Force dynamic rendering (no caching)
export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Fallback defaults if config cannot be read
// These should match trading_config.yaml
function getDefaultDates(): PipelineDates {
  const today = new Date().toISOString().split('T')[0];
  return {
    model_version: 'current',
    dates: {
      data_start: '2020-03-01',
      data_end: today,
      training_start: '2020-03-01',  // Matches trading_config.yaml
      training_end: '2024-12-31',
      validation_start: '2025-01-01',
      validation_end: '2025-06-30',
      test_start: '2025-07-01',
      test_end: today,
    },
    metadata: {
      config_source: 'defaults',
      last_updated: new Date().toISOString(),
    },
  };
}

export async function GET() {
  try {
    // Try to read from official config file
    // Next.js cwd can vary, so we try multiple approaches
    const cwd = process.cwd();

    const configPaths = [
      // Relative to dashboard folder
      path.join(cwd, '..', 'config', 'trading_config.yaml'),
      path.resolve(cwd, '..', 'config', 'trading_config.yaml'),
      // Absolute path (Windows)
      'C:\\Users\\pedro\\OneDrive\\Documents\\ALGO TRADING\\USDCOP\\USDCOP-RL-Models\\config\\trading_config.yaml',
      // Absolute path (Unix style for WSL/Git Bash)
      '/c/Users/pedro/OneDrive/Documents/ALGO TRADING/USDCOP/USDCOP-RL-Models/config/trading_config.yaml',
      // Docker path
      '/app/config/trading_config.yaml',
    ];

    console.log('[API/pipeline/dates] CWD:', cwd);
    console.log('[API/pipeline/dates] Searching config paths...');

    let configData: any = null;
    let configSource = 'defaults';

    for (const configPath of configPaths) {
      try {
        console.log('[API/pipeline/dates] Trying:', configPath);
        if (fs.existsSync(configPath)) {
          console.log('[API/pipeline/dates] Found config at:', configPath);
          const fileContent = fs.readFileSync(configPath, 'utf8');
          configData = yaml.load(fileContent);
          configSource = configPath;
          console.log('[API/pipeline/dates] Loaded dates:', configData?.dates);
          break;
        }
      } catch (e) {
        console.log('[API/pipeline/dates] Error reading:', configPath, e);
      }
    }

    const today = new Date().toISOString().split('T')[0];

    if (configData && configData.dates) {
      const defaults = getDefaultDates();
      const response: PipelineDates = {
        model_version: configData.model?.version || 'current',
        dates: {
          // Data availability - starts March 2020 based on historical records
          data_start: configData.dates.data_start || '2020-03-01',
          data_end: today,

          // From config file (with fallbacks)
          training_start: configData.dates.training_start || defaults.dates.training_start,
          training_end: configData.dates.training_end || defaults.dates.training_end,
          validation_start: configData.dates.validation_start || defaults.dates.validation_start,
          validation_end: configData.dates.validation_end || defaults.dates.validation_end,
          test_start: configData.dates.test_start || defaults.dates.test_start,
          test_end: today,
        },
        metadata: {
          config_source: configSource,
          last_updated: new Date().toISOString(),
        },
      };

      console.log('[API/pipeline/dates] Returning config dates:', response.dates);
      return NextResponse.json(response);
    }

    // Return defaults if config not found
    console.log('[API/pipeline/dates] Config not found, using defaults');
    return NextResponse.json(getDefaultDates());

  } catch (error) {
    console.error('[API/pipeline/dates] Error:', error);

    const defaults = getDefaultDates();
    defaults.metadata.config_source = 'error_fallback';
    return NextResponse.json(defaults);
  }
}
