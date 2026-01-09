/**
 * Custom Indicator Builder
 * =======================
 *
 * Visual interface for creating custom technical indicators
 * with code generation, validation, and testing capabilities.
 */

'use client';

import React, { useState, useCallback, useMemo } from 'react';
import { CustomIndicator, IndicatorConfig, CandleData } from '../types';
import { IndicatorEngine } from '../engine/IndicatorEngine';

interface CustomIndicatorBuilderProps {
  indicatorEngine: IndicatorEngine;
  sampleData?: CandleData[];
  onIndicatorCreated: (indicator: CustomIndicator) => void;
  onTestResult: (result: any) => void;
}

interface InputParameter {
  name: string;
  type: 'number' | 'select' | 'boolean';
  default: any;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  description: string;
}

interface OutputParameter {
  name: string;
  type: 'line' | 'histogram' | 'area' | 'scatter';
  color: string;
  description: string;
}

interface ValidationError {
  type: 'syntax' | 'runtime' | 'logic';
  message: string;
  line?: number;
}

export const CustomIndicatorBuilder: React.FC<CustomIndicatorBuilderProps> = ({
  indicatorEngine,
  sampleData = [],
  onIndicatorCreated,
  onTestResult
}) => {
  const [indicator, setIndicator] = useState<Partial<CustomIndicator>>({
    name: '',
    description: '',
    formula: '',
    inputs: [],
    outputs: [],
    code: '',
    validation: {
      minPeriods: 1,
      requiresVolume: false,
      outputs: []
    }
  });

  const [currentTab, setCurrentTab] = useState<'setup' | 'inputs' | 'outputs' | 'code' | 'test'>('setup');
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>([]);
  const [testResults, setTestResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Predefined code templates
  const codeTemplates = {
    simple_ma: `// Simple Moving Average
function calculate(data, params) {
  const { period } = params;
  if (data.length < period) return [];

  const results = [];
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1);
    const sum = slice.reduce((acc, candle) => acc + candle.close, 0);
    const average = sum / period;

    results.push({
      timestamp: data[i].timestamp,
      value: average,
      signal: average > data[i].close ? 'SELL' : 'BUY'
    });
  }

  return results;
}`,

    rsi_divergence: `// RSI with Divergence Detection
function calculate(data, params) {
  const { period = 14, overbought = 70, oversold = 30 } = params;
  if (data.length < period + 10) return [];

  const gains = [];
  const losses = [];

  // Calculate gains and losses
  for (let i = 1; i < data.length; i++) {
    const change = data[i].close - data[i - 1].close;
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? Math.abs(change) : 0);
  }

  const results = [];
  let avgGain = gains.slice(0, period).reduce((sum, g) => sum + g, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((sum, l) => sum + l, 0) / period;

  for (let i = period; i < data.length; i++) {
    const gain = gains[i - 1];
    const loss = losses[i - 1];

    avgGain = ((avgGain * (period - 1)) + gain) / period;
    avgLoss = ((avgLoss * (period - 1)) + loss) / period;

    const rs = avgGain / avgLoss;
    const rsi = 100 - (100 / (1 + rs));

    // Simple divergence detection
    let signal = 'NEUTRAL';
    if (i > period + 5) {
      const currentPrice = data[i].close;
      const pastPrice = data[i - 5].close;
      const currentRSI = rsi;
      const pastRSI = results[results.length - 5]?.value || rsi;

      if (currentPrice < pastPrice && currentRSI > pastRSI && rsi < oversold) {
        signal = 'BUY'; // Bullish divergence
      } else if (currentPrice > pastPrice && currentRSI < pastRSI && rsi > overbought) {
        signal = 'SELL'; // Bearish divergence
      }
    }

    results.push({
      timestamp: data[i].timestamp,
      value: rsi,
      signal,
      zone: rsi > overbought ? 'OVERBOUGHT' : rsi < oversold ? 'OVERSOLD' : 'NEUTRAL'
    });
  }

  return results;
}`,

    volume_momentum: `// Volume-Weighted Momentum
function calculate(data, params) {
  const { period = 20, volumePeriod = 10 } = params;
  if (data.length < Math.max(period, volumePeriod)) return [];

  const results = [];

  for (let i = Math.max(period, volumePeriod) - 1; i < data.length; i++) {
    // Calculate price momentum
    const currentPrice = data[i].close;
    const pastPrice = data[i - period + 1].close;
    const priceMomentum = (currentPrice - pastPrice) / pastPrice;

    // Calculate volume momentum
    const volumeSlice = data.slice(i - volumePeriod + 1, i + 1);
    const avgVolume = volumeSlice.reduce((sum, d) => sum + d.volume, 0) / volumePeriod;
    const currentVolume = data[i].volume;
    const volumeRatio = currentVolume / avgVolume;

    // Combine price and volume momentum
    const momentum = priceMomentum * Math.log(volumeRatio + 1);

    let signal = 'NEUTRAL';
    if (momentum > 0.02 && volumeRatio > 1.5) signal = 'BUY';
    else if (momentum < -0.02 && volumeRatio > 1.5) signal = 'SELL';

    results.push({
      timestamp: data[i].timestamp,
      value: momentum,
      signal,
      volumeRatio,
      priceMomentum
    });
  }

  return results;
}`
  };

  const addInputParameter = useCallback(() => {
    const newInput: InputParameter = {
      name: `param${indicator.inputs?.length || 0 + 1}`,
      type: 'number',
      default: 14,
      description: 'Parameter description'
    };

    setIndicator(prev => ({
      ...prev,
      inputs: [...(prev.inputs || []), newInput]
    }));
  }, [indicator.inputs]);

  const updateInputParameter = useCallback((index: number, field: string, value: any) => {
    setIndicator(prev => ({
      ...prev,
      inputs: prev.inputs?.map((input, i) =>
        i === index ? { ...input, [field]: value } : input
      ) || []
    }));
  }, []);

  const removeInputParameter = useCallback((index: number) => {
    setIndicator(prev => ({
      ...prev,
      inputs: prev.inputs?.filter((_, i) => i !== index) || []
    }));
  }, []);

  const addOutputParameter = useCallback(() => {
    const newOutput: OutputParameter = {
      name: `output${indicator.outputs?.length || 0 + 1}`,
      type: 'line',
      color: '#3b82f6',
      description: 'Output description'
    };

    setIndicator(prev => ({
      ...prev,
      outputs: [...(prev.outputs || []), newOutput]
    }));
  }, [indicator.outputs]);

  const updateOutputParameter = useCallback((index: number, field: string, value: any) => {
    setIndicator(prev => ({
      ...prev,
      outputs: prev.outputs?.map((output, i) =>
        i === index ? { ...output, [field]: value } : output
      ) || []
    }));
  }, []);

  const removeOutputParameter = useCallback((index: number) => {
    setIndicator(prev => ({
      ...prev,
      outputs: prev.outputs?.filter((_, i) => i !== index) || []
    }));
  }, []);

  const validateCode = useCallback((code: string): ValidationError[] => {
    const errors: ValidationError[] = [];

    // Basic syntax validation
    if (!code.includes('function calculate')) {
      errors.push({
        type: 'syntax',
        message: 'Code must include a calculate function'
      });
    }

    if (!code.includes('return')) {
      errors.push({
        type: 'syntax',
        message: 'Calculate function must return results'
      });
    }

    // Check for dangerous patterns
    const dangerousPatterns = [
      /eval\s*\(/,
      /Function\s*\(/,
      /import\s+/,
      /require\s*\(/,
      /process\./,
      /global\./,
      /window\./,
      /document\./
    ];

    dangerousPatterns.forEach((pattern, index) => {
      if (pattern.test(code)) {
        errors.push({
          type: 'syntax',
          message: `Dangerous pattern detected: ${pattern.source}`
        });
      }
    });

    // Runtime validation would go here in a real implementation
    try {
      // Basic syntax check
      new Function('data', 'params', code);
    } catch (error) {
      errors.push({
        type: 'runtime',
        message: `Syntax error: ${error}`
      });
    }

    return errors;
  }, []);

  const testIndicator = useCallback(async () => {
    if (!indicator.code || !sampleData.length) {
      setValidationErrors([{
        type: 'logic',
        message: 'Code and sample data are required for testing'
      }]);
      return;
    }

    setIsLoading(true);
    setValidationErrors([]);

    try {
      // Validate code first
      const errors = validateCode(indicator.code);
      if (errors.length > 0) {
        setValidationErrors(errors);
        return;
      }

      // Create test parameters from inputs
      const testParams: { [key: string]: any } = {};
      indicator.inputs?.forEach(input => {
        testParams[input.name] = input.default;
      });

      // Execute the custom function
      const customFunction = new Function('data', 'params', `
        ${indicator.code}
        return calculate(data, params);
      `);

      const results = customFunction(sampleData, testParams);

      // Validate results
      if (!Array.isArray(results)) {
        throw new Error('Function must return an array');
      }

      if (results.length === 0) {
        throw new Error('Function returned empty results');
      }

      // Check result structure
      const firstResult = results[0];
      if (!firstResult.timestamp || firstResult.value === undefined) {
        throw new Error('Results must have timestamp and value properties');
      }

      setTestResults({
        success: true,
        data: results,
        summary: {
          dataPoints: results.length,
          firstValue: firstResult.value,
          lastValue: results[results.length - 1].value,
          avgValue: results.reduce((sum, r) => sum + (r.value || 0), 0) / results.length
        }
      });

      onTestResult({
        indicator: indicator.name,
        success: true,
        results
      });

    } catch (error) {
      const errorResult = {
        success: false,
        error: String(error)
      };

      setTestResults(errorResult);
      setValidationErrors([{
        type: 'runtime',
        message: String(error)
      }]);

      onTestResult({
        indicator: indicator.name,
        success: false,
        error: String(error)
      });
    } finally {
      setIsLoading(false);
    }
  }, [indicator, sampleData, validateCode, onTestResult]);

  const saveIndicator = useCallback(() => {
    if (!indicator.name || !indicator.code) {
      setValidationErrors([{
        type: 'logic',
        message: 'Name and code are required'
      }]);
      return;
    }

    const errors = validateCode(indicator.code);
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }

    const completeIndicator: CustomIndicator = {
      id: `custom_${Date.now()}`,
      name: indicator.name!,
      description: indicator.description || '',
      formula: indicator.formula || '',
      inputs: indicator.inputs || [],
      outputs: indicator.outputs || [],
      code: indicator.code!,
      validation: {
        minPeriods: indicator.validation?.minPeriods || 1,
        requiresVolume: indicator.validation?.requiresVolume || false,
        outputs: indicator.outputs?.map(o => o.name) || []
      }
    };

    onIndicatorCreated(completeIndicator);
  }, [indicator, validateCode, onIndicatorCreated]);

  const loadTemplate = useCallback((templateKey: string) => {
    const template = codeTemplates[templateKey as keyof typeof codeTemplates];
    if (template) {
      setIndicator(prev => ({
        ...prev,
        code: template
      }));
    }
  }, []);

  const renderSetupTab = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium mb-2">Indicator Name</label>
        <input
          type="text"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          value={indicator.name || ''}
          onChange={(e) => setIndicator(prev => ({ ...prev, name: e.target.value }))}
          placeholder="Enter indicator name"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Description</label>
        <textarea
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          rows={3}
          value={indicator.description || ''}
          onChange={(e) => setIndicator(prev => ({ ...prev, description: e.target.value }))}
          placeholder="Describe what this indicator does"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-2">Mathematical Formula</label>
        <input
          type="text"
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          value={indicator.formula || ''}
          onChange={(e) => setIndicator(prev => ({ ...prev, formula: e.target.value }))}
          placeholder="e.g., SMA = Σ(Close) / n"
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">Minimum Periods</label>
          <input
            type="number"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
            value={indicator.validation?.minPeriods || 1}
            onChange={(e) => setIndicator(prev => ({
              ...prev,
              validation: {
                ...prev.validation!,
                minPeriods: parseInt(e.target.value)
              }
            }))}
            min="1"
          />
        </div>

        <div>
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              className="rounded"
              checked={indicator.validation?.requiresVolume || false}
              onChange={(e) => setIndicator(prev => ({
                ...prev,
                validation: {
                  ...prev.validation!,
                  requiresVolume: e.target.checked
                }
              }))}
            />
            <span className="text-sm font-medium">Requires Volume Data</span>
          </label>
        </div>
      </div>
    </div>
  );

  const renderInputsTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Input Parameters</h3>
        <button
          onClick={addInputParameter}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Add Parameter
        </button>
      </div>

      <div className="space-y-4">
        {indicator.inputs?.map((input, index) => (
          <div key={index} className="p-4 border border-gray-200 rounded-lg">
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium mb-1">Parameter Name</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  value={input.name}
                  onChange={(e) => updateInputParameter(index, 'name', e.target.value)}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Type</label>
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  value={input.type}
                  onChange={(e) => updateInputParameter(index, 'type', e.target.value)}
                >
                  <option value="number">Number</option>
                  <option value="select">Select</option>
                  <option value="boolean">Boolean</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium mb-1">Default Value</label>
                <input
                  type={input.type === 'number' ? 'number' : 'text'}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  value={input.default}
                  onChange={(e) => updateInputParameter(index, 'default',
                    input.type === 'number' ? parseFloat(e.target.value) : e.target.value
                  )}
                />
              </div>

              {input.type === 'number' && (
                <>
                  <div>
                    <label className="block text-sm font-medium mb-1">Min</label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      value={input.min || ''}
                      onChange={(e) => updateInputParameter(index, 'min', parseFloat(e.target.value))}
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1">Max</label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      value={input.max || ''}
                      onChange={(e) => updateInputParameter(index, 'max', parseFloat(e.target.value))}
                    />
                  </div>
                </>
              )}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Description</label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
                value={input.description}
                onChange={(e) => updateInputParameter(index, 'description', e.target.value)}
                placeholder="Parameter description"
              />
            </div>

            <button
              onClick={() => removeInputParameter(index)}
              className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
            >
              Remove
            </button>
          </div>
        ))}

        {(!indicator.inputs || indicator.inputs.length === 0) && (
          <div className="text-center py-8 text-gray-500">
            No input parameters defined. Click "Add Parameter" to get started.
          </div>
        )}
      </div>
    </div>
  );

  const renderOutputsTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Output Parameters</h3>
        <button
          onClick={addOutputParameter}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          Add Output
        </button>
      </div>

      <div className="space-y-4">
        {indicator.outputs?.map((output, index) => (
          <div key={index} className="p-4 border border-gray-200 rounded-lg">
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium mb-1">Output Name</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  value={output.name}
                  onChange={(e) => updateOutputParameter(index, 'name', e.target.value)}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Chart Type</label>
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  value={output.type}
                  onChange={(e) => updateOutputParameter(index, 'type', e.target.value)}
                >
                  <option value="line">Line</option>
                  <option value="histogram">Histogram</option>
                  <option value="area">Area</option>
                  <option value="scatter">Scatter</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium mb-1">Color</label>
                <input
                  type="color"
                  className="w-full h-10 border border-gray-300 rounded-md"
                  value={output.color}
                  onChange={(e) => updateOutputParameter(index, 'color', e.target.value)}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Description</label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  value={output.description}
                  onChange={(e) => updateOutputParameter(index, 'description', e.target.value)}
                  placeholder="Output description"
                />
              </div>
            </div>

            <button
              onClick={() => removeOutputParameter(index)}
              className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
            >
              Remove
            </button>
          </div>
        ))}

        {(!indicator.outputs || indicator.outputs.length === 0) && (
          <div className="text-center py-8 text-gray-500">
            No output parameters defined. Click "Add Output" to get started.
          </div>
        )}
      </div>
    </div>
  );

  const renderCodeTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Implementation Code</h3>
        <div className="space-x-2">
          <select
            className="px-3 py-2 border border-gray-300 rounded-md"
            onChange={(e) => e.target.value && loadTemplate(e.target.value)}
            defaultValue=""
          >
            <option value="">Load Template...</option>
            <option value="simple_ma">Simple Moving Average</option>
            <option value="rsi_divergence">RSI with Divergence</option>
            <option value="volume_momentum">Volume Momentum</option>
          </select>
        </div>
      </div>

      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-medium mb-2">Code Requirements:</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Must include a <code>calculate(data, params)</code> function</li>
          <li>• <code>data</code> is an array of candle objects with open, high, low, close, volume, timestamp</li>
          <li>• <code>params</code> contains the input parameters as key-value pairs</li>
          <li>• Must return an array of objects with at least <code>timestamp</code> and <code>value</code> properties</li>
          <li>• Optional properties: <code>signal</code>, <code>confidence</code>, etc.</li>
        </ul>
      </div>

      <div className="relative">
        <textarea
          className="w-full h-96 px-3 py-2 border border-gray-300 rounded-md font-mono text-sm focus:ring-2 focus:ring-blue-500"
          value={indicator.code || ''}
          onChange={(e) => setIndicator(prev => ({ ...prev, code: e.target.value }))}
          placeholder="Enter your indicator implementation code here..."
        />
      </div>

      {validationErrors.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="font-medium text-red-800 mb-2">Validation Errors:</h4>
          <ul className="text-sm text-red-600 space-y-1">
            {validationErrors.map((error, index) => (
              <li key={index}>
                <span className="font-medium">{error.type}:</span> {error.message}
                {error.line && <span className="ml-2">(Line {error.line})</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );

  const renderTestTab = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Test Indicator</h3>
        <button
          onClick={testIndicator}
          disabled={isLoading || !indicator.code}
          className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Testing...' : 'Run Test'}
        </button>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="font-medium text-blue-800 mb-2">Test Configuration:</h4>
        <div className="text-sm text-blue-600 space-y-1">
          <div>Sample data points: {sampleData.length}</div>
          <div>Input parameters: {indicator.inputs?.length || 0}</div>
          <div>Expected outputs: {indicator.outputs?.length || 0}</div>
        </div>
      </div>

      {testResults && (
        <div className={`border rounded-lg p-4 ${
          testResults.success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
        }`}>
          <h4 className={`font-medium mb-2 ${
            testResults.success ? 'text-green-800' : 'text-red-800'
          }`}>
            Test Results:
          </h4>

          {testResults.success ? (
            <div className="text-sm text-green-600 space-y-1">
              <div>✓ Test passed successfully</div>
              <div>Data points generated: {testResults.summary.dataPoints}</div>
              <div>First value: {testResults.summary.firstValue?.toFixed(4)}</div>
              <div>Last value: {testResults.summary.lastValue?.toFixed(4)}</div>
              <div>Average value: {testResults.summary.avgValue?.toFixed(4)}</div>
            </div>
          ) : (
            <div className="text-sm text-red-600">
              ✗ Test failed: {testResults.error}
            </div>
          )}
        </div>
      )}

      {indicator.inputs && indicator.inputs.length > 0 && (
        <div className="border border-gray-200 rounded-lg p-4">
          <h4 className="font-medium mb-2">Test Parameters:</h4>
          <div className="grid grid-cols-2 gap-4">
            {indicator.inputs.map((input, index) => (
              <div key={index}>
                <label className="block text-sm font-medium mb-1">{input.name}</label>
                <input
                  type={input.type === 'number' ? 'number' : 'text'}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  defaultValue={input.default}
                  min={input.min}
                  max={input.max}
                  step={input.step}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Custom Indicator Builder</h1>
        <p className="text-gray-600">Create and test your own technical indicators with custom logic and parameters.</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex space-x-8">
          {[
            { id: 'setup', label: 'Setup' },
            { id: 'inputs', label: 'Inputs' },
            { id: 'outputs', label: 'Outputs' },
            { id: 'code', label: 'Code' },
            { id: 'test', label: 'Test' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setCurrentTab(tab.id as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                currentTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        {currentTab === 'setup' && renderSetupTab()}
        {currentTab === 'inputs' && renderInputsTab()}
        {currentTab === 'outputs' && renderOutputsTab()}
        {currentTab === 'code' && renderCodeTab()}
        {currentTab === 'test' && renderTestTab()}
      </div>

      {/* Actions */}
      <div className="mt-6 flex justify-end space-x-4">
        <button
          onClick={() => setIndicator({
            name: '',
            description: '',
            formula: '',
            inputs: [],
            outputs: [],
            code: '',
            validation: { minPeriods: 1, requiresVolume: false, outputs: [] }
          })}
          className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50"
        >
          Reset
        </button>
        <button
          onClick={saveIndicator}
          disabled={!indicator.name || !indicator.code}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Save Indicator
        </button>
      </div>
    </div>
  );
};