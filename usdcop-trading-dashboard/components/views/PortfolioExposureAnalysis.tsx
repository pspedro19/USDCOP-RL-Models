'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle, Shield, Code, Database, Activity } from 'lucide-react';

/**
 * PORTFOLIO EXPOSURE ANALYSIS - DISABLED PENDING BACKEND IMPLEMENTATION
 *
 * This component was previously using 100% mock/hardcoded data and has been
 * DISABLED to maintain system integrity.
 *
 * REQUIRED BACKEND ENDPOINTS:
 * ============================
 *
 * 1. GET /api/risk/portfolio/exposure
 *    Returns: {
 *      countryExposure: Array<{country, exposure, percentage, risk}>,
 *      currencyExposure: Array<{currency, exposure, percentage, hedged}>,
 *      sectorExposure: Array<{sector, exposure, percentage, beta}>,
 *      maturityBuckets: Array<{bucket, exposure, avgMaturity}>,
 *      riskFactors: Array<{factor, exposure, sensitivity, contribution}>,
 *      concentrationMetrics: {herfindahlIndex, top5Concentration, ...},
 *      correlationStructure: {avgCorrelation, maxCorrelation, ...}
 *    }
 *
 * 2. GET /api/risk/attribution
 *    Returns: {
 *      totalRisk: number (daily volatility),
 *      components: Array<{name, contribution, percentage, marginalContribution}>,
 *      factorBreakdown: {systematic, specific, interaction}
 *    }
 *
 * 3. GET /api/risk/liquidity
 *    Returns: {
 *      liquidityTiers: Array<{tier, exposure, averageDays, positions}>,
 *      liquidityConcentration: number,
 *      worstCaseLiquidation: number (days),
 *      liquidityBuffer: number,
 *      marketImpactCost: number (basis points)
 *    }
 *
 * 4. GET /api/risk/stress-tests
 *    Returns: {
 *      scenarios: Array<{name, impact, probability, timeHorizon, contributors}>,
 *      tailRisk: {var95, var99, expectedShortfall, maxLoss}
 *    }
 *
 * IMPLEMENTATION NOTES:
 * =====================
 * - All metrics should be calculated from real trading positions
 * - Risk attribution requires covariance matrix computation
 * - Stress tests should use historical scenarios or Monte Carlo simulation
 * - Liquidity analysis needs market depth data and position sizes
 * - Concentration metrics use Herfindahl-Hirschman Index (HHI) formula
 *
 * PREVIOUS MOCK DATA REMOVED:
 * ============================
 * - generateMockExposureData() - Lines 142-191 (REMOVED)
 * - generateRiskAttribution() - Lines 193-208 (REMOVED)
 * - generateLiquidityAnalysis() - Lines 210-223 (REMOVED)
 * - generateStressResults() - Lines 225-275 (REMOVED)
 */

export default function PortfolioExposureAnalysis() {
  return (
    <div className="flex items-center justify-center min-h-screen bg-slate-950 p-6">
      <Card className="w-full max-w-4xl bg-slate-900 border-red-500/30">
        <CardHeader className="border-b border-red-500/20">
          <CardTitle className="text-2xl text-red-400 font-mono flex items-center gap-3">
            <Shield className="h-8 w-8" />
            PORTFOLIO EXPOSURE ANALYSIS - COMPONENT DISABLED
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6 pt-6">
          {/* Warning Alert */}
          <Alert className="border-red-500/50 bg-red-950/30">
            <AlertTriangle className="h-5 w-5 text-red-400" />
            <AlertDescription className="text-red-200 ml-2">
              <div className="font-bold text-lg mb-2">This component has been intentionally disabled</div>
              <div className="text-sm">
                The previous implementation used 100% hardcoded/mock data. To maintain system integrity
                and ensure accurate risk reporting, this component requires real backend API endpoints.
              </div>
            </AlertDescription>
          </Alert>

          {/* What Was Removed */}
          <div className="bg-slate-800 border border-yellow-500/20 rounded-lg p-6">
            <h3 className="text-yellow-400 font-bold text-lg mb-4 flex items-center gap-2">
              <Code className="h-5 w-5" />
              Mock Data Functions Removed
            </h3>
            <div className="space-y-3 text-sm font-mono">
              <div className="flex items-start gap-3">
                <span className="text-red-400">✗</span>
                <div>
                  <div className="text-white">generateMockExposureData()</div>
                  <div className="text-slate-400 text-xs">Lines 142-191 - Hardcoded country/currency/sector exposure data</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-red-400">✗</span>
                <div>
                  <div className="text-white">generateRiskAttribution()</div>
                  <div className="text-slate-400 text-xs">Lines 193-208 - Hardcoded risk contribution values (2.47% daily vol)</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-red-400">✗</span>
                <div>
                  <div className="text-white">generateLiquidityAnalysis()</div>
                  <div className="text-slate-400 text-xs">Lines 210-223 - Hardcoded liquidity tiers and market impact</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <span className="text-red-400">✗</span>
                <div>
                  <div className="text-white">generateStressResults()</div>
                  <div className="text-slate-400 text-xs">Lines 225-275 - Hardcoded stress test scenarios and VaR values</div>
                </div>
              </div>
            </div>
          </div>

          {/* Required Backend Endpoints */}
          <div className="bg-slate-800 border border-green-500/20 rounded-lg p-6">
            <h3 className="text-green-400 font-bold text-lg mb-4 flex items-center gap-2">
              <Database className="h-5 w-5" />
              Required Backend API Endpoints
            </h3>
            <div className="space-y-4 text-sm">
              <div className="bg-slate-900 p-4 rounded border border-green-500/10">
                <div className="font-mono text-green-400 mb-2">GET /api/risk/portfolio/exposure</div>
                <div className="text-slate-300 text-xs">
                  Multi-dimensional portfolio exposure breakdown (geographic, currency, sector, risk factors)
                  with concentration and correlation metrics
                </div>
              </div>

              <div className="bg-slate-900 p-4 rounded border border-green-500/10">
                <div className="font-mono text-green-400 mb-2">GET /api/risk/attribution</div>
                <div className="text-slate-300 text-xs">
                  Risk attribution analysis showing contribution of each position/factor to total portfolio risk,
                  including systematic vs specific risk breakdown
                </div>
              </div>

              <div className="bg-slate-900 p-4 rounded border border-green-500/10">
                <div className="font-mono text-green-400 mb-2">GET /api/risk/liquidity</div>
                <div className="text-slate-300 text-xs">
                  Liquidity analysis with tiered classification, worst-case liquidation time,
                  and estimated market impact costs
                </div>
              </div>

              <div className="bg-slate-900 p-4 rounded border border-green-500/10">
                <div className="font-mono text-green-400 mb-2">GET /api/risk/stress-tests</div>
                <div className="text-slate-300 text-xs">
                  Portfolio stress testing results with scenario impacts, probabilities,
                  and tail risk metrics (VaR, Expected Shortfall)
                </div>
              </div>
            </div>
          </div>

          {/* Implementation Requirements */}
          <div className="bg-slate-800 border border-blue-500/20 rounded-lg p-6">
            <h3 className="text-blue-400 font-bold text-lg mb-4 flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Implementation Requirements
            </h3>
            <div className="space-y-3 text-sm text-slate-300">
              <div className="flex items-start gap-3">
                <span className="text-blue-400 font-bold">1.</span>
                <div>
                  <div className="font-semibold text-white">Real Trading Positions</div>
                  <div className="text-xs text-slate-400">
                    Backend must query actual trading positions from position tracking system or trading database
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span className="text-blue-400 font-bold">2.</span>
                <div>
                  <div className="font-semibold text-white">Risk Calculations</div>
                  <div className="text-xs text-slate-400">
                    Implement covariance matrix computation for multi-factor risk attribution.
                    Use historical price data for volatility and correlation estimates.
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span className="text-blue-400 font-bold">3.</span>
                <div>
                  <div className="font-semibold text-white">Stress Testing Engine</div>
                  <div className="text-xs text-slate-400">
                    Historical scenario replay or Monte Carlo simulation for tail risk analysis.
                    Calculate VaR/ES using percentile methods on return distribution.
                  </div>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <span className="text-blue-400 font-bold">4.</span>
                <div>
                  <div className="font-semibold text-white">Market Data Integration</div>
                  <div className="text-xs text-slate-400">
                    Requires market depth/liquidity data for impact cost estimation.
                    Need FX rates, commodity prices, and rate curves for exposure calculations.
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Call to Action */}
          <Alert className="border-amber-500/50 bg-amber-950/20">
            <AlertDescription className="text-amber-200">
              <div className="font-bold mb-2">📋 Next Steps</div>
              <div className="text-sm">
                To re-enable this component, implement the 4 required backend endpoints listed above.
                Once implemented, update this component to fetch from those endpoints instead of generating mock data.
                See the original component code (git history) for the full UI implementation.
              </div>
            </AlertDescription>
          </Alert>

          {/* Footer Info */}
          <div className="text-center pt-4 border-t border-slate-700">
            <p className="text-slate-500 text-xs font-mono">
              Component disabled on {new Date().toLocaleDateString()} •
              Original implementation: 790 lines •
              Mock functions removed: 4 •
              Required endpoints: 4 •
              Status: Awaiting backend implementation
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
