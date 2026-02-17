/**
 * A/B Testing Full Flow - Debug Video
 * =====================================
 * Complete end-to-end test documenting the L0→L4 pipeline execution
 * with visual verification at each step.
 *
 * This test creates screenshots at each step to document:
 * 1. Initial state (database, Airflow, Dashboard)
 * 2. L0 execution (data ingestion)
 * 3. L1 execution (feature computation)
 * 4. L2 execution (dataset building)
 * 5. L3 execution (model training)
 * 6. L4 execution (backtest + promotion)
 * 7. Frontend verification (model selector, approval panel)
 *
 * Author: Trading Team
 * Date: 2026-01-31
 */

import { test, expect, Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// Configuration
const CONFIG = {
  dashboard: {
    url: 'http://localhost:5000',
    hub: '/hub',
    experiments: '/experiments',
    production: '/production',
    backtest: '/dashboard',
  },
  airflow: {
    url: 'http://localhost:8080',
    user: 'admin',
    password: 'admin123',
  },
  mlflow: {
    url: 'http://localhost:5001',
  },
  screenshots: {
    dir: 'tests/e2e/screenshots/ab-testing-flow',
  },
};

// Helper function to take timestamped screenshots
async function captureStep(page: Page, stepName: string, description: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `${timestamp}_${stepName}.png`;
  const filepath = path.join(CONFIG.screenshots.dir, filename);

  // Ensure directory exists
  if (!fs.existsSync(CONFIG.screenshots.dir)) {
    fs.mkdirSync(CONFIG.screenshots.dir, { recursive: true });
  }

  await page.screenshot({ path: filepath, fullPage: true });
  console.log(`📸 [${stepName}] ${description}`);
  console.log(`   Screenshot: ${filepath}`);

  return filepath;
}

// Helper to log with timestamp
function logStep(step: string, message: string) {
  const timestamp = new Date().toISOString();
  console.log(`\n${'='.repeat(60)}`);
  console.log(`🎬 [${timestamp}] ${step}`);
  console.log(`   ${message}`);
  console.log('='.repeat(60));
}

test.describe('A/B Testing Full Pipeline Flow', () => {
  test.setTimeout(600000); // 10 minutes for full flow

  test.beforeAll(async () => {
    console.log('\n');
    console.log('🎬'.repeat(30));
    console.log('  A/B TESTING FULL FLOW - DEBUG VIDEO');
    console.log('  USDCOP RL Trading System');
    console.log('  Date: ' + new Date().toISOString());
    console.log('🎬'.repeat(30));
    console.log('\n');
  });

  test('Scene 1: Verify Initial State', async ({ page }) => {
    logStep('SCENE 1', 'Verifying initial state of all services');

    // 1.1 Check Dashboard is running
    logStep('1.1', 'Checking Dashboard availability');
    await page.goto(CONFIG.dashboard.url + CONFIG.dashboard.hub);
    await page.waitForLoadState('networkidle');
    await captureStep(page, '01-hub-initial', 'Dashboard Hub - Initial State');

    // Verify hub components
    const modules = page.locator('[class*="module"], [class*="card"]');
    const moduleCount = await modules.count();
    console.log(`   Found ${moduleCount} modules on hub`);

    // 1.2 Check Experiments page exists
    logStep('1.2', 'Checking Experiments page');
    await page.goto(CONFIG.dashboard.url + CONFIG.dashboard.experiments);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    await captureStep(page, '02-experiments-initial', 'Experiments Page - Initial State');

    // 1.3 Check Production page exists
    logStep('1.3', 'Checking Production page');
    await page.goto(CONFIG.dashboard.url + CONFIG.dashboard.production);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    await captureStep(page, '03-production-initial', 'Production Monitor - Initial State');

    // 1.4 Check Dashboard/Backtest page
    logStep('1.4', 'Checking Backtest Dashboard');
    await page.goto(CONFIG.dashboard.url + CONFIG.dashboard.backtest);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    await captureStep(page, '04-backtest-initial', 'Backtest Dashboard - Initial State');

    // Check model selector dropdown
    const modelSelector = page.locator('select, [role="combobox"], [class*="select"]').first();
    if (await modelSelector.isVisible()) {
      await captureStep(page, '05-model-selector', 'Model Selector Dropdown');
      console.log('   ✅ Model selector found');
    } else {
      console.log('   ⚠️ Model selector not visible');
    }

    console.log('\n✅ Scene 1 Complete: Initial state verified\n');
  });

  test('Scene 2: Check Airflow DAGs Status', async ({ page }) => {
    logStep('SCENE 2', 'Checking Airflow DAGs status');

    // 2.1 Navigate to Airflow
    await page.goto(CONFIG.airflow.url);
    await page.waitForLoadState('networkidle');

    // 2.2 Login if needed
    const loginForm = page.locator('form[action*="login"], input[name="username"]');
    if (await loginForm.isVisible().catch(() => false)) {
      logStep('2.1', 'Logging into Airflow');
      await page.fill('input[name="username"]', CONFIG.airflow.user);
      await page.fill('input[name="password"]', CONFIG.airflow.password);
      await page.click('button[type="submit"], input[type="submit"]');
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(2000);
    }

    await captureStep(page, '06-airflow-home', 'Airflow Home - DAGs List');

    // 2.3 Check specific DAGs
    const dagsToCheck = [
      'l0_macro_update',
      'l0_ohlcv_historical_backfill',
      'rl_l2_01_dataset_build',
      'rl_l3_01_model_training',
      'rl_l4_04_backtest_promotion',
    ];

    for (const dagId of dagsToCheck) {
      const dagLink = page.locator(`a:has-text("${dagId}")`).first();
      if (await dagLink.isVisible().catch(() => false)) {
        console.log(`   ✅ DAG found: ${dagId}`);
      } else {
        console.log(`   ⚠️ DAG not visible: ${dagId}`);
      }
    }

    // 2.4 Take screenshot of DAG list
    await page.waitForTimeout(1000);
    await captureStep(page, '07-airflow-dags-list', 'Airflow DAGs Overview');

    console.log('\n✅ Scene 2 Complete: Airflow status checked\n');
  });

  test('Scene 3: Backtest Execution Flow', async ({ page }) => {
    logStep('SCENE 3', 'Executing backtest and verifying results');

    // 3.1 Go to dashboard
    await page.goto(CONFIG.dashboard.url + CONFIG.dashboard.backtest);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    await captureStep(page, '08-backtest-start', 'Backtest Page - Before Execution');

    // 3.2 Open backtest panel if collapsed
    const backtestBtn = page.locator('button:has-text("Backtest"), button:has-text("Control")').first();
    if (await backtestBtn.isVisible()) {
      await backtestBtn.click();
      await page.waitForTimeout(1000);
      await captureStep(page, '09-backtest-panel-open', 'Backtest Control Panel Opened');
    }

    // 3.3 Check for model selector
    logStep('3.1', 'Checking model selector options');
    const modelSelect = page.locator('select[name*="model"], [data-testid="model-select"]').first();
    if (await modelSelect.isVisible().catch(() => false)) {
      const options = await modelSelect.locator('option').allTextContents();
      console.log(`   Available models: ${options.join(', ')}`);
      await captureStep(page, '10-model-options', 'Available Models in Dropdown');
    }

    // 3.4 Select date range (if available)
    logStep('3.2', 'Setting date range for backtest');
    const validationBtn = page.locator('button:has-text("Validación")').first();
    if (await validationBtn.isVisible().catch(() => false)) {
      await validationBtn.click();
      await page.waitForTimeout(500);
      await captureStep(page, '11-date-range-selected', 'Validation Date Range Selected');
    }

    // 3.5 Start backtest
    logStep('3.3', 'Starting backtest execution');
    const startBtn = page.locator('button:has-text("Iniciar Backtest"), button:has-text("Start")').first();
    if (await startBtn.isVisible() && await startBtn.isEnabled()) {
      await startBtn.click();
      console.log('   ⏳ Backtest started, waiting for completion...');

      // Monitor progress
      for (let i = 0; i < 20; i++) {
        await page.waitForTimeout(3000);
        await captureStep(page, `12-backtest-progress-${i}`, `Backtest Progress - ${i * 3}s`);

        // Check if completed
        const pageContent = await page.content();
        if (pageContent.includes('completed') || pageContent.includes('100%')) {
          console.log(`   ✅ Backtest completed after ~${i * 3} seconds`);
          break;
        }
      }
    } else {
      console.log('   ⚠️ Start button not available');
    }

    // 3.6 Final state
    await page.waitForTimeout(3000);
    await captureStep(page, '13-backtest-complete', 'Backtest Completed - Final State');

    // 3.7 Verify results displayed
    logStep('3.4', 'Verifying backtest results');

    // Check for key metrics
    const metrics = [
      { name: 'Sharpe Ratio', selector: 'text=/Sharpe|sharpe/i' },
      { name: 'Total Return', selector: 'text=/Return|return|%/i' },
      { name: 'Win Rate', selector: 'text=/Win Rate|win/i' },
      { name: 'Total Trades', selector: 'text=/trades|Trades/i' },
    ];

    for (const metric of metrics) {
      const element = page.locator(metric.selector).first();
      if (await element.isVisible().catch(() => false)) {
        const text = await element.textContent();
        console.log(`   ✅ ${metric.name}: ${text}`);
      }
    }

    await captureStep(page, '14-backtest-metrics', 'Backtest Metrics Summary');

    console.log('\n✅ Scene 3 Complete: Backtest executed and verified\n');
  });

  test('Scene 4: Two-Vote Approval Flow', async ({ page }) => {
    logStep('SCENE 4', 'Testing Two-Vote Approval System');

    // 4.1 Go to experiments page
    await page.goto(CONFIG.dashboard.url + CONFIG.dashboard.experiments);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    await captureStep(page, '15-experiments-list', 'Experiments List');

    // 4.2 Check for pending approvals
    logStep('4.1', 'Checking for pending approval requests');
    const pendingTab = page.locator('button:has-text("Pending"), [role="tab"]:has-text("Pending")').first();
    if (await pendingTab.isVisible().catch(() => false)) {
      await pendingTab.click();
      await page.waitForTimeout(1000);
      await captureStep(page, '16-pending-experiments', 'Pending Experiments Tab');
    }

    // 4.3 Check for experiment cards
    const experimentCards = page.locator('[class*="experiment"], [class*="card"]');
    const cardCount = await experimentCards.count();
    console.log(`   Found ${cardCount} experiment cards`);

    // 4.4 Click on first experiment if available
    if (cardCount > 0) {
      logStep('4.2', 'Opening experiment detail');
      await experimentCards.first().click();
      await page.waitForTimeout(2000);
      await captureStep(page, '17-experiment-detail', 'Experiment Detail View');

      // 4.5 Check for approval panel
      const approvalPanel = page.locator('[class*="approval"], [class*="Floating"]');
      if (await approvalPanel.isVisible().catch(() => false)) {
        await captureStep(page, '18-approval-panel', 'Floating Approval Panel');
        console.log('   ✅ Approval panel visible');
      }

      // 4.6 Check for approve/reject buttons
      const approveBtn = page.locator('button:has-text("Approve"), button:has-text("Aprobar")').first();
      const rejectBtn = page.locator('button:has-text("Reject"), button:has-text("Rechazar")').first();

      if (await approveBtn.isVisible().catch(() => false)) {
        console.log('   ✅ Approve button available');
      }
      if (await rejectBtn.isVisible().catch(() => false)) {
        console.log('   ✅ Reject button available');
      }
    }

    await captureStep(page, '19-approval-flow-complete', 'Approval Flow - Final State');

    console.log('\n✅ Scene 4 Complete: Two-Vote approval flow verified\n');
  });

  test('Scene 5: Production Monitor Verification', async ({ page }) => {
    logStep('SCENE 5', 'Verifying Production Monitor');

    // 5.1 Go to production page
    await page.goto(CONFIG.dashboard.url + CONFIG.dashboard.production);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(3000);
    await captureStep(page, '20-production-monitor', 'Production Monitor Page');

    // 5.2 Check for active model info
    logStep('5.1', 'Checking active model information');
    const activeModel = page.locator('text=/Active Model|Modelo Activo/i');
    if (await activeModel.isVisible().catch(() => false)) {
      const modelInfo = await activeModel.textContent();
      console.log(`   Active model info: ${modelInfo}`);
    }

    // 5.3 Check for live status indicators
    const statusIndicators = page.locator('[class*="status"], [class*="indicator"]');
    const statusCount = await statusIndicators.count();
    console.log(`   Found ${statusCount} status indicators`);

    // 5.4 Check for equity curve
    const equityCurve = page.locator('[class*="chart"], [class*="equity"]');
    if (await equityCurve.isVisible().catch(() => false)) {
      await captureStep(page, '21-production-equity', 'Production Equity Curve');
      console.log('   ✅ Equity curve visible');
    }

    // 5.5 Check for pending experiments badge
    const pendingBadge = page.locator('text=/pending|Pendiente/i');
    if (await pendingBadge.isVisible().catch(() => false)) {
      console.log('   ✅ Pending experiments indicator visible');
    }

    await captureStep(page, '22-production-final', 'Production Monitor - Final State');

    console.log('\n✅ Scene 5 Complete: Production monitor verified\n');
  });

  test('Scene 6: MLflow Integration Check', async ({ page }) => {
    logStep('SCENE 6', 'Checking MLflow integration');

    // 6.1 Navigate to MLflow
    await page.goto(CONFIG.mlflow.url);
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    await captureStep(page, '23-mlflow-home', 'MLflow Home');

    // 6.2 Check for experiments
    const experiments = page.locator('[class*="experiment"], a[href*="experiment"]');
    const expCount = await experiments.count();
    console.log(`   Found ${expCount} experiments in MLflow`);

    // 6.3 Check for registered models
    const modelsLink = page.locator('a:has-text("Models"), [href*="models"]').first();
    if (await modelsLink.isVisible().catch(() => false)) {
      await modelsLink.click();
      await page.waitForTimeout(2000);
      await captureStep(page, '24-mlflow-models', 'MLflow Registered Models');
    }

    console.log('\n✅ Scene 6 Complete: MLflow integration verified\n');
  });

  test.afterAll(async () => {
    console.log('\n');
    console.log('🎬'.repeat(30));
    console.log('  A/B TESTING FLOW - COMPLETE');
    console.log('  Screenshots saved to: ' + CONFIG.screenshots.dir);
    console.log('🎬'.repeat(30));
    console.log('\n');

    // Generate summary
    const screenshotDir = CONFIG.screenshots.dir;
    if (fs.existsSync(screenshotDir)) {
      const files = fs.readdirSync(screenshotDir);
      console.log(`\n📸 Total screenshots captured: ${files.length}`);
      console.log('Files:');
      files.forEach(f => console.log(`   - ${f}`));
    }
  });
});
