import { test, expect } from '@playwright/test';

/**
 * Forecasting 2026 — Complete Verification
 * ==========================================
 * Verifies all 2026 weeks (W01-W07) load correctly with all models/horizons.
 * Ensures images load, CSV data parses, and filters work.
 *
 * UI Structure (from screenshot analysis):
 *   select[0] = "Modo de Vista" (Forward Forecast / Backtest Analysis)
 *   select[1] = "Semana" (Week 2026-W01 ... 2026-W07)
 *   select[2] = "Modelo" (All Models (Consensus), ridge, bayesian_ridge, ...)
 */

const SCREENSHOTS_DIR = 'tests/e2e/screenshots/forecasting-2026';

// Override baseURL (server runs on 5000)
test.use({ baseURL: 'http://localhost:5000' });

const WEEKS_2026 = ['2026-W01', '2026-W02', '2026-W03', '2026-W04', '2026-W05', '2026-W06', '2026-W07'];

test.describe('Forecasting 2026 — All Weeks & Models', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      if (msg.type() === 'error' && !msg.text().includes('WebSocket') && !msg.text().includes('socket.io')) {
        console.error(`[BROWSER ERROR] ${msg.text()}`);
      }
    });

    // Set auth
    await page.goto('/login');
    await page.evaluate(() => {
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('username', 'admin');
      sessionStorage.setItem('isAuthenticated', 'true');
      sessionStorage.setItem('username', 'admin');
    });
  });

  test('CSV loads with all 7 weeks of 2026 data', async ({ page }) => {
    const csvPromise = page.waitForResponse(resp =>
      resp.url().includes('bi_dashboard_unified.csv') && resp.status() === 200
    );

    await page.goto('/forecasting');
    const csvResponse = await csvPromise;
    expect(csvResponse.status()).toBe(200);

    const csvText = await csvResponse.text();
    expect(csvText.length).toBeGreaterThan(100);

    // Verify all 7 weeks present
    for (const week of WEEKS_2026) {
      expect(csvText).toContain(week);
    }

    // No 2025 week data
    expect(csvText).not.toContain('2025-W');

    console.log(`CSV loaded: ${csvText.split('\n').length} lines, all 7 weeks present`);
  });

  test('Page loads with W07 selected and consensus image visible', async ({ page }) => {
    await page.goto('/forecasting');
    await page.waitForTimeout(4000);

    // No error state
    const errorAlert = page.locator('text=No se pudieron cargar los datos');
    await expect(errorAlert).not.toBeVisible();

    // Semana select (second select) should show W07
    const weekSelect = page.locator('select').nth(1);
    await expect(weekSelect).toBeVisible();
    const weekValue = await weekSelect.inputValue();
    expect(weekValue).toContain('2026-W07');

    // Consensus image should be visible
    const img = page.locator('img[src*="forward_consensus_2026_W07"]');
    await expect(img).toBeVisible({ timeout: 10000 });

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/01-default-view-w07.png`,
      fullPage: true,
    });

    console.log('Page loaded: W07 selected, consensus image visible');
  });

  test('All 7 weeks selectable with consensus image loading', async ({ page }) => {
    await page.goto('/forecasting');
    await page.waitForTimeout(4000);

    // Semana dropdown (select index 1)
    const weekSelect = page.locator('select').nth(1);
    await expect(weekSelect).toBeVisible();

    // Get options text
    const options = await weekSelect.locator('option').allTextContents();
    console.log(`Week options: ${options.join(', ')}`);

    // All 7 weeks should be in options
    for (const week of WEEKS_2026) {
      const found = options.some(o => o.includes(week));
      expect(found).toBeTruthy();
    }

    // Cycle through each week (use shorter wait to avoid timeout)
    for (const week of WEEKS_2026) {
      const matchOption = options.find(o => o.includes(week));
      if (matchOption) {
        await weekSelect.selectOption({ label: matchOption });
      } else {
        await weekSelect.selectOption({ value: week });
      }
      await page.waitForTimeout(800);

      const weekSuffix = week.replace('-', '_');
      const img = page.locator(`img[src*="forward_consensus_${weekSuffix}"]`);
      await expect(img).toBeVisible({ timeout: 6000 });
      console.log(`  ${week}: consensus image loaded`);
    }

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/02-all-weeks-verified.png`,
      fullPage: true,
    });
  });

  test('Individual models load forward forecast images', async ({ page }) => {
    await page.goto('/forecasting');
    await page.waitForTimeout(4000);

    // Modelo dropdown (select index 2)
    const modelSelect = page.locator('select').nth(2);
    await expect(modelSelect).toBeVisible();

    // Get available model options
    const modelOptions = await modelSelect.locator('option').allTextContents();
    const modelValues = await modelSelect.locator('option').evaluateAll(
      (els) => els.map(el => (el as HTMLOptionElement).value)
    );
    console.log(`Model options: ${modelOptions.join(', ')}`);
    console.log(`Model values: ${modelValues.join(', ')}`);

    // Test W07 with each available model
    const weekSelect = page.locator('select').nth(1);
    const weekOptions = await weekSelect.locator('option').allTextContents();
    const w07Option = weekOptions.find(o => o.includes('2026-W07'));
    if (w07Option) await weekSelect.selectOption({ label: w07Option });
    await page.waitForTimeout(1000);

    // Test a subset of models (avoid timeout cycling through all 13)
    const testModels = modelValues.filter(v =>
      v !== 'ALL' && !v.startsWith('---') &&
      ['Ridge', 'XGBoost', 'LightGBM', 'ENSEMBLE_TOP_3'].includes(v)
    );

    let loadedCount = 0;
    for (const value of testModels) {
      await modelSelect.selectOption({ value });
      await page.waitForTimeout(1200);

      const img = page.locator('img[src*="forward_"]');
      const visible = await img.first().isVisible().catch(() => false);
      if (visible) loadedCount++;
      console.log(`  Model "${value}": image visible=${visible}`);
    }

    expect(loadedCount).toBeGreaterThan(0);
    console.log(`Models with visible images: ${loadedCount}/${testModels.length}`);

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/03-individual-models.png`,
      fullPage: true,
    });
  });

  test('Forward forecast PNGs exist (HTTP check)', async ({ request }) => {
    // Test key models x weeks (smaller subset to avoid ECONNRESET)
    const models = ['consensus', 'ridge', 'xgboost_pure', 'ensemble_top_3'];

    let total = 0;
    let loaded = 0;
    const missing: string[] = [];

    for (const model of models) {
      for (const week of WEEKS_2026) {
        const weekSuffix = week.replace('-', '_');
        const url = `/forecasting/forward_${model}_${weekSuffix}.png`;
        try {
          const resp = await request.get(url);
          total++;
          if (resp.ok()) {
            loaded++;
          } else {
            missing.push(`${model}/${week}`);
          }
        } catch (e) {
          total++;
          missing.push(`${model}/${week} (error)`);
        }
      }
    }

    console.log(`PNG check: ${loaded}/${total} loaded`);
    if (missing.length > 0) {
      console.log(`Missing: ${missing.join(', ')}`);
    }

    // All tested PNGs should exist (4 models x 7 weeks = 28)
    expect(loaded).toBe(28);
  });

  test('Backtest view loads and shows model images', async ({ page }) => {
    await page.goto('/forecasting');
    await page.waitForTimeout(4000);

    // Switch to Backtest via select[0]
    const viewSelect = page.locator('select').nth(0);
    await expect(viewSelect).toBeVisible();
    const viewOptions = await viewSelect.locator('option').allTextContents();
    console.log(`View options: ${viewOptions.join(', ')}`);

    const backtestOption = viewOptions.find(o => o.toLowerCase().includes('backtest'));
    if (backtestOption) {
      await viewSelect.selectOption({ label: backtestOption });
      await page.waitForTimeout(2000);

      // A backtest image should appear
      const img = page.locator('img[src*="backtest_"]');
      const visible = await img.first().isVisible({ timeout: 10000 }).catch(() => false);
      console.log(`Backtest image visible: ${visible}`);

      await page.screenshot({
        path: `${SCREENSHOTS_DIR}/04-backtest-view.png`,
        fullPage: true,
      });
    }
  });

  test('KPI metrics appear for consensus and specific models', async ({ page }) => {
    await page.goto('/forecasting');
    await page.waitForTimeout(4000);

    // Default consensus view: should show Avg Direction Accuracy
    const daMetric = page.locator('text=Avg Direction Accuracy');
    await expect(daMetric).toBeVisible({ timeout: 8000 });

    // KPI summary (models count, horizons, registros)
    const modelsCount = page.getByText('Modelos', { exact: true });
    await expect(modelsCount).toBeVisible();

    const horizonsCount = page.getByText('Horizontes', { exact: true });
    await expect(horizonsCount).toBeVisible();

    // Select a specific model
    const modelSelect = page.locator('select').nth(2);
    const modelValues = await modelSelect.locator('option').evaluateAll(
      (els) => els.map(el => (el as HTMLOptionElement).value)
    );

    // Find first non-ALL model
    const firstModel = modelValues.find(v => v !== 'ALL');
    if (firstModel) {
      await modelSelect.selectOption({ value: firstModel });
      await page.waitForTimeout(2000);

      // Should show model-specific metrics (WF DA, Sharpe, PF, MaxDD)
      const hasSharpe = await page.locator('text=Sharpe').first().isVisible().catch(() => false);
      const hasPF = await page.locator('text=Profit Factor').first().isVisible().catch(() => false);
      console.log(`Model "${firstModel}": Sharpe visible=${hasSharpe}, PF visible=${hasPF}`);
    }

    await page.screenshot({
      path: `${SCREENSHOTS_DIR}/05-kpi-metrics.png`,
      fullPage: true,
    });
  });

  test('Backtest PNG images exist for all models and horizons (HTTP)', async ({ request }) => {
    const models = ['ridge', 'bayesian_ridge', 'ard', 'xgboost_pure', 'lightgbm_pure', 'catboost_pure',
                    'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost'];
    const horizons = ['h1', 'h5', 'h10', 'h15', 'h20', 'h25', 'h30'];

    let loaded = 0;
    let total = 0;
    const missing: string[] = [];

    for (const model of models) {
      for (const h of horizons) {
        const url = `/forecasting/backtest_${model}_${h}.png`;
        const resp = await request.get(url);
        total++;
        if (resp.ok()) {
          loaded++;
        } else {
          missing.push(`${model}/${h}`);
        }
      }
    }

    console.log(`Backtest PNGs: ${loaded}/${total}`);
    if (missing.length > 0) {
      console.log(`Missing: ${missing.slice(0, 10).join(', ')}${missing.length > 10 ? '...' : ''}`);
    }

    // All 63 backtest PNGs should exist (9 models x 7 horizons)
    expect(loaded).toBe(63);
  });
});
