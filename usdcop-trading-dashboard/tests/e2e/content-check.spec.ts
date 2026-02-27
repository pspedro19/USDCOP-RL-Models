import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5000';

test.describe('Dashboard page content', () => {
  test('renders backtest metrics and strategy selector', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(`PAGE_ERROR: ${err.message}`));

    await page.goto(`${BASE_URL}/dashboard`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(5000);

    // Screenshot
    await page.screenshot({ path: 'test-results/dashboard-full.png', fullPage: true });

    // Check page loaded
    const body = await page.textContent('body');
    console.log('\n=== DASHBOARD PAGE ===');
    console.log('Body length:', body?.length);
    console.log('Console errors:', errors.length);
    errors.forEach(e => console.log('  ERROR:', e));

    // Look for key content
    const hasStrategy = body?.includes('Smart Simple') || body?.includes('smart_simple');
    const hasSharpe = body?.includes('Sharpe') || body?.includes('sharpe');
    const hasReturn = body?.includes('Return') || body?.includes('return') || body?.includes('%');
    const hasApproval = body?.includes('APPROVED') || body?.includes('PROMOTE') || body?.includes('Approve');
    const hasTrades = body?.includes('trade') || body?.includes('Trade') || body?.includes('SHORT') || body?.includes('LONG');

    console.log('Has strategy name:', hasStrategy);
    console.log('Has Sharpe:', hasSharpe);
    console.log('Has Return %:', hasReturn);
    console.log('Has approval:', hasApproval);
    console.log('Has trades:', hasTrades);

    // Check for dropdown/selector
    const selectors = await page.$$('[role="combobox"], select, [data-strategy], button:has-text("Smart Simple")');
    console.log('Strategy selectors found:', selectors.length);

    // Look for specific metric values
    const has2307 = body?.includes('23.07');
    const has3822 = body?.includes('3.822') || body?.includes('3.82');
    console.log('Has 23.07% return:', has2307);
    console.log('Has 3.822 Sharpe:', has3822);

    // Check for visible text snippets
    const snippets = ['Smart Simple', 'Sharpe', 'Backtest', '2025', 'Gate', 'Approve'];
    for (const s of snippets) {
      const found = body?.includes(s);
      console.log(`  "${s}": ${found}`);
    }
  });
});

test.describe('Production page content', () => {
  test('renders 2026 trades and metrics', async ({ page }) => {
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(`PAGE_ERROR: ${err.message}`));

    await page.goto(`${BASE_URL}/production`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(5000);

    // Screenshot
    await page.screenshot({ path: 'test-results/production-full.png', fullPage: true });

    const body = await page.textContent('body');
    console.log('\n=== PRODUCTION PAGE ===');
    console.log('Body length:', body?.length);
    console.log('Console errors:', errors.length);
    errors.forEach(e => console.log('  ERROR:', e));

    // Look for 2026 content
    const has2026 = body?.includes('2026');
    const hasTrades = body?.includes('trade') || body?.includes('Trade');
    const hasEquity = body?.includes('10,259') || body?.includes('10259') || body?.includes('$10');
    const hasReturn = body?.includes('2.59') || body?.includes('+2.59');
    const hasSharpe = body?.includes('5.602') || body?.includes('5.60');
    const hasShort = body?.includes('SHORT');
    const hasLong = body?.includes('LONG');
    const hasApproved = body?.includes('APPROVED') || body?.includes('Approved');

    console.log('Has 2026:', has2026);
    console.log('Has trades:', hasTrades);
    console.log('Has equity ~10259:', hasEquity);
    console.log('Has return 2.59%:', hasReturn);
    console.log('Has Sharpe 5.602:', hasSharpe);
    console.log('Has SHORT:', hasShort);
    console.log('Has LONG:', hasLong);
    console.log('Has APPROVED badge:', hasApproved);

    // Check network requests
    const requests: string[] = [];
    page.on('response', (res) => {
      if (res.url().includes('/data/production/') || res.url().includes('/api/production/')) {
        requests.push(`${res.status()} ${res.url()}`);
      }
    });

    // Reload to capture requests
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);
    console.log('\nNetwork requests:');
    requests.forEach(r => console.log('  ', r));

    // Check for visible text
    const snippets = ['Production', '2026', 'Smart Simple', 'Sharpe', 'trade', 'Deploy'];
    for (const s of snippets) {
      const found = body?.includes(s);
      console.log(`  "${s}": ${found}`);
    }
  });
});

test.describe('Forecasting page content', () => {
  test('renders model zoo with 9 models', async ({ page }) => {
    await page.goto(`${BASE_URL}/forecasting`, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(3000);

    await page.screenshot({ path: 'test-results/forecasting-full.png', fullPage: true });

    const body = await page.textContent('body');
    console.log('\n=== FORECASTING PAGE ===');
    console.log('Body length:', body?.length);

    const models = ['Ridge', 'Bayesian', 'ARD', 'XGBoost', 'LightGBM', 'CatBoost'];
    for (const m of models) {
      console.log(`  Model "${m}": ${body?.includes(m)}`);
    }

    const hasHorizon = body?.includes('H=') || body?.includes('horizon');
    const hasDA = body?.includes('Direction') || body?.includes('DA') || body?.includes('Accuracy');
    const hasWeek = body?.includes('W0') || body?.includes('2026');
    console.log('Has horizons:', hasHorizon);
    console.log('Has DA/Accuracy:', hasDA);
    console.log('Has weeks:', hasWeek);
  });
});
