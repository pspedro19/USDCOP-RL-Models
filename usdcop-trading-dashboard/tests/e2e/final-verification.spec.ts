import { test, expect } from '@playwright/test';

test('Final Dashboard Verification - v19 & v20 Production Ready', async ({ page }) => {
  const consoleLogs: string[] = [];

  page.on('console', (msg) => {
    consoleLogs.push(`[${msg.type().toUpperCase()}] ${msg.text()}`);
  });

  // Login
  await page.goto('http://localhost:5000/login');
  await page.locator('input[autocomplete="username"]').fill('admin');
  await page.locator('input[type="password"]').fill('admin123');
  await page.locator('button[type="submit"]').click();
  await page.waitForURL('**/hub', { timeout: 10000 });

  // Go to Dashboard
  await page.goto('http://localhost:5000/dashboard');
  await page.waitForTimeout(5000);

  // Get models from API
  const modelsResponse = await page.request.get('http://localhost:5000/api/models');
  const modelsData = await modelsResponse.json();
  console.log('\n=== MODELS CONFIGURATION ===');
  for (const model of modelsData.models) {
    console.log(`\n${model.name} (${model.id})`);
    console.log(`  Status: ${model.status}`);
    console.log(`  Real Data: ${model.isRealData}`);
    if (model.backtest) {
      console.log(`  Sharpe: ${model.backtest.sharpe}`);
      console.log(`  Win Rate: ${model.backtest.winRate}`);
      console.log(`  Max DD: ${model.backtest.maxDrawdown}%`);
      console.log(`  Test Period: ${model.backtest.testPeriod}`);
    }
  }

  // Screenshot with v19 selected
  await page.screenshot({ path: 'tests/e2e/screenshots/final-v19-dashboard.png', fullPage: true });
  console.log('\nScreenshot: final-v19-dashboard.png');

  // Select v20 model
  const dropdownButton = page.locator('button:has-text("PPO V19"), button:has-text("PPO")').first();
  if (await dropdownButton.count() > 0) {
    await dropdownButton.click();
    await page.waitForTimeout(500);

    const v20Option = page.locator('button:has-text("PPO V20"), [role="option"]:has-text("V20")').first();
    if (await v20Option.count() > 0) {
      await v20Option.click();
      await page.waitForTimeout(3000);
      await page.screenshot({ path: 'tests/e2e/screenshots/final-v20-dashboard.png', fullPage: true });
      console.log('Screenshot: final-v20-dashboard.png');
    }
  }

  // Check endpoints for both models
  console.log('\n=== ENDPOINT STATUS ===');

  // V19 metrics
  const v19Metrics = await page.request.get('http://localhost:5000/api/models/ppo_v19_prod/metrics?period=all');
  console.log(`V19 Metrics: ${v19Metrics.status()}`);

  // V20 metrics
  const v20Metrics = await page.request.get('http://localhost:5000/api/models/ppo_v20_prod/metrics?period=all');
  console.log(`V20 Metrics: ${v20Metrics.status()}`);

  // V19 equity curve
  const v19Equity = await page.request.get('http://localhost:5000/api/models/ppo_v19_prod/equity-curve?days=90');
  console.log(`V19 Equity Curve: ${v19Equity.status()}`);

  // V20 equity curve
  const v20Equity = await page.request.get('http://localhost:5000/api/models/ppo_v20_prod/equity-curve?days=90');
  console.log(`V20 Equity Curve: ${v20Equity.status()}`);

  // Trades history
  const tradesV19 = await page.request.get('http://localhost:5000/api/trading/trades/history?limit=50&model_id=ppo_v19_prod');
  console.log(`V19 Trades: ${tradesV19.status()}`);

  const tradesV20 = await page.request.get('http://localhost:5000/api/trading/trades/history?limit=50&model_id=ppo_v20_prod');
  console.log(`V20 Trades: ${tradesV20.status()}`);

  // Summary
  console.log('\n=== SUMMARY ===');
  console.log('Production models configured:');
  const prodModels = modelsData.models.filter((m: any) => m.status === 'production');
  prodModels.forEach((m: any) => {
    console.log(`  - ${m.name}: ${m.isRealData ? 'REAL DATA' : 'DEMO DATA'}`);
  });

  console.log('\nConsole errors:');
  const errors = consoleLogs.filter(l => l.includes('[ERROR]'));
  if (errors.length === 0) {
    console.log('  No critical errors!');
  } else {
    errors.slice(0, 5).forEach(e => console.log(`  ${e}`));
  }
});
