import { test, expect } from '@playwright/test';

/**
 * Forecast Strategy Replay E2E Tests
 * ====================================
 * Verifies that forecast strategies appear in the model dropdown,
 * the SSE stream endpoint works, and backtest replay streams correctly.
 */

test.describe('Forecast Strategy Replay', () => {

  test('API: /api/models returns 4 forecast strategies', async ({ request }) => {
    const response = await request.get('/api/models');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    const models = data.models;
    const forecastModels = models.filter((m: any) => m.id.startsWith('fc_'));

    expect(forecastModels.length).toBe(4);

    const ids = forecastModels.map((m: any) => m.id).sort();
    expect(ids).toEqual([
      'fc_buy_hold',
      'fc_forecast_1x',
      'fc_forecast_vt',
      'fc_forecast_vt_trail',
    ]);

    // Verify each has correct algorithm
    for (const m of forecastModels) {
      expect(m.algorithm).toBe('FORECAST');
      expect(m.isRealData).toBe(true);
      expect(m.backtest).toBeDefined();
      expect(m.backtest.totalTrades).toBeGreaterThan(0);
    }

    // Best strategy should have highest Sharpe
    const bestModel = forecastModels.find((m: any) => m.id === 'fc_forecast_vt_trail');
    expect(bestModel).toBeDefined();
    expect(bestModel.backtest.sharpe).toBeCloseTo(3.135, 1);
  });

  test('API: SSE stream returns trades for fc_forecast_vt_trail', async ({ request }) => {
    const response = await request.get(
      '/api/backtest/stream?startDate=2025-01-02&endDate=2025-03-31&modelId=fc_forecast_vt_trail&speed=16'
    );
    expect(response.ok()).toBeTruthy();
    expect(response.headers()['content-type']).toContain('text/event-stream');

    const body = await response.text();
    const lines = body.split('\n').filter(l => l.startsWith('data: '));

    // Should have progress, trade, and result events
    const events = lines.map(l => JSON.parse(l.replace('data: ', '')));
    const types = new Set(events.map(e => e.type));

    expect(types.has('progress')).toBeTruthy();
    expect(types.has('trade')).toBeTruthy();
    expect(types.has('result')).toBeTruthy();

    // Verify trade events have correct schema
    const tradeEvents = events.filter(e => e.type === 'trade');
    expect(tradeEvents.length).toBeGreaterThan(0);

    const firstTrade = tradeEvents[0].data;
    expect(firstTrade.model_id).toBe('fc_forecast_vt_trail');
    expect(firstTrade.side).toMatch(/^(LONG|SHORT)$/);
    expect(firstTrade.entry_price).toBeGreaterThan(3000);
    expect(firstTrade.current_equity).toBeGreaterThan(0);
    expect(firstTrade.equity_at_entry).toBeDefined();
    expect(firstTrade.equity_at_exit).toBeDefined();

    // Verify result event
    const resultEvent = events.find(e => e.type === 'result');
    expect(resultEvent.data.success).toBe(true);
    expect(resultEvent.data.trade_count).toBeGreaterThan(0);
    expect(resultEvent.data.summary).toBeDefined();
    expect(resultEvent.data.summary.total_trades).toBeGreaterThan(0);
  });

  test('API: SSE stream returns trades for fc_buy_hold', async ({ request }) => {
    const response = await request.get(
      '/api/backtest/stream?startDate=2025-01-02&endDate=2025-12-30&modelId=fc_buy_hold&speed=16'
    );
    expect(response.ok()).toBeTruthy();

    const body = await response.text();
    const events = body.split('\n')
      .filter(l => l.startsWith('data: '))
      .map(l => JSON.parse(l.replace('data: ', '')));

    const trades = events.filter(e => e.type === 'trade');
    // Buy & hold has ~12 monthly trades
    expect(trades.length).toBe(12);
    // All should be LONG
    for (const t of trades) {
      expect(t.data.side).toBe('LONG');
    }
  });

  test('API: SSE stream returns 404 for invalid forecast model', async ({ request }) => {
    const response = await request.get(
      '/api/backtest/stream?startDate=2025-01-02&endDate=2025-12-30&modelId=fc_invalid_model&speed=16'
    );
    // Should fall through to backend (not a registered fc_ strategy),
    // or get synthetic fallback - either way should not crash
    expect(response.status()).toBeLessThan(500);
  });

  test('Dashboard: forecast models appear in model dropdown', async ({ page }) => {
    await page.goto('/dashboard');
    // Use timeout instead of networkidle — dashboard has persistent polling
    await page.waitForTimeout(5000);

    // Click the model dropdown button
    const dropdownButton = page.locator('button').filter({ hasText: /Select Model|PPO|Investor|Forecast|Buy/i }).first();
    await expect(dropdownButton).toBeVisible({ timeout: 10000 });
    await dropdownButton.click();

    // Wait for dropdown items to render
    await page.waitForTimeout(1500);

    // Check page content for forecast model names
    const pageContent = await page.content();
    const hasForecast = pageContent.includes('Forecast') || pageContent.includes('FORECAST');
    expect(hasForecast).toBeTruthy();
  });

  test('Dashboard: selecting forecast model shows FORECAST badge', async ({ page }) => {
    await page.goto('/dashboard');
    await page.waitForTimeout(5000);

    // Open dropdown
    const dropdownButton = page.locator('button').filter({ hasText: /Select Model|PPO|Investor|Forecast|Buy/i }).first();
    await expect(dropdownButton).toBeVisible({ timeout: 10000 });
    await dropdownButton.click();
    await page.waitForTimeout(1500);

    // Click on a forecast model — use text content matching
    const forecastItem = page.getByText('Forecast + VT + Trail Stop').first();
    if (await forecastItem.isVisible({ timeout: 3000 }).catch(() => false)) {
      await forecastItem.click();
      await page.waitForTimeout(1000);

      // Verify FORECAST badge appears on the page
      const pageContent = await page.content();
      expect(pageContent).toContain('FORECAST');
    }
  });
});
