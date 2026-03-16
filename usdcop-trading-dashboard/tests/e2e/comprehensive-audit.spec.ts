import { test, expect } from '@playwright/test';

const DIR = 'tests/e2e/screenshots/comprehensive-audit';

// Track console errors globally
const consoleErrors: { page: string; errors: string[] }[] = [];

test.describe('Comprehensive Feature Audit — All Pages', () => {
  // ============================================================
  // 1. HUB PAGE
  // ============================================================
  test.describe('1. Hub Page', () => {
    test('1.1 Full page render', async ({ page }) => {
      await page.goto('/hub', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/01-hub-full.png`, fullPage: true });

      // Verify page has substantial content
      const body = await page.content();
      expect(body.length).toBeGreaterThan(2000);
    });

    test('1.2 Navigation cards present', async ({ page }) => {
      await page.goto('/hub', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(2000);

      // Check for card-like elements or navigation links
      const links = page.locator('a, [role="link"], [role="button"]');
      const linkCount = await links.count();
      console.log(`Hub: ${linkCount} interactive elements found`);

      // Check for key section titles
      const titles = ['Trading Dashboard', 'Forecasting', 'Production', 'Analisis', 'Monitor'];
      for (const t of titles) {
        const el = page.locator(`text=/${t}/i`).first();
        const visible = await el.isVisible().catch(() => false);
        console.log(`  Card "${t}": ${visible ? 'VISIBLE' : 'not found'}`);
      }
    });

    test('1.3 Hub navbar', async ({ page }) => {
      await page.goto('/hub', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(1000);

      const nav = page.locator('nav').first();
      const hasNav = await nav.isVisible().catch(() => false);
      if (hasNav) {
        await nav.screenshot({ path: `${DIR}/01-hub-navbar.png` });
      }
      console.log(`Hub navbar: ${hasNav ? 'OK' : 'NOT FOUND'}`);
    });
  });

  // ============================================================
  // 2. FORECASTING PAGE
  // ============================================================
  test.describe('2. Forecasting Page', () => {
    test('2.1 Full page render', async ({ page }) => {
      await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(5000);
      await page.screenshot({ path: `${DIR}/02-forecasting-full.png`, fullPage: true });
    });

    test('2.2 Top section — model selector and KPIs', async ({ page }) => {
      await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(3000);

      // Screenshot viewport (top section)
      await page.screenshot({ path: `${DIR}/02-forecasting-top.png`, fullPage: false });

      // Check for model-related text
      const models = ['ridge', 'bayesian', 'xgboost', 'lightgbm', 'catboost', 'hybrid'];
      for (const m of models) {
        const el = page.locator(`text=/${m}/i`).first();
        const visible = await el.isVisible().catch(() => false);
        if (visible) console.log(`  Model "${m}": VISIBLE`);
      }
    });

    test('2.3 Forecasting charts area', async ({ page }) => {
      await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(4000);

      // Scroll to chart area
      await page.evaluate(() => window.scrollBy(0, 600));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/02-forecasting-charts.png`, fullPage: false });

      // Check for chart images or SVG
      const images = page.locator('img[src*="backtest"], img[src*="forward"], img[src*="forecast"]');
      const imgCount = await images.count();
      const svgs = page.locator('svg, .recharts-wrapper');
      const svgCount = await svgs.count();
      console.log(`Forecasting: ${imgCount} chart images, ${svgCount} SVGs`);
    });

    test('2.4 Forecasting bottom section', async ({ page }) => {
      await page.goto('/forecasting', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(3000);

      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/02-forecasting-bottom.png`, fullPage: false });
    });
  });

  // ============================================================
  // 3. DASHBOARD PAGE (Backtest Review + Approval)
  // ============================================================
  test.describe('3. Dashboard Page', () => {
    test('3.1 Full page render', async ({ page }) => {
      await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(10000);
      await page.screenshot({ path: `${DIR}/03-dashboard-full.png`, fullPage: true });
    });

    test('3.2 KPI cards area', async ({ page }) => {
      await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(8000);
      await page.screenshot({ path: `${DIR}/03-dashboard-kpis.png`, fullPage: false });

      // Check for KPI values
      const kpiLabels = ['Return', 'Sharpe', 'Win Rate', 'Drawdown', 'p-value', 'Retorno'];
      for (const k of kpiLabels) {
        const el = page.locator(`text=/${k}/i`).first();
        const visible = await el.isVisible().catch(() => false);
        if (visible) console.log(`  KPI "${k}": VISIBLE`);
      }
    });

    test('3.3 Backtest chart section', async ({ page }) => {
      await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(8000);

      await page.evaluate(() => window.scrollBy(0, 500));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/03-dashboard-chart.png`, fullPage: false });

      // Check for chart containers
      const charts = page.locator('canvas, svg, .tv-lightweight-charts, [class*="chart"]');
      const chartCount = await charts.count();
      console.log(`Dashboard charts: ${chartCount} chart elements`);
    });

    test('3.4 Trade table section', async ({ page }) => {
      await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(8000);

      await page.evaluate(() => window.scrollBy(0, 1200));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/03-dashboard-trades.png`, fullPage: false });

      // Check for table rows
      const rows = page.locator('table tr, [role="row"]');
      const rowCount = await rows.count();
      console.log(`Dashboard trade rows: ${rowCount}`);
    });

    test('3.5 Approval gates section', async ({ page }) => {
      await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(8000);

      await page.evaluate(() => window.scrollBy(0, 2000));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/03-dashboard-gates.png`, fullPage: false });

      // Check for gate-related text
      const gateTexts = ['PROMOTE', 'APPROVE', 'Aprobar', 'Gate', 'Sharpe', 'Significancia', 'PENDING'];
      for (const g of gateTexts) {
        const el = page.locator(`text=/${g}/i`).first();
        const visible = await el.isVisible().catch(() => false);
        if (visible) console.log(`  Gate "${g}": VISIBLE`);
      }
    });

    test('3.6 Dashboard bottom / RL section', async ({ page }) => {
      await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(8000);

      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/03-dashboard-bottom.png`, fullPage: false });
    });
  });

  // ============================================================
  // 4. PRODUCTION PAGE
  // ============================================================
  test.describe('4. Production Page', () => {
    test('4.1 Full page render', async ({ page }) => {
      await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(6000);
      await page.screenshot({ path: `${DIR}/04-production-full.png`, fullPage: true });
    });

    test('4.2 Production KPIs', async ({ page }) => {
      await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(5000);
      await page.screenshot({ path: `${DIR}/04-production-kpis.png`, fullPage: false });

      // Check for strategy name and metrics
      const labels = ['Smart Simple', 'Return', 'Sharpe', '2026', 'APPROVED', 'PENDING', 'Trades'];
      for (const l of labels) {
        const el = page.locator(`text=/${l}/i`).first();
        const visible = await el.isVisible().catch(() => false);
        if (visible) console.log(`  Prod "${l}": VISIBLE`);
      }
    });

    test('4.3 Production chart', async ({ page }) => {
      await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(5000);

      await page.evaluate(() => window.scrollBy(0, 500));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/04-production-chart.png`, fullPage: false });
    });

    test('4.4 Production trades & images', async ({ page }) => {
      await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(5000);

      await page.evaluate(() => window.scrollBy(0, 1200));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/04-production-trades.png`, fullPage: false });
    });

    test('4.5 Production bottom', async ({ page }) => {
      await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(5000);

      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/04-production-bottom.png`, fullPage: false });
    });
  });

  // ============================================================
  // 5. ANALYSIS PAGE — Core Sections
  // ============================================================
  test.describe('5. Analysis Page', () => {
    test('5.1 Full page render', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(8000);
      await page.screenshot({ path: `${DIR}/05-analysis-full.png`, fullPage: true });
    });

    test('5.2 Weekly summary header', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      // Wait for content
      await page.locator('text=/SEMANA|Analisis|Indicadores/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/05-analysis-header.png`, fullPage: false });

      // Check week selector
      const weekLabel = page.locator('text=/SEMANA\\s+\\d+/i');
      const visible = await weekLabel.first().isVisible().catch(() => false);
      if (visible) {
        const text = await weekLabel.first().textContent();
        console.log(`Analysis week selector: "${text}"`);
      }
    });

    test('5.3 Macro snapshot bar (DXY, VIX, Oil, EMBI)', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/Indicadores Macro|DXY|VIX/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(2000);

      // Look for macro indicators
      const macros = ['DXY', 'VIX', 'Oil', 'WTI', 'EMBI', 'Brent'];
      for (const m of macros) {
        const el = page.locator(`text=/${m}/i`).first();
        const visible = await el.isVisible().catch(() => false);
        if (visible) console.log(`  Macro bar "${m}": VISIBLE`);
      }

      await page.screenshot({ path: `${DIR}/05-analysis-macro-bar.png`, fullPage: false });
    });

    test('5.4 Macro chart grid', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/Indicadores Macro|Graficos/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(3000);

      // Scroll to chart grid
      await page.evaluate(() => window.scrollBy(0, 600));
      await page.waitForTimeout(3000);
      await page.screenshot({ path: `${DIR}/05-analysis-macro-charts.png`, fullPage: false });

      // Count chart SVGs
      const svgs = page.locator('.recharts-surface, .recharts-wrapper');
      const count = await svgs.count();
      console.log(`Analysis Recharts SVGs: ${count}`);
    });

    test('5.5 More macro charts (scrolled)', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/Indicadores Macro/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(3000);

      await page.evaluate(() => window.scrollBy(0, 1200));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/05-analysis-macro-charts-2.png`, fullPage: false });
    });

    test('5.6 Signal summary cards', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/SEMANA/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(3000);

      // Scroll to signal area
      await page.evaluate(() => window.scrollBy(0, 1800));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/05-analysis-signals.png`, fullPage: false });

      const signalTexts = ['H5', 'H1', 'Signal', 'Senal', 'SHORT', 'LONG', 'Direction'];
      for (const s of signalTexts) {
        const el = page.locator(`text=/${s}/i`).first();
        const visible = await el.isVisible().catch(() => false);
        if (visible) console.log(`  Signal "${s}": VISIBLE`);
      }
    });

    test('5.7 Daily timeline section', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/Timeline Diario|Indicadores/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(3000);

      await page.evaluate(() => window.scrollBy(0, 2400));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/05-analysis-timeline.png`, fullPage: false });

      // Check for daily entry elements
      const dayNames = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Monday', 'Tuesday'];
      for (const d of dayNames) {
        const el = page.locator(`text=/${d}/i`).first();
        const visible = await el.isVisible().catch(() => false);
        if (visible) console.log(`  Day "${d}": VISIBLE`);
      }
    });

    test('5.8 Daily timeline entries expanded', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/Timeline Diario|SEMANA/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(3000);

      await page.evaluate(() => window.scrollBy(0, 3200));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/05-analysis-timeline-entries.png`, fullPage: false });
    });

    test('5.9 Bottom of analysis page', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(5000);

      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/05-analysis-bottom.png`, fullPage: false });
    });
  });

  // ============================================================
  // 6. ANALYSIS — Week Navigation
  // ============================================================
  test.describe('6. Analysis Week Navigation', () => {
    test('6.1 Navigate to previous week', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/SEMANA\\s+\\d+/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(2000);

      const weekBefore = await page.locator('text=/SEMANA\\s+\\d+/i').first().textContent();
      console.log(`Current week: ${weekBefore}`);

      // Find and click previous button (arrow left)
      const prevBtn = page.locator('button:has-text("<"), button:has-text("\\u2190"), button:has-text("\\u25C0"), button[aria-label*="prev"], button[aria-label*="anterior"]').first();
      if (await prevBtn.isVisible().catch(() => false)) {
        await prevBtn.click();
        await page.waitForTimeout(4000);
        const weekAfter = await page.locator('text=/SEMANA\\s+\\d+/i').first().textContent().catch(() => 'N/A');
        console.log(`After prev: ${weekAfter}`);
        await page.screenshot({ path: `${DIR}/06-analysis-prev-week.png`, fullPage: false });
      } else {
        console.log('Previous button not found, trying generic approach');
        // Try any button that seems like a prev arrow
        const allBtns = page.locator('button');
        const count = await allBtns.count();
        for (let i = 0; i < count; i++) {
          const text = await allBtns.nth(i).textContent().catch(() => '');
          const ariaLabel = await allBtns.nth(i).getAttribute('aria-label').catch(() => '');
          if (text?.includes('<') || text?.includes('\u276E') || ariaLabel?.includes('prev')) {
            await allBtns.nth(i).click();
            await page.waitForTimeout(4000);
            await page.screenshot({ path: `${DIR}/06-analysis-prev-week.png`, fullPage: false });
            break;
          }
        }
      }
    });

    test('6.2 Navigate to oldest week (W01)', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/SEMANA\\s+\\d+/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(2000);

      // Click prev multiple times to go to oldest week
      for (let i = 0; i < 10; i++) {
        const prevBtn = page.locator('button').filter({ hasText: /[<\u276E\u2190\u25C0]/ }).first();
        if (await prevBtn.isVisible().catch(() => false)) {
          await prevBtn.click();
          await page.waitForTimeout(1500);
        }
      }

      await page.waitForTimeout(2000);
      const weekText = await page.locator('text=/SEMANA\\s+\\d+/i').first().textContent().catch(() => 'N/A');
      console.log(`Oldest week reached: ${weekText}`);
      await page.screenshot({ path: `${DIR}/06-analysis-oldest-week.png`, fullPage: false });
    });
  });

  // ============================================================
  // 7. ANALYSIS — Macro Detail Modal
  // ============================================================
  test.describe('7. Analysis Macro Interactions', () => {
    test('7.1 Click a macro chart to open detail modal', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.locator('text=/Indicadores Macro|DXY/i').first().waitFor({ timeout: 25000 });
      await page.waitForTimeout(3000);

      // Scroll to charts
      await page.evaluate(() => window.scrollBy(0, 600));
      await page.waitForTimeout(2000);

      // Try clicking on a chart card or DXY element
      const chartCard = page.locator('[class*="chart"], [class*="macro"]').filter({ hasText: /DXY/i }).first();
      if (await chartCard.isVisible().catch(() => false)) {
        await chartCard.click();
        await page.waitForTimeout(2000);

        // Check if modal opened
        const modal = page.locator('[role="dialog"], [class*="modal"], [class*="Modal"]');
        const modalVisible = await modal.first().isVisible().catch(() => false);
        if (modalVisible) {
          await page.screenshot({ path: `${DIR}/07-analysis-macro-modal.png`, fullPage: false });
          console.log('Macro detail modal: OPENED');

          // Close modal
          const closeBtn = page.locator('button:has-text("X"), button:has-text("\\u2715"), button[aria-label*="close"]').first();
          if (await closeBtn.isVisible().catch(() => false)) {
            await closeBtn.click();
            await page.waitForTimeout(1000);
          }
        } else {
          console.log('Macro detail modal: no modal appeared');
          await page.screenshot({ path: `${DIR}/07-analysis-macro-click-result.png`, fullPage: false });
        }
      } else {
        console.log('No clickable macro chart found');
      }
    });
  });

  // ============================================================
  // 8. ANALYSIS — Floating Chat Widget
  // ============================================================
  test.describe('8. Analysis Chat Widget', () => {
    test('8.1 Find and open chat widget', async ({ page }) => {
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(5000);

      // Look for floating chat button
      const chatBtn = page.locator('button[class*="chat"], button[class*="float"], [class*="FloatingChat"], button:has-text("Chat"), button:has-text("\\uD83D\\uDCAC")').first();
      const chatVisible = await chatBtn.isVisible().catch(() => false);

      if (chatVisible) {
        await chatBtn.click();
        await page.waitForTimeout(2000);
        await page.screenshot({ path: `${DIR}/08-analysis-chat-open.png`, fullPage: false });
        console.log('Chat widget: OPENED');

        // Try typing in chat
        const input = page.locator('input[type="text"], textarea, [contenteditable="true"]').last();
        if (await input.isVisible().catch(() => false)) {
          await input.fill('Cual es el outlook para el USDCOP?');
          await page.screenshot({ path: `${DIR}/08-analysis-chat-typing.png`, fullPage: false });
          console.log('Chat widget: input filled');
        }
      } else {
        console.log('Chat widget button: NOT FOUND');
        // Screenshot bottom-right corner where chat usually floats
        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
        await page.waitForTimeout(1000);
        await page.screenshot({ path: `${DIR}/08-analysis-no-chat.png`, fullPage: false });
      }
    });
  });

  // ============================================================
  // 9. API ENDPOINT VALIDATION
  // ============================================================
  test.describe('9. API Endpoints', () => {
    test('9.1 Production status API', async ({ request }) => {
      const resp = await request.get('/api/production/status');
      expect(resp.status()).toBe(200);
      const data = await resp.json();
      console.log(`Production status: ${data.status}, strategy: ${data.strategy}`);
      expect(data.status).toBeTruthy();
    });

    test('9.2 Analysis weeks API', async ({ request }) => {
      const resp = await request.get('/api/analysis/weeks');
      expect(resp.status()).toBe(200);
      const data = await resp.json();
      const weeks = data.weeks || data;
      console.log(`Analysis weeks: ${Array.isArray(weeks) ? weeks.length : 'N/A'} available`);
    });

    test('9.3 Analysis latest week data', async ({ request }) => {
      const indexResp = await request.get('/api/analysis/weeks');
      const index = await indexResp.json();
      const latest = (index.weeks || index)[0];

      if (latest) {
        const resp = await request.get(`/api/analysis/week/${latest.year}/${latest.week}`);
        expect(resp.status()).toBe(200);
        const data = await resp.json();

        console.log(`Latest week: ${latest.year}-W${latest.week}`);
        console.log(`  weekly_summary: ${data.weekly_summary ? 'OK' : 'MISSING'}`);
        console.log(`  daily_entries: ${data.daily_entries?.length || 0}`);
        console.log(`  macro_snapshots: ${data.macro_snapshots ? Object.keys(data.macro_snapshots).length : 0} vars`);
        console.log(`  macro_charts: ${data.macro_charts ? Object.keys(data.macro_charts).length : 0} charts`);
        console.log(`  signals: ${data.signals ? 'present' : 'MISSING'}`);

        if (data.weekly_summary) {
          console.log(`  headline: "${data.weekly_summary.headline?.substring(0, 60)}..."`);
          console.log(`  sentiment: ${data.weekly_summary.sentiment}`);
          console.log(`  markdown length: ${data.weekly_summary.markdown?.length || 0} chars`);
        }
      }
    });

    test('9.4 Analysis calendar API', async ({ request }) => {
      const resp = await request.get('/api/analysis/calendar');
      expect(resp.status()).toBe(200);
      const data = await resp.json();
      console.log(`Calendar: ${JSON.stringify(data).length} bytes`);
    });

    test('9.5 Production summary data files', async ({ request }) => {
      const files = [
        '/data/production/summary.json',
        '/data/production/summary_2025.json',
        '/data/production/approval_state.json',
      ];
      for (const f of files) {
        const resp = await request.get(f);
        console.log(`${f}: ${resp.status()}`);
        if (resp.status() === 200) {
          const data = await resp.json();
          if (f.includes('summary')) {
            console.log(`  strategy: ${data.strategy_id}, year: ${data.year}`);
          }
        }
      }
    });

    test('9.6 Trade files', async ({ request }) => {
      const files = [
        '/data/production/trades/smart_simple_v11.json',
        '/data/production/trades/smart_simple_v11_2025.json',
      ];
      for (const f of files) {
        const resp = await request.get(f);
        console.log(`${f}: ${resp.status()}`);
        if (resp.status() === 200) {
          const data = await resp.json();
          console.log(`  trades: ${data.trades?.length || 0}, summary return: ${data.summary?.total_return_pct}%`);
        }
      }
    });
  });

  // ============================================================
  // 10. RESPONSIVE & MOBILE VIEW
  // ============================================================
  test.describe('10. Responsive Views', () => {
    test('10.1 Mobile Hub (375px)', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 });
      await page.goto('/hub', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/10-mobile-hub.png`, fullPage: true });
    });

    test('10.2 Mobile Dashboard (375px)', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 });
      await page.goto('/dashboard', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(8000);
      await page.screenshot({ path: `${DIR}/10-mobile-dashboard.png`, fullPage: true });
    });

    test('10.3 Mobile Analysis (375px)', async ({ page }) => {
      await page.setViewportSize({ width: 375, height: 812 });
      await page.goto('/analysis', { waitUntil: 'load', timeout: 60000 });
      await page.waitForTimeout(6000);
      await page.screenshot({ path: `${DIR}/10-mobile-analysis.png`, fullPage: true });
    });

    test('10.4 Tablet Production (768px)', async ({ page }) => {
      await page.setViewportSize({ width: 768, height: 1024 });
      await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(5000);
      await page.screenshot({ path: `${DIR}/10-tablet-production.png`, fullPage: true });
    });
  });

  // ============================================================
  // 11. CONSOLE ERROR AUDIT
  // ============================================================
  test.describe('11. Console Error Audit', () => {
    test('11.1 Collect errors across all pages', async ({ page }) => {
      const pageErrors: { url: string; errors: string[] }[] = [];
      const routes = ['/hub', '/forecasting', '/dashboard', '/production', '/analysis'];

      for (const route of routes) {
        const errors: string[] = [];
        page.on('console', msg => {
          if (msg.type() === 'error') {
            const text = msg.text();
            // Filter noise
            if (text.includes('MetaMask') || text.includes('ethereum') || text.includes('favicon')) return;
            if (text.includes('ws://localhost:8000')) return;
            if (text.includes('net::ERR_CONNECTION_REFUSED')) return;
            errors.push(text.substring(0, 200));
          }
        });

        await page.goto(route, { waitUntil: 'load', timeout: 60000 });
        await page.waitForTimeout(5000);

        pageErrors.push({ url: route, errors: [...errors] });
        page.removeAllListeners('console');
      }

      console.log('\n=== Console Error Summary ===');
      for (const { url, errors } of pageErrors) {
        if (errors.length === 0) {
          console.log(`  ${url}: CLEAN (no errors)`);
        } else {
          console.log(`  ${url}: ${errors.length} errors`);
          errors.slice(0, 3).forEach(e => console.log(`    - ${e}`));
        }
      }
    });
  });

  // ============================================================
  // 12. CROSS-PAGE NAVIGATION
  // ============================================================
  test.describe('12. Navigation Flow', () => {
    test('12.1 Navigate Hub -> Dashboard -> Production -> Analysis', async ({ page }) => {
      // Start at Hub
      await page.goto('/hub', { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(2000);
      await page.screenshot({ path: `${DIR}/12-nav-01-hub.png`, fullPage: false });

      // Navigate to Dashboard via navbar
      const dashLink = page.locator('nav a[href*="dashboard"], nav button:has-text("Dashboard")').first();
      if (await dashLink.isVisible().catch(() => false)) {
        await dashLink.click();
        await page.waitForTimeout(5000);
        await page.screenshot({ path: `${DIR}/12-nav-02-dashboard.png`, fullPage: false });
      } else {
        await page.goto('/dashboard', { waitUntil: 'load', timeout: 30000 });
        await page.waitForTimeout(5000);
        await page.screenshot({ path: `${DIR}/12-nav-02-dashboard.png`, fullPage: false });
      }

      // Navigate to Production
      const prodLink = page.locator('nav a[href*="production"], nav button:has-text("Production")').first();
      if (await prodLink.isVisible().catch(() => false)) {
        await prodLink.click();
        await page.waitForTimeout(5000);
        await page.screenshot({ path: `${DIR}/12-nav-03-production.png`, fullPage: false });
      } else {
        await page.goto('/production', { waitUntil: 'networkidle', timeout: 30000 });
        await page.waitForTimeout(5000);
        await page.screenshot({ path: `${DIR}/12-nav-03-production.png`, fullPage: false });
      }

      // Navigate to Analysis
      const analysisLink = page.locator('nav a[href*="analysis"], nav button:has-text("Analysis")').first();
      if (await analysisLink.isVisible().catch(() => false)) {
        await analysisLink.click();
        await page.waitForTimeout(5000);
        await page.screenshot({ path: `${DIR}/12-nav-04-analysis.png`, fullPage: false });
      } else {
        await page.goto('/analysis', { waitUntil: 'load', timeout: 30000 });
        await page.waitForTimeout(5000);
        await page.screenshot({ path: `${DIR}/12-nav-04-analysis.png`, fullPage: false });
      }

      console.log('Navigation flow completed: Hub -> Dashboard -> Production -> Analysis');
    });
  });
});
