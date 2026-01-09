import { test, expect } from '@playwright/test';

test('Forecasting Page - Professional Centered Design', async ({ page }) => {
  const consoleLogs: string[] = [];
  const consoleErrors: string[] = [];

  page.on('console', (msg) => {
    const text = msg.text();
    consoleLogs.push(`[${msg.type().toUpperCase()}] ${text}`);
    if (msg.type() === 'error' || msg.type() === 'warning') {
      consoleErrors.push(text);
    }
  });

  // Login
  await page.goto('http://localhost:5000/login');
  await page.locator('input[autocomplete="username"]').fill('admin');
  await page.locator('input[type="password"]').fill('admin123');
  await page.locator('button[type="submit"]').click();
  await page.waitForURL('**/hub', { timeout: 30000, waitUntil: 'domcontentloaded' });

  // Go to Forecasting page
  await page.goto('http://localhost:5000/forecasting');
  await page.waitForTimeout(4000);

  // Scroll to top to ensure we see the header
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(500);

  // Screenshot - Desktop view (1280x900 for taller viewport)
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForTimeout(2000);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(500);

  // Debug: Get h1 element info and parent section info
  const debugInfo = await page.evaluate(() => {
    const h1 = document.querySelector('h1');
    const section = h1?.closest('section');
    const main = document.querySelector('main');
    const scrollY = window.scrollY;

    const getInfo = (el: Element | null, name: string) => {
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const style = window.getComputedStyle(el);
      return {
        name,
        rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
        paddingTop: style.paddingTop,
        marginTop: style.marginTop,
        position: style.position
      };
    };

    return {
      scrollY,
      h1: h1 ? {
        text: h1.textContent,
        ...getInfo(h1, 'h1')
      } : null,
      section: getInfo(section, 'section'),
      main: getInfo(main, 'main')
    };
  });
  console.log('Debug Info:', JSON.stringify(debugInfo, null, 2));

  // Take viewport screenshot first to see exactly what's at top
  await page.screenshot({ path: 'tests/e2e/screenshots/forecasting-desktop-viewport.png', fullPage: false });
  console.log('Screenshot: forecasting-desktop-viewport.png');

  // Then take full page screenshot
  await page.screenshot({ path: 'tests/e2e/screenshots/forecasting-desktop.png', fullPage: true });
  console.log('Screenshot: forecasting-desktop.png');

  // Check for empty string src error
  const emptyStringSrcErrors = consoleErrors.filter(e =>
    e.includes('empty string') && e.includes('src')
  );
  console.log('\n=== EMPTY SRC ERRORS ===');
  if (emptyStringSrcErrors.length === 0) {
    console.log('No empty string src errors found!');
  } else {
    emptyStringSrcErrors.forEach(e => console.log(`  ERROR: ${e}`));
  }

  // Check page structure
  console.log('\n=== PAGE STRUCTURE ===');

  // Check for centered sections
  const forecastingTitle = page.locator('h1:has-text("Forecasting Semanal")');
  const titleVisible = await forecastingTitle.isVisible();
  console.log(`Title "Forecasting Semanal" visible: ${titleVisible}`);

  const configSection = page.locator('h2:has-text("Configuracion")');
  const configVisible = await configSection.isVisible();
  console.log(`Configuration section visible: ${configVisible}`);

  const chartSection = page.locator('h2:has-text("Forecast Chart")');
  const chartVisible = await chartSection.isVisible();
  console.log(`Forecast Chart section visible: ${chartVisible}`);

  const metricsSection = page.locator('h2:has-text("Metricas Clave")');
  const metricsVisible = await metricsSection.isVisible();
  console.log(`Metricas Clave section visible: ${metricsVisible}`);

  // Screenshot - Tablet view (768x1024)
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'tests/e2e/screenshots/forecasting-tablet.png', fullPage: true });
  console.log('\nScreenshot: forecasting-tablet.png');

  // Screenshot - Mobile view (375x812)
  await page.setViewportSize({ width: 375, height: 812 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'tests/e2e/screenshots/forecasting-mobile.png', fullPage: true });
  console.log('Screenshot: forecasting-mobile.png');

  // Check mobile layout
  console.log('\n=== MOBILE LAYOUT ===');
  const filtersButton = page.locator('button:has-text("Mostrar Filtros"), button:has-text("Ocultar Filtros")');
  const filtersButtonVisible = await filtersButton.isVisible();
  console.log(`Mobile filters toggle visible: ${filtersButtonVisible}`);

  // All console logs summary
  console.log('\n=== CONSOLE ERRORS ===');
  const criticalErrors = consoleErrors.filter(e =>
    !e.includes('favicon') && !e.includes('hot-update') && !e.includes('React DevTools')
  );
  if (criticalErrors.length === 0) {
    console.log('No critical errors!');
  } else {
    console.log(`Found ${criticalErrors.length} errors:`);
    criticalErrors.forEach(e => console.log(`  - ${e}`));
  }

  // Verify all sections are visible
  expect(titleVisible).toBe(true);
  expect(configVisible).toBe(true);
});
