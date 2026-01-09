import { test, expect } from '@playwright/test';

/**
 * Comprehensive Page Analysis Test
 * Analyzes Dashboard and Forecasting pages for:
 * - Header behavior and overlap issues
 * - Console logs and errors
 * - Mobile responsiveness
 * - Layout organization
 */
test('Analyze Dashboard and Forecasting Pages', async ({ page }) => {
  const consoleLogs: string[] = [];
  const networkErrors: string[] = [];

  // Capture all console messages
  page.on('console', (msg) => {
    const type = msg.type().toUpperCase();
    const text = msg.text();
    consoleLogs.push(`[${type}] ${text}`);
  });

  page.on('pageerror', (error) => {
    consoleLogs.push(`[PAGE_ERROR] ${error.message}`);
  });

  page.on('requestfailed', (request) => {
    networkErrors.push(`[FAILED] ${request.method()} ${request.url()} - ${request.failure()?.errorText}`);
  });

  // Test different viewport sizes
  const viewports = [
    { name: 'desktop', width: 1920, height: 1080 },
    { name: 'tablet', width: 768, height: 1024 },
    { name: 'mobile', width: 375, height: 812 },
  ];

  // === LOGIN FIRST ===
  console.log('\n=== LOGGING IN ===');
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto('http://localhost:5000/login', { waitUntil: 'networkidle' });

  await page.locator('input[autocomplete="username"]').fill('admin');
  await page.locator('input[type="password"]').fill('admin123');
  await page.locator('button[type="submit"]').click();

  await page.waitForURL('**/hub', { timeout: 10000 });
  console.log('Logged in successfully');

  // === ANALYZE DASHBOARD ===
  console.log('\n' + '='.repeat(80));
  console.log('ANALYZING DASHBOARD PAGE');
  console.log('='.repeat(80));

  // Navigate to Dashboard
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);

  for (const viewport of viewports) {
    console.log(`\n--- Dashboard: ${viewport.name} (${viewport.width}x${viewport.height}) ---`);
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.waitForTimeout(1000);

    // Take screenshot
    await page.screenshot({
      path: `tests/e2e/screenshots/dashboard-${viewport.name}.png`,
      fullPage: true
    });
    console.log(`Screenshot: dashboard-${viewport.name}.png`);

    // Check header position and behavior
    const navbar = await page.locator('nav, [class*="navbar"], [class*="Navbar"]').first();
    if (await navbar.count() > 0) {
      const navbarBox = await navbar.boundingBox();
      console.log(`Navbar bounding box: ${JSON.stringify(navbarBox)}`);

      // Check if navbar has fixed/sticky positioning
      const navbarStyles = await navbar.evaluate((el) => {
        const styles = window.getComputedStyle(el);
        return {
          position: styles.position,
          top: styles.top,
          zIndex: styles.zIndex,
          backgroundColor: styles.backgroundColor,
          backdropFilter: styles.backdropFilter
        };
      });
      console.log(`Navbar styles: ${JSON.stringify(navbarStyles)}`);
    }

    // Check main content position relative to header
    const mainContent = await page.locator('main, [class*="main"], [class*="content"]').first();
    if (await mainContent.count() > 0) {
      const contentBox = await mainContent.boundingBox();
      console.log(`Main content starts at Y: ${contentBox?.y}`);
    }

    // Check for overlapping elements
    const firstSection = await page.locator('section, [class*="section"], [class*="Overview"]').first();
    if (await firstSection.count() > 0) {
      const sectionBox = await firstSection.boundingBox();
      console.log(`First section Y position: ${sectionBox?.y}`);
    }
  }

  // === ANALYZE FORECASTING ===
  console.log('\n' + '='.repeat(80));
  console.log('ANALYZING FORECASTING PAGE');
  console.log('='.repeat(80));

  // Navigate to Forecasting
  await page.goto('http://localhost:5000/forecasting', { waitUntil: 'networkidle' });
  await page.waitForTimeout(3000);

  for (const viewport of viewports) {
    console.log(`\n--- Forecasting: ${viewport.name} (${viewport.width}x${viewport.height}) ---`);
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.waitForTimeout(1000);

    // Take screenshot
    await page.screenshot({
      path: `tests/e2e/screenshots/forecasting-${viewport.name}.png`,
      fullPage: true
    });
    console.log(`Screenshot: forecasting-${viewport.name}.png`);

    // Check navbar
    const navbar = await page.locator('nav, [class*="navbar"], [class*="Navbar"]').first();
    if (await navbar.count() > 0) {
      const navbarBox = await navbar.boundingBox();
      console.log(`Navbar bounding box: ${JSON.stringify(navbarBox)}`);
    }

    // Check main content
    const mainContent = await page.locator('main, [class*="main"], [class*="content"]').first();
    if (await mainContent.count() > 0) {
      const contentBox = await mainContent.boundingBox();
      console.log(`Main content starts at Y: ${contentBox?.y}`);
    }
  }

  // === HEADER OVERLAP TEST ===
  console.log('\n' + '='.repeat(80));
  console.log('HEADER OVERLAP ANALYSIS');
  console.log('='.repeat(80));

  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto('http://localhost:5000/dashboard', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000);

  // Scroll down to see if header stays fixed
  await page.evaluate(() => window.scrollTo(0, 300));
  await page.waitForTimeout(500);

  await page.screenshot({
    path: 'tests/e2e/screenshots/dashboard-scrolled.png',
    fullPage: false
  });
  console.log('Screenshot: dashboard-scrolled.png (after scroll)');

  // Check header transparency
  const header = await page.locator('nav, header, [class*="Navbar"]').first();
  if (await header.count() > 0) {
    const headerStyles = await header.evaluate((el) => {
      const styles = window.getComputedStyle(el);
      return {
        backgroundColor: styles.backgroundColor,
        backdropFilter: styles.backdropFilter,
        opacity: styles.opacity,
        position: styles.position,
        zIndex: styles.zIndex
      };
    });
    console.log(`Header transparency analysis: ${JSON.stringify(headerStyles)}`);
  }

  // === PRINT CONSOLE LOGS ===
  console.log('\n' + '='.repeat(80));
  console.log('CONSOLE LOGS SUMMARY');
  console.log('='.repeat(80));

  const errors = consoleLogs.filter(l =>
    l.includes('[ERROR]') || l.includes('[PAGE_ERROR]')
  );
  const warnings = consoleLogs.filter(l => l.includes('[WARNING]'));

  console.log(`Total logs: ${consoleLogs.length}`);
  console.log(`Errors: ${errors.length}`);
  console.log(`Warnings: ${warnings.length}`);
  console.log(`Network Errors: ${networkErrors.length}`);

  if (errors.length > 0) {
    console.log('\n--- ERRORS ---');
    errors.forEach(e => console.log(e));
  }

  if (warnings.length > 0) {
    console.log('\n--- WARNINGS ---');
    warnings.slice(0, 10).forEach(w => console.log(w));
  }

  if (networkErrors.length > 0) {
    console.log('\n--- NETWORK ERRORS ---');
    networkErrors.slice(0, 10).forEach(e => console.log(e));
  }

  // Print all logs for detailed analysis
  console.log('\n--- ALL LOGS (last 50) ---');
  consoleLogs.slice(-50).forEach(log => console.log(log));
});
