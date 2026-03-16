import { test, expect } from '@playwright/test';

const BASE = 'http://localhost:3000/analysis';
const SCREENSHOT_DIR = 'tests/e2e/screenshots/analysis-audit';

test.describe('Analysis Page — Visual Professional Audit', () => {
  test.setTimeout(120_000);

  test('01 — Full page overview (above the fold)', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000); // let animations settle

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/01-full-page-top.png`,
      fullPage: false,
    });
  });

  test('02 — Full page scroll (entire content)', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/02-full-page-scroll.png`,
      fullPage: true,
    });
  });

  test('03 — Week selector component', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(2000);

    const selector = page.locator('[class*="WeekSelector"], [class*="week-selector"]').first();
    if (await selector.isVisible()) {
      await selector.screenshot({ path: `${SCREENSHOT_DIR}/03-week-selector.png` });
    } else {
      // Try finding by content
      const weekArea = page.locator('text=/SEMANA|W[0-9]/i').first();
      if (await weekArea.isVisible()) {
        const parent = weekArea.locator('..').locator('..');
        await parent.screenshot({ path: `${SCREENSHOT_DIR}/03-week-selector.png` });
      }
    }
  });

  test('04 — Weekly summary header with sentiment', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Look for the summary section — typically the first major card after week selector
    const summaryCards = page.locator('[class*="rounded-xl"], [class*="rounded-2xl"]');
    const count = await summaryCards.count();

    // Screenshot the first 2 cards (week selector + weekly summary)
    for (let i = 0; i < Math.min(count, 3); i++) {
      const card = summaryCards.nth(i);
      if (await card.isVisible()) {
        const box = await card.boundingBox();
        if (box && box.height > 50) {
          await card.screenshot({ path: `${SCREENSHOT_DIR}/04-card-${i}.png` });
        }
      }
    }
  });

  test('05 — Technical analysis & trading scenarios section', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Scroll to find technical analysis section
    await page.evaluate(() => window.scrollTo(0, 600));
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/05-technical-section.png`,
      fullPage: false,
    });
  });

  test('06 — Macro indicators section', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Look for "Indicadores Macro" heading
    const macroHeading = page.locator('text=/Indicadores Macro/i').first();
    if (await macroHeading.isVisible()) {
      await macroHeading.scrollIntoViewIfNeeded();
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/06-macro-indicators.png`,
        fullPage: false,
      });
    } else {
      // Scroll to middle area
      await page.evaluate(() => window.scrollTo(0, 1200));
      await page.waitForTimeout(500);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/06-macro-indicators.png`,
        fullPage: false,
      });
    }
  });

  test('07 — Macro charts grid', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Scroll further down to find charts
    await page.evaluate(() => window.scrollTo(0, 1800));
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/07-macro-charts.png`,
      fullPage: false,
    });
  });

  test('08 — News clusters / bias section', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Scroll to news section
    await page.evaluate(() => window.scrollTo(0, 2400));
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/08-news-clusters.png`,
      fullPage: false,
    });
  });

  test('09 — Daily timeline section', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Look for daily timeline
    const timeline = page.locator('text=/Timeline|Cronolog/i').first();
    if (await timeline.isVisible()) {
      await timeline.scrollIntoViewIfNeeded();
      await page.waitForTimeout(500);
    } else {
      await page.evaluate(() => window.scrollTo(0, 3000));
      await page.waitForTimeout(500);
    }

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/09-daily-timeline.png`,
      fullPage: false,
    });
  });

  test('10 — Bottom of page (events + references)', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Scroll to the very bottom
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(1000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/10-bottom-events-refs.png`,
      fullPage: false,
    });
  });

  test('11 — Switch to older week (W05) and verify content', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Try clicking week navigation arrows or week buttons to go to W05
    // Look for left arrow or previous week button
    const prevButtons = page.locator('button').filter({ hasText: /</});
    const prevCount = await prevButtons.count();

    // Click previous several times to get to an older week
    for (let i = 0; i < 6; i++) {
      const prevBtn = page.locator('button').filter({ hasText: /<|←|chevron/i }).first();
      if (await prevBtn.isVisible()) {
        await prevBtn.click();
        await page.waitForTimeout(1500);
      }
    }

    await page.waitForTimeout(2000);
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/11-older-week.png`,
      fullPage: false,
    });

    // Now take full page of this older week
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/11-older-week-full.png`,
      fullPage: true,
    });
  });

  test('12 — Check for hyperlinks and source references in markdown', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Count hyperlinks within the analysis content
    const links = page.locator('a[href^="http"]');
    const linkCount = await links.count();

    // Count "Fuentes" section if visible
    const fuentesSection = page.locator('text=/Fuentes|Referencias|Sources/i');
    const hasFuentes = await fuentesSection.count();

    // Check for source attribution text
    const sourceRefs = page.locator('text=/FRED|BanRep|Investing\\.com|Fedesarrollo|DANE/i');
    const sourceCount = await sourceRefs.count();

    // Screenshot the full page with annotations
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/12-hyperlinks-check.png`,
      fullPage: true,
    });

    // Log results
    console.log(`=== HYPERLINK & SOURCE AUDIT ===`);
    console.log(`Total external links: ${linkCount}`);
    console.log(`Has "Fuentes" section: ${hasFuentes > 0 ? 'YES' : 'NO'}`);
    console.log(`Source attributions found: ${sourceCount}`);

    // Collect all link texts and hrefs
    for (let i = 0; i < Math.min(linkCount, 20); i++) {
      const link = links.nth(i);
      const text = await link.textContent();
      const href = await link.getAttribute('href');
      console.log(`  Link ${i + 1}: "${text?.trim()}" -> ${href}`);
    }
  });

  test('13 — Sentiment values are non-zero', async ({ page }) => {
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    // Get all text content and check for "0.000" sentiment
    const bodyText = await page.textContent('body');

    const hasZeroSentiment = bodyText?.includes('0.000');
    const sentimentMatches = bodyText?.match(/sentimiento[^:]*:\s*([^\n,]+)/gi);

    console.log(`=== SENTIMENT AUDIT ===`);
    console.log(`Contains "0.000": ${hasZeroSentiment ? 'YES (BAD)' : 'NO (GOOD)'}`);
    if (sentimentMatches) {
      sentimentMatches.forEach(m => console.log(`  ${m.trim()}`));
    }

    // Check sentiment badges
    const badges = page.locator('[class*="rounded-full"]');
    const badgeCount = await badges.count();
    console.log(`Sentiment badges found: ${badgeCount}`);
  });

  test('14 — Mobile responsive check (375px width)', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/14-mobile-top.png`,
      fullPage: false,
    });

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/14-mobile-full.png`,
      fullPage: true,
    });
  });

  test('15 — Wide desktop check (1920px)', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto(BASE, { waitUntil: 'networkidle' });
    await page.waitForTimeout(3000);

    await page.screenshot({
      path: `${SCREENSHOT_DIR}/15-desktop-wide.png`,
      fullPage: false,
    });
  });
});
