import { expect, test } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const publicRoutes = ['/login', '/register', '/pricing'];

for (const route of publicRoutes) {
  test(`${route} renders without critical accessibility or console failures`, async ({ page }, testInfo) => {
    const consoleErrors: string[] = [];
    page.on('console', (message) => {
      if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', (error) => consoleErrors.push(error.message));

    const response = await page.goto(route, { waitUntil: 'networkidle' });
    expect(response, `No navigation response for ${route}`).not.toBeNull();
    expect(response!.status(), `${route} returned ${response!.status()}`).toBeLessThan(400);
    await expect(page.locator('body')).toBeVisible();
    await expect(page.locator('h1').first()).toBeVisible();

    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
    expect(overflow, `${route} has horizontal overflow`).toBe(false);

    const axe = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();
    const serious = axe.violations.filter((v) => v.impact === 'critical' || v.impact === 'serious');
    expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);

    const screenshot = testInfo.outputPath(`${route.slice(1)}.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    await testInfo.attach(`${route}-screenshot`, { path: screenshot, contentType: 'image/png' });
    await testInfo.attach(`${route}-axe`, {
      body: Buffer.from(JSON.stringify(axe, null, 2)),
      contentType: 'application/json',
    });

    expect(consoleErrors, consoleErrors.join('\n')).toEqual([]);
  });
}

test('keyboard focus is visible and reaches the primary login controls', async ({ page }) => {
  await page.goto('/login', { waitUntil: 'networkidle' });
  await page.keyboard.press('Tab');
  const focused = page.locator(':focus');
  await expect(focused).toBeVisible();
  const outline = await focused.evaluate((el) => {
    const style = getComputedStyle(el);
    return { outlineStyle: style.outlineStyle, outlineWidth: style.outlineWidth, boxShadow: style.boxShadow };
  });
  expect(
    outline.outlineStyle !== 'none' || outline.outlineWidth !== '0px' || outline.boxShadow !== 'none',
    `Focused element has no visible focus treatment: ${JSON.stringify(outline)}`,
  ).toBe(true);
});

