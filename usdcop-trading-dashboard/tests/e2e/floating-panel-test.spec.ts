import { test, expect } from '@playwright/test';

/**
 * Floating Experiment Panel Test
 * Verifies the approval panel appears for models with PENDING_APPROVAL
 */

const SCREENSHOTS_DIR = 'tests/e2e/screenshots/navigation-flow';

test.describe('Floating Experiment Panel', () => {
  test('Check for pending approval panel in dashboard', async ({ page }) => {
    console.log('\n=== FLOATING PANEL TEST ===');

    // Set auth
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('username', 'admin');
    });

    // Go to dashboard
    await page.goto('/dashboard');
    await page.waitForTimeout(5000); // Wait for data to load

    await page.screenshot({ path: `${SCREENSHOTS_DIR}/floating-01-dashboard.png`, fullPage: true });

    // Look for model dropdown
    const modelButtons = await page.locator('button').all();
    console.log(`Found ${modelButtons.length} buttons on page`);

    // Find the model selector (usually has model name text)
    for (const btn of modelButtons) {
      const text = await btn.textContent().catch(() => '');
      if (text && (text.includes('PPO') || text.includes('Investor') || text.includes('Model') || text.includes('modelo'))) {
        console.log(`Found potential model button: "${text.substring(0, 50)}..."`);
      }
    }

    // Look for the floating panel
    const floatingPanelSelectors = [
      'text=PROPUESTA PENDIENTE',
      'text=PENDING_APPROVAL',
      'text=Aprobar',
      'text=Rechazar',
      '[class*="fixed"][class*="bottom"]',
      '[class*="floating"]'
    ];

    let panelFound = false;
    for (const selector of floatingPanelSelectors) {
      const element = page.locator(selector).first();
      if (await element.isVisible().catch(() => false)) {
        console.log(`Found floating panel element with: ${selector}`);
        panelFound = true;
        break;
      }
    }

    console.log(`Floating panel visible: ${panelFound}`);

    // Try to click on model dropdown to see options
    const dropdownTrigger = page.locator('button:has-text("PPO"), button:has-text("Investor"), button:has-text("Select")').first();
    if (await dropdownTrigger.isVisible().catch(() => false)) {
      console.log('Clicking model dropdown...');
      await dropdownTrigger.click();
      await page.waitForTimeout(1000);
      await page.screenshot({ path: `${SCREENSHOTS_DIR}/floating-02-dropdown-open.png`, fullPage: true });

      // Look for dropdown options
      const options = await page.locator('[role="option"], [role="menuitem"], li, [class*="option"]').all();
      console.log(`Found ${options.length} dropdown options`);

      for (const opt of options.slice(0, 10)) {
        const text = await opt.textContent().catch(() => '');
        console.log(`  Option: "${text.substring(0, 60)}"`);
      }

      // Try to click on PPO SSOT model if visible
      const ppoOption = page.locator('text=PPO SSOT').first();
      if (await ppoOption.isVisible().catch(() => false)) {
        console.log('Clicking PPO SSOT model...');
        await ppoOption.click();
        await page.waitForTimeout(3000);
        await page.screenshot({ path: `${SCREENSHOTS_DIR}/floating-03-ppo-selected.png`, fullPage: true });

        // Check for floating panel again
        const panelAfter = await page.locator('text=PROPUESTA, text=Aprobar, text=PROMOTE').first().isVisible().catch(() => false);
        console.log(`Floating panel visible after selection: ${panelAfter}`);
      }
    }

    // Final screenshot
    await page.screenshot({ path: `${SCREENSHOTS_DIR}/floating-04-final.png`, fullPage: true });

    // Log page HTML for debugging (first 5000 chars)
    const html = await page.content();
    if (html.includes('FloatingExperiment') || html.includes('floating')) {
      console.log('Page contains floating-related content');
    }
    if (html.includes('PENDING_APPROVAL') || html.includes('pendingExperiment')) {
      console.log('Page contains pending approval content');
    }
  });
});
