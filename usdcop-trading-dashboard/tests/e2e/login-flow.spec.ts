import { test, expect } from '@playwright/test';

/**
 * Login Flow E2E Tests
 * Tests the complete authentication flow with admin/admin123
 */

test.describe('Login Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Capture console logs
    page.on('console', msg => {
      console.log(`[BROWSER ${msg.type().toUpperCase()}] ${msg.text()}`);
    });

    // Capture page errors
    page.on('pageerror', error => {
      console.error(`[BROWSER ERROR] ${error.message}`);
    });
  });

  test('should display login page correctly', async ({ page }) => {
    await page.goto('/login');

    // Take screenshot of login page
    await page.screenshot({
      path: 'tests/e2e/screenshots/login-page.png',
      fullPage: true
    });

    // Verify login form elements using more specific selectors
    await expect(page.locator('input[autocomplete="username"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(page.getByRole('button', { name: /iniciar sesión/i })).toBeVisible();

    console.log('[TEST] Login page loaded successfully');
  });

  test('should login with admin/admin123 and redirect to dashboard', async ({ page }) => {
    // Go to login page
    await page.goto('/login');
    console.log('[TEST] Navigated to /login');

    // Take screenshot before login
    await page.screenshot({
      path: 'tests/e2e/screenshots/login-before.png',
      fullPage: true
    });

    // Fill credentials using specific selectors
    const usernameInput = page.locator('input[autocomplete="username"]');
    const passwordInput = page.locator('input[type="password"]');

    await usernameInput.fill('admin');
    console.log('[TEST] Entered username: admin');

    await passwordInput.fill('admin123');
    console.log('[TEST] Entered password: admin123');

    // Take screenshot after filling credentials
    await page.screenshot({
      path: 'tests/e2e/screenshots/login-filled.png',
      fullPage: true
    });

    // Click login button
    const loginButton = page.getByRole('button', { name: /iniciar sesión/i });
    await loginButton.click();
    console.log('[TEST] Clicked login button');

    // Wait for navigation to dashboard
    await page.waitForURL('**/dashboard', { timeout: 10000 });
    console.log('[TEST] Redirected to dashboard');

    // Wait for dashboard to load
    await page.waitForLoadState('networkidle');

    // Take screenshot of dashboard
    await page.screenshot({
      path: 'tests/e2e/screenshots/dashboard-after-login.png',
      fullPage: true
    });

    // Verify we're on the dashboard
    await expect(page).toHaveURL(/.*dashboard/);

    // Verify dashboard elements
    await expect(page.locator('h1')).toContainText('USDCOP');

    console.log('[TEST] Dashboard loaded successfully');
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');

    // Fill invalid credentials
    await page.locator('input[autocomplete="username"]').fill('wronguser');
    await page.locator('input[type="password"]').fill('wrongpass');

    // Click login
    await page.getByRole('button', { name: /iniciar sesión/i }).click();

    // Wait for error message
    await page.waitForTimeout(2000);

    // Take screenshot of error
    await page.screenshot({
      path: 'tests/e2e/screenshots/login-error.png',
      fullPage: true
    });

    // Verify error message is shown
    await expect(page.locator('text=Credenciales inválidas')).toBeVisible();

    console.log('[TEST] Error message displayed correctly');
  });

  test('should maintain session after login', async ({ page }) => {
    // Login first
    await page.goto('/login');
    await page.locator('input[autocomplete="username"]').fill('admin');
    await page.locator('input[type="password"]').fill('admin123');
    await page.getByRole('button', { name: /iniciar sesión/i }).click();
    await page.waitForURL('**/dashboard', { timeout: 10000 });

    // Verify localStorage
    const isAuthenticated = await page.evaluate(() => {
      return localStorage.getItem('isAuthenticated');
    });

    expect(isAuthenticated).toBe('true');
    console.log('[TEST] Session stored in localStorage');

    // Navigate away and back
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');

    // Should still be on dashboard (not redirected to login)
    await expect(page).toHaveURL(/.*dashboard/);
    console.log('[TEST] Session maintained after navigation');

    // Take final screenshot
    await page.screenshot({
      path: 'tests/e2e/screenshots/session-maintained.png',
      fullPage: true
    });
  });
});
