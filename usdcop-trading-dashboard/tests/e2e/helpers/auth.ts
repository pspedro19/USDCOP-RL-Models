/**
 * Shared E2E test helpers — authentication and console error capture.
 * Extracted from duplicated patterns across 28+ test files.
 */
import { Page } from '@playwright/test';

/**
 * Authenticate the user by setting localStorage/sessionStorage tokens.
 * Must be called after page.goto('/login') or any navigation.
 */
export async function authenticateUser(page: Page): Promise<void> {
  await page.goto('/login');
  await page.evaluate(() => {
    localStorage.setItem('isAuthenticated', 'true');
    localStorage.setItem('username', 'admin');
    sessionStorage.setItem('isAuthenticated', 'true');
    sessionStorage.setItem('username', 'admin');
  });
}

/**
 * Attach console error and page error listeners.
 * Returns the mutable errors array that fills as errors occur.
 */
export function setupConsoleErrorCapture(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text());
  });
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

/** Filter out known benign console errors (hydration warnings, favicon, etc.) */
export function filterBenignErrors(errors: string[]): string[] {
  const benign = [
    'favicon',
    'hydrat',
    'ERR_BLOCKED_BY_CLIENT',
    'net::ERR_FAILED',
    'net::ERR_CONNECTION_REFUSED',
    'Download the React DevTools',
    'WebSocket connection',
    'Failed to load resource',
    'the server responded with a status of 404',
  ];
  return errors.filter(
    (e) => !benign.some((b) => e.toLowerCase().includes(b.toLowerCase()))
  );
}
