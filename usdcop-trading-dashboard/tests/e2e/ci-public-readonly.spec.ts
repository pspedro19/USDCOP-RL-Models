import { expect, test, type Page, type TestInfo } from '@playwright/test';

type BrowserDiagnostics = {
  consoleErrors: string[];
  pageErrors: string[];
  requestFailures: string[];
  serverErrors: string[];
};

function collectDiagnostics(page: Page): BrowserDiagnostics {
  const diagnostics: BrowserDiagnostics = {
    consoleErrors: [],
    pageErrors: [],
    requestFailures: [],
    serverErrors: [],
  };

  page.on('console', (message) => {
    if (message.type() === 'error') {
      diagnostics.consoleErrors.push(message.text());
    }
  });
  page.on('pageerror', (error) => {
    diagnostics.pageErrors.push(error.message);
  });
  page.on('requestfailed', (request) => {
    diagnostics.requestFailures.push(
      `${request.method()} ${request.url()} :: ${request.failure()?.errorText ?? 'unknown failure'}`,
    );
  });
  page.on('response', (response) => {
    if (response.status() >= 500) {
      diagnostics.serverErrors.push(`${response.status()} ${response.request().method()} ${response.url()}`);
    }
  });

  return diagnostics;
}

async function attachDiagnostics(testInfo: TestInfo, diagnostics: BrowserDiagnostics): Promise<void> {
  await testInfo.attach('browser-diagnostics.json', {
    body: Buffer.from(JSON.stringify(diagnostics, null, 2)),
    contentType: 'application/json',
  });
}

test.describe('CI public/read-only browser contract', () => {
  test('fresh production artifact exposes login but never serves anonymous /replay', async ({
    page,
    request,
  }, testInfo) => {
    const diagnostics = collectDiagnostics(page);

    const login = await page.goto('/login', { waitUntil: 'domcontentloaded' });
    expect(login, 'the production artifact did not answer /login').not.toBeNull();
    expect(login?.status()).toBeLessThan(400);
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await page.screenshot({
      path: testInfo.outputPath('01-login.png'),
      fullPage: true,
    });

    await page.goto('/replay', { waitUntil: 'domcontentloaded' });
    await expect(page).toHaveURL(/\/login(?:[/?#]|$)/);
    await page.screenshot({
      path: testInfo.outputPath('02-anonymous-replay-redirect.png'),
      fullPage: true,
    });

    const registry = await request.get('/api/registry', { maxRedirects: 0 });
    expect(registry.status()).toBe(401);
    expect(await registry.json()).toMatchObject({ error: expect.anything() });

    await attachDiagnostics(testInfo, diagnostics);
    expect(
      diagnostics,
      'browser/runtime diagnostics must remain empty; inspect the attached JSON and server log',
    ).toEqual({
      consoleErrors: [],
      pageErrors: [],
      requestFailures: [],
      serverErrors: [],
    });
  });
});
