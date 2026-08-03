import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';

const repoRoot = path.resolve(__dirname, '../../..');
const dashboardRoot = path.join(repoRoot, 'usdcop-trading-dashboard');
const evidenceRoot = path.join(__dirname, '../evidence/playwright');

export default defineConfig({
  testDir: '.',
  testMatch: /visual-evidence\.spec\.ts/,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: 0,
  workers: 1,
  outputDir: path.join(evidenceRoot, 'artifacts'),
  reporter: [
    ['html', { outputFolder: path.join(evidenceRoot, 'html'), open: 'never' }],
    ['json', { outputFile: path.join(evidenceRoot, 'results.json') }],
    ['list'],
  ],
  use: {
    baseURL: 'http://127.0.0.1:5000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    reducedMotion: 'reduce',
  },
  projects: [
    { name: 'desktop-chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 900 } } },
    { name: 'mobile-chromium', use: { ...devices['Pixel 5'] } },
  ],
  webServer: {
    command: 'npm run dev',
    cwd: dashboardRoot,
    url: 'http://127.0.0.1:5000/login',
    reuseExistingServer: true,
    timeout: 180_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});

