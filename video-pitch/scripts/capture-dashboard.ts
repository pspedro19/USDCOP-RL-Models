/**
 * Playwright capture script v2 — QA-reviewed flows.
 *
 * Captures 9 clips + 1 extra Airflow DAG detail for pitch video.
 * Fixes from v1:
 *  - S01 login: actually submits form and waits for /hub redirect
 *  - S03-replay: clicks correct "Replay" button (not "Play") and waits for animation
 *  - S05 production: waits for data to finish loading before recording
 *  - I04 Airflow: logs in (admin/admin) then navigates to DAG tree + opens a specific DAG
 *
 * Usage:
 *   npx tsx scripts/capture-dashboard.ts            (all)
 *   npx tsx scripts/capture-dashboard.ts S03        (single)
 *
 * Pre-reqs: localhost:5000 + localhost:8080 UP, creds admin/admin123.
 */

import { chromium, Browser, BrowserContext, Page } from "playwright";
import * as path from "path";
import * as fs from "fs";

const DASHBOARD = "http://localhost:5000";
const AIRFLOW = "http://localhost:8080";
const CREDS_DASH = { email: "admin", password: "admin123" };
const CREDS_AIRFLOW = { username: "admin", password: "admin123" };
const OUTPUT_DIR = path.resolve(__dirname, "..", "public", "captures");
const STORAGE_STATE = path.resolve(__dirname, "..", ".playwright-state.json");
const VIEWPORT = { width: 1920, height: 1080 };

interface ClipSpec {
  id: string;
  duration: number;
  fn: (page: Page) => Promise<void>;
  needsDashAuth?: boolean;
  freshContext?: boolean; // if true do NOT reuse storage state
}

const CLIPS: ClipSpec[] = [
  // =========================================================================
  // S01 — Login flow (full): type → submit → redirect to /hub
  // =========================================================================
  {
    id: "S01-login",
    duration: 12000,
    freshContext: true,
    fn: async (page) => {
      await page.goto(`${DASHBOARD}/login`, { waitUntil: "networkidle" });
      await page.waitForTimeout(2000);

      // Simple approach from v1 that worked: use type=email|text first match
      const emailField = page
        .locator('input[type="email"], input[type="text"]')
        .first();
      await emailField.waitFor({ state: "visible", timeout: 5000 });
      await emailField.click();
      await page.waitForTimeout(300);
      await emailField.type(CREDS_DASH.email, { delay: 60 });
      await page.waitForTimeout(500);

      const passField = page.locator('input[type="password"]').first();
      await passField.click();
      await page.waitForTimeout(200);
      await passField.type(CREDS_DASH.password, { delay: 60 });
      await page.waitForTimeout(700);

      // Submit via Enter (works for most React forms)
      await passField.press("Enter");
      try {
        await page.waitForURL(/\/hub|\/dashboard/, { timeout: 4500 });
      } catch {
        // Fallback: click the button
        try {
          const submitBtn = page
            .locator(
              'button[type="submit"], button:has-text("Iniciar"), button:has-text("Ingresa")'
            )
            .first();
          await submitBtn.click({ timeout: 2500 });
          await page.waitForURL(/\/hub|\/dashboard/, { timeout: 4000 });
        } catch {
          /* show current state */
        }
      }
      // Settle on destination
      await page.waitForTimeout(2500);
    },
  },

  // =========================================================================
  // S02 — Hub (6 modules)
  // =========================================================================
  {
    id: "S02-hub",
    duration: 6500,
    fn: async (page) => {
      await page.goto(`${DASHBOARD}/hub`, { waitUntil: "networkidle" });
      await page.waitForTimeout(1500);
      // Subtle hover over multiple cards for visual motion
      for (const title of [
        "Trading Dashboard",
        "Monitor",
        "Forecasting",
        "Analisis",
        "SignalBridge",
      ]) {
        try {
          await page
            .locator(`text=${title}`)
            .first()
            .hover({ timeout: 600 });
          await page.waitForTimeout(400);
        } catch {
          /* skip */
        }
      }
      await page.waitForTimeout(1200);
    },
  },

  // =========================================================================
  // S03 — Dashboard scroll (KPIs → equity → trades → gates)
  // =========================================================================
  {
    id: "S03-dashboard-scroll",
    duration: 12000,
    fn: async (page) => {
      await page.goto(`${DASHBOARD}/dashboard`, { waitUntil: "networkidle" });
      // Wait for actual data to load (not just "Cargando datos...")
      await page.waitForTimeout(4500);
      // Slow progressive scroll
      const total = 1400;
      const steps = 20;
      for (let i = 0; i <= steps; i++) {
        await page.evaluate(
          (y) => window.scrollTo({ top: y, behavior: "auto" }),
          (total * i) / steps
        );
        await page.waitForTimeout(350);
      }
    },
  },

  // =========================================================================
  // S03-replay — BACKTEST REPLAY (money shot) v2
  //   Choreographed scroll: controls → up to chart → down → back up
  //   Total ~22s so the entire replay animation is visible with motion
  // =========================================================================
  {
    id: "S03-replay",
    duration: 22000,
    fn: async (page) => {
      await page.goto(`${DASHBOARD}/dashboard`, { waitUntil: "networkidle" });
      // Wait for full data load (chart + metrics + candles)
      await page.waitForTimeout(5500);

      // Step 1: scroll to replay controls area
      await page.evaluate(() =>
        window.scrollTo({ top: 850, behavior: "auto" })
      );
      await page.waitForTimeout(1200);

      // Step 2: click "Replay" button to start animation
      try {
        const replayBtn = page
          .locator('button:has-text("Replay"), button:has-text("Reproducir")')
          .first();
        await replayBtn.click({ timeout: 3000 });
        console.log("  ↪ replay button clicked");
      } catch (e) {
        console.warn("  ⚠ replay button not found");
      }

      // Step 3: let replay run 2s with controls visible (see early trades appearing)
      await page.waitForTimeout(2500);

      // Step 4: scroll UP smoothly to see the chart at full view (candles + progressive reveal)
      for (const y of [700, 550, 400, 250]) {
        await page.evaluate(
          (top) => window.scrollTo({ top, behavior: "smooth" }),
          y
        );
        await page.waitForTimeout(650);
      }
      await page.waitForTimeout(1800); // dwell on chart view while replay animates

      // Step 5: scroll back DOWN to see the equity curve updating with replay
      for (const y of [400, 650, 900, 1150]) {
        await page.evaluate(
          (top) => window.scrollTo({ top, behavior: "smooth" }),
          y
        );
        await page.waitForTimeout(550);
      }
      await page.waitForTimeout(1500); // dwell on equity curve as replay progresses

      // Step 6: scroll UP again to see chart one more time as replay reaches end
      for (const y of [900, 600, 350]) {
        await page.evaluate(
          (top) => window.scrollTo({ top, behavior: "smooth" }),
          y
        );
        await page.waitForTimeout(500);
      }
      await page.waitForTimeout(1500); // final dwell — replay near complete

      // Step 7: scroll to KPIs at top for final view with updated metrics
      await page.evaluate(() =>
        window.scrollTo({ top: 150, behavior: "smooth" })
      );
      await page.waitForTimeout(2500);
    },
  },

  // =========================================================================
  // S05 — Production live (2026 YTD with actual data, not loading)
  // =========================================================================
  {
    id: "S05-production-live",
    duration: 11000,
    fn: async (page) => {
      await page.goto(`${DASHBOARD}/production`, { waitUntil: "networkidle" });
      // Key fix: wait for data to load (previously started recording while loading)
      await page.waitForTimeout(5500);
      // Scroll through production view
      await page.evaluate(() =>
        window.scrollTo({ top: 200, behavior: "auto" })
      );
      await page.waitForTimeout(2500);
      await page.evaluate(() =>
        window.scrollTo({ top: 550, behavior: "auto" })
      );
      await page.waitForTimeout(2500);
    },
  },

  // =========================================================================
  // S06 — Analysis: multi-week navigation + full scroll coverage
  //   Pattern per week: show week → scroll DOWN seeing everything →
  //   scroll UP → click previous week → repeat for 3 weeks total
  // =========================================================================
  {
    id: "S06-analysis-chat",
    duration: 22000,
    fn: async (page) => {
      await page.goto(`${DASHBOARD}/analysis`, { waitUntil: "networkidle" });
      await page.waitForTimeout(4500);

      // Helper: full scroll cycle showing all content of current week
      const scrollCycle = async () => {
        for (const y of [250, 550, 900, 1250, 1500]) {
          await page.evaluate(
            (top) => window.scrollTo({ top, behavior: "smooth" }),
            y
          );
          await page.waitForTimeout(500);
        }
        // scroll back up
        for (const y of [1000, 500, 0]) {
          await page.evaluate(
            (top) => window.scrollTo({ top, behavior: "smooth" }),
            y
          );
          await page.waitForTimeout(400);
        }
      };

      // WEEK 1 (current week on landing) — full scroll cycle
      await scrollCycle();

      // Click previous week arrow
      try {
        const prevBtn = page
          .locator(
            'button[aria-label*="previ" i], button[aria-label*="anterior" i], button:has-text("‹"), button:has-text("<")'
          )
          .first();
        await prevBtn.click({ timeout: 1500 });
        await page.waitForTimeout(2000);
      } catch {
        /* fallback: use keyboard */
        await page.keyboard.press("ArrowLeft");
        await page.waitForTimeout(1500);
      }

      // WEEK 2 — partial scroll
      for (const y of [400, 900, 0]) {
        await page.evaluate(
          (top) => window.scrollTo({ top, behavior: "smooth" }),
          y
        );
        await page.waitForTimeout(700);
      }

      // Click previous week again
      try {
        const prevBtn = page
          .locator(
            'button[aria-label*="previ" i], button[aria-label*="anterior" i], button:has-text("‹"), button:has-text("<")'
          )
          .first();
        await prevBtn.click({ timeout: 1500 });
        await page.waitForTimeout(1800);
      } catch {
        await page.keyboard.press("ArrowLeft");
        await page.waitForTimeout(1500);
      }

      // WEEK 3 — dwell briefly
      await page.waitForTimeout(1500);
      await page.evaluate(() =>
        window.scrollTo({ top: 400, behavior: "smooth" })
      );
      await page.waitForTimeout(1500);
    },
  },

  // =========================================================================
  // S07 — Forward Forecast + Backtest multi-week navigation with full scroll
  //   Shows: Forward view W16 → W15 → switch to Backtest → cycle models
  //   Each step: complete scroll down + up to show ALL content
  //   NOTE: Forward PNGs currently 404 from Next.js cache — but UI still
  //   shows the config, model list, metrics correctly; Backtest view has
  //   working charts.
  // =========================================================================
  {
    id: "S07-forecasting-zoo",
    duration: 24000,
    fn: async (page) => {
      await page.goto(`${DASHBOARD}/forecasting`, { waitUntil: "networkidle" });
      await page.waitForTimeout(5000);

      const scrollCycle = async (ys: number[], dwellMs = 600) => {
        for (const y of ys) {
          await page.evaluate(
            (top) => window.scrollTo({ top, behavior: "smooth" }),
            y
          );
          await page.waitForTimeout(dwellMs);
        }
      };

      // PHASE 1: Forward Forecast W16 (current) — scroll down + up
      await scrollCycle([100, 350, 600, 900], 550);
      await page.waitForTimeout(800);
      await scrollCycle([500, 200, 0], 450);

      // PHASE 2: switch to Backtest Analysis (PNGs load reliably here)
      try {
        const viewSelect = page.locator("select").first();
        await viewSelect.selectOption("backtest");
        await page.waitForTimeout(2500);
      } catch {
        /* stay */
      }

      // PHASE 3: Backtest view — scroll down showing Ranking + chart
      await scrollCycle([100, 350, 600, 900, 1150], 600);
      await page.waitForTimeout(800);

      // PHASE 4: cycle through models to show different backtest charts
      try {
        const modelSelect = page.locator("select").nth(2);
        await modelSelect.selectOption("Ridge");
        await page.waitForTimeout(1500);
        await scrollCycle([400, 650], 500);
        await modelSelect.selectOption("XGBoost");
        await page.waitForTimeout(1500);
        await scrollCycle([400, 650], 500);
      } catch {
        /* ok */
      }

      // PHASE 5: scroll back up for final view
      await scrollCycle([400, 150], 500);
      await page.waitForTimeout(1200);
    },
  },

  // =========================================================================
  // S08 — SignalBridge (dashboard → exchanges → settings → executions)
  //   Emphasizes: connect to ANY exchange API (MEXC, Binance, custom)
  //   Full scroll down+up on each view.
  // =========================================================================
  {
    id: "S08-signalbridge",
    duration: 18000,
    fn: async (page) => {
      // Step 1: auth + go to execution dashboard
      await page.goto(`${DASHBOARD}/execution/dashboard`, {
        waitUntil: "networkidle",
      });
      await page.waitForTimeout(3000);
      if (page.url().includes("/login")) {
        try {
          const emailField = page
            .locator('input[type="email"], input[type="text"]')
            .first();
          await emailField.click();
          await emailField.type(CREDS_DASH.email, { delay: 55 });
          await page.waitForTimeout(300);
          const passField = page.locator('input[type="password"]').first();
          await passField.click();
          await passField.type(CREDS_DASH.password, { delay: 55 });
          await page.waitForTimeout(400);
          await passField.press("Enter");
          await page.waitForURL(/\/execution|\/hub|\/dashboard/, {
            timeout: 5000,
          });
          await page.goto(`${DASHBOARD}/execution/dashboard`, {
            waitUntil: "networkidle",
          });
          await page.waitForTimeout(2500);
        } catch {
          /* continue */
        }
      }

      const scrollCycle = async (ys: number[], dwellMs = 550) => {
        for (const y of ys) {
          await page.evaluate(
            (top) => window.scrollTo({ top, behavior: "smooth" }),
            y
          );
          await page.waitForTimeout(dwellMs);
        }
      };

      // Step 2: scroll dashboard view down + up
      await scrollCycle([150, 400, 650], 650);
      await scrollCycle([300, 0], 500);

      // Step 3: navigate to Exchanges (shows MEXC + Binance + AES-256 encryption)
      try {
        const exchangesLink = page
          .locator(
            'a[href*="exchanges"], button:has-text("Exchanges"), a:has-text("Exchanges")'
          )
          .first();
        await exchangesLink.click({ timeout: 2000 });
        await page.waitForTimeout(2500);
      } catch {
        await page.goto(`${DASHBOARD}/execution/exchanges`, {
          waitUntil: "networkidle",
        });
        await page.waitForTimeout(2500);
      }

      // Step 4: scroll exchanges page — show security notice + available exchanges
      await scrollCycle([0, 200, 400, 200, 0], 700);

      // Step 5: navigate to Settings or Executions for configuration visibility
      try {
        const settingsLink = page
          .locator('a[href*="settings"], button:has-text("Settings"), a:has-text("Settings")')
          .first();
        await settingsLink.click({ timeout: 1500 });
        await page.waitForTimeout(2500);
        await scrollCycle([0, 350, 0], 650);
      } catch {
        /* skip */
      }
    },
  },

  // =========================================================================
  // I04-tour — Airflow MEGA TOUR (~4min · 240s):
  //   Login → DAG list scroll → for each of 10 DAGs:
  //     click DAG → wait → Code tab → scroll down → scroll up → back to home
  //   Each DAG segment ~22s. 10 DAGs cover L0→L8 + watchdog.
  // =========================================================================
  {
    id: "I04-airflow-code-tour",
    duration: 250000,
    freshContext: true,
    fn: async (page) => {
      // --- Helper: direct URL navigation to DAG code view + scroll ---
      //   Using /dags/{id}/code is more reliable than locator+click
      const codeTourForDag = async (dagId: string, label: string) => {
        console.log(`  → [${label}] ${dagId}`);
        try {
          // Direct URL to code view
          await page.goto(`${AIRFLOW}/dags/${dagId}/code`, {
            waitUntil: "domcontentloaded",
            timeout: 10000,
          });
          // Wait for code to render (Monaco/CodeMirror)
          await page.waitForTimeout(2800);

          // Scroll down slowly through code
          for (const y of [200, 500, 900, 1400, 1900, 2400]) {
            await page.evaluate(
              (top) => window.scrollTo({ top, behavior: "smooth" }),
              y
            );
            await page.waitForTimeout(500);
          }
          // Brief dwell at bottom
          await page.waitForTimeout(800);
          // Scroll back up
          for (const y of [1500, 800, 0]) {
            await page.evaluate(
              (top) => window.scrollTo({ top, behavior: "smooth" }),
              y
            );
            await page.waitForTimeout(400);
          }
          await page.waitForTimeout(600);
          console.log(`    ✓ done ${dagId}`);
        } catch (e) {
          console.warn(`    ⚠ failed: ${dagId} — continuing`);
          await page.waitForTimeout(1500);
        }
      };

      // =====================================================================
      // STEP 1 (0-3s): Airflow login
      await page.goto(`${AIRFLOW}/login`, { waitUntil: "networkidle" });
      await page.waitForTimeout(1500);
      try {
        const userField = page
          .locator('input[name="username"], input#username, input[type="text"]')
          .first();
        await userField.waitFor({ state: "visible", timeout: 3000 });
        await userField.click({ force: true });
        await userField.pressSequentially(CREDS_AIRFLOW.username, { delay: 55 });
        await page.waitForTimeout(300);
        const passField = page.locator('input[type="password"]').first();
        await passField.click({ force: true });
        await passField.pressSequentially(CREDS_AIRFLOW.password, { delay: 55 });
        await page.waitForTimeout(400);
        await passField.press("Enter");
        await page.waitForURL(/\/home|\/dags|\/$/, { timeout: 6000 });
        console.log("  ↪ airflow login successful");
      } catch {
        /* fallback */
      }
      await page.waitForTimeout(1800);
      if (page.url().includes("/login")) {
        await page.goto(`${AIRFLOW}/home`, { waitUntil: "domcontentloaded" });
        await page.waitForTimeout(2000);
      }

      // STEP 2 (3-12s): DAG list overview scroll
      await page.waitForTimeout(1200);
      for (const y of [150, 400, 700, 1000, 1300]) {
        await page.evaluate(
          (top) => window.scrollTo({ top, behavior: "smooth" }),
          y
        );
        await page.waitForTimeout(500);
      }
      await page.evaluate(() =>
        window.scrollTo({ top: 0, behavior: "smooth" })
      );
      await page.waitForTimeout(1000);

      // STEP 3: TOUR THROUGH 10 DAGs — each ~22s
      //   L0 Ingesta (2 DAGs)
      await codeTourForDag("core_l0_01_ohlcv_backfill", "L0·1");
      await codeTourForDag("core_l0_03_macro_backfill", "L0·2");
      //   L3/L5 Modelado + régimen (3 DAGs)
      await codeTourForDag("forecast_h5_l3_weekly_training", "L3·MVP");
      await codeTourForDag("forecast_h5_l5_weekly_signal", "L5·Signal");
      await codeTourForDag("forecast_h5_l5_vol_targeting", "L5·RegimeGate★");
      //   L7 Ejecución (1 DAG)
      await codeTourForDag("forecast_h5_l7_multiday_executor", "L7·Exec");
      //   L6 Monitoreo (1 DAG)
      await codeTourForDag("forecast_h5_l6_weekly_monitor", "L6·Monitor");
      //   L8 Intelligence (2 DAGs)
      await codeTourForDag("news_daily_pipeline", "L8·News");
      await codeTourForDag("analysis_l8_daily_generation", "L8·Analysis");
      //   Watchdog (1 DAG)
      await codeTourForDag("core_watchdog", "Watchdog");

      // Final dwell
      await page.waitForTimeout(3000);
    },
  },
];

async function ensureDashLogin(browser: Browser): Promise<void> {
  if (fs.existsSync(STORAGE_STATE)) {
    console.log("↪ dashboard storage state exists, skipping pre-login");
    return;
  }
  console.log("→ pre-logging in to dashboard to save storage state");
  const ctx = await browser.newContext({ viewport: VIEWPORT });
  const page = await ctx.newPage();
  try {
    await page.goto(`${DASHBOARD}/login`, { waitUntil: "domcontentloaded", timeout: 15000 });
    await page.waitForTimeout(2000);
    const emailField = page
      .locator('input[type="email"], input[type="text"]')
      .first();
    await emailField.waitFor({ state: "visible", timeout: 5000 });
    await emailField.click();
    await emailField.type(CREDS_DASH.email, { delay: 60 });
    await page.waitForTimeout(300);
    const passField = page.locator('input[type="password"]').first();
    await passField.click();
    await passField.type(CREDS_DASH.password, { delay: 60 });
    await page.waitForTimeout(300);
    await passField.press("Enter");
    await page.waitForURL(/\/hub|\/dashboard/, { timeout: 6000 });
    console.log("  ↪ pre-login successful");
  } catch (e) {
    console.warn("⚠ pre-login failed, proceeding anyway");
  }
  try {
    await ctx.storageState({ path: STORAGE_STATE });
  } catch {
    /* tolerate */
  }
  await ctx.close();
}

async function captureClip(browser: Browser, clip: ClipSpec): Promise<void> {
  const target = path.join(OUTPUT_DIR, clip.id);
  if (fs.existsSync(target)) {
    fs.rmSync(target, { recursive: true, force: true });
  }
  fs.mkdirSync(target, { recursive: true });

  const contextOpts: Parameters<Browser["newContext"]>[0] = {
    viewport: VIEWPORT,
    recordVideo: { dir: target, size: VIEWPORT },
    deviceScaleFactor: 1,
  };
  if (
    !clip.freshContext &&
    fs.existsSync(STORAGE_STATE) &&
    !clip.id.startsWith("I04")
  ) {
    contextOpts.storageState = STORAGE_STATE;
  }

  const ctx = await browser.newContext(contextOpts);
  const page = await ctx.newPage();
  console.log(`→ capturing ${clip.id} (${clip.duration}ms)`);
  const t0 = Date.now();
  try {
    await clip.fn(page);
    const elapsed = Date.now() - t0;
    if (elapsed < clip.duration) {
      await page.waitForTimeout(clip.duration - elapsed);
    }
  } catch (e) {
    console.error(`✗ ${clip.id} failed:`, e);
  }
  await page.close();
  await ctx.close();

  const files = fs
    .readdirSync(target)
    .filter((f) => f.endsWith(".webm"));
  if (files.length > 0) {
    const src = path.join(target, files[0]);
    const dest = path.join(OUTPUT_DIR, `${clip.id}.webm`);
    fs.renameSync(src, dest);
    fs.rmdirSync(target);
    const stat = fs.statSync(dest);
    console.log(
      `✓ ${clip.id}.webm saved (${(stat.size / 1024 / 1024).toFixed(2)}MB)`
    );
  } else {
    console.warn(`⚠ no video generated for ${clip.id}`);
  }
}

async function main() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const only = process.argv[2];
  const clips = only ? CLIPS.filter((c) => c.id.includes(only)) : CLIPS;
  if (clips.length === 0) {
    console.error(`No clip matches "${only}"`);
    CLIPS.forEach((c) => console.error(`  - ${c.id}`));
    process.exit(1);
  }

  // Kill old storage state so we get fresh auth
  if (fs.existsSync(STORAGE_STATE)) {
    fs.unlinkSync(STORAGE_STATE);
  }

  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  try {
    await ensureDashLogin(browser);
    for (const clip of clips) {
      await captureClip(browser, clip);
    }
  } finally {
    await browser.close();
  }
  console.log("\n✅ all captures complete");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
