import { chromium } from "playwright";

(async () => {
  const browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
  const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 } });
  const page = await ctx.newPage();
  await page.goto("http://localhost:5000/login", { waitUntil: "networkidle" });
  await page.waitForTimeout(1500);
  await page.locator('input[type="email"], input[type="text"]').first().fill("admin");
  await page.locator('input[type="password"]').first().fill("admin123");
  await page.locator('input[type="password"]').first().press("Enter");
  await page.waitForURL(/\/hub|\/dashboard/, { timeout: 5000 });
  await page.goto("http://localhost:5000/forecasting", { waitUntil: "networkidle" });
  await page.waitForTimeout(4000);
  const selects = await page.locator("select").all();
  console.log(`found ${selects.length} <select> elements`);
  for (let i = 0; i < selects.length; i++) {
    const options = await selects[i].locator("option").all();
    const optStrs: string[] = [];
    for (const o of options) {
      const text = await o.textContent();
      const val = await o.getAttribute("value");
      optStrs.push(`${val}:"${text?.trim()}"`);
    }
    const value = await selects[i].inputValue();
    console.log(`  [${i}] current="${value}" options=[${optStrs.join(", ")}]`);
  }
  await browser.close();
})();
