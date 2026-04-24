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
  await page.goto("http://localhost:5000/analysis", { waitUntil: "networkidle" });
  await page.waitForTimeout(4000);

  // Find week buttons
  const buttons = await page.locator("button").all();
  console.log(`found ${buttons.length} buttons`);
  for (let i = 0; i < Math.min(30, buttons.length); i++) {
    const text = (await buttons[i].textContent())?.trim();
    const aria = await buttons[i].getAttribute("aria-label");
    if (text && (text.includes("W") || text.includes("Sem") || text.includes("<") || text.includes(">") || aria?.toLowerCase().includes("week") || aria?.toLowerCase().includes("sem"))) {
      console.log(`  [${i}] text="${text}" aria="${aria}"`);
    }
  }

  // Check for select elements
  const selects = await page.locator("select").all();
  console.log(`\nfound ${selects.length} selects`);
  for (let i = 0; i < selects.length; i++) {
    const options = await selects[i].locator("option").all();
    const optStrs: string[] = [];
    for (const o of options) {
      const text = await o.textContent();
      const val = await o.getAttribute("value");
      optStrs.push(`${val}:"${text?.trim()}"`);
    }
    console.log(`  [${i}] options=[${optStrs.slice(0, 5).join(", ")}...]`);
  }
  await browser.close();
})();
