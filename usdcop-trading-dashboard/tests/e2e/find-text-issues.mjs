import { chromium } from 'playwright';

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();

await page.goto('http://localhost:3000/analysis', { waitUntil: 'domcontentloaded', timeout: 30000 });
for (let i = 0; i < 25; i++) {
  const text = await page.textContent('body');
  if (!text.includes('Cargando analisis')) break;
  await page.waitForTimeout(1000);
}
await page.waitForTimeout(3000);

const results = await page.evaluate(() => {
  const body = document.body.textContent || '';

  // Find "undefined" occurrences with context
  const undefinedMatches = [];
  let idx = 0;
  while ((idx = body.indexOf('undefined', idx)) !== -1) {
    const start = Math.max(0, idx - 40);
    const end = Math.min(body.length, idx + 50);
    undefinedMatches.push(body.slice(start, end).replace(/\s+/g, ' '));
    idx++;
  }

  // Find "null" occurrences with context
  const nullMatches = [];
  idx = 0;
  while ((idx = body.indexOf('null', idx)) !== -1) {
    const start = Math.max(0, idx - 40);
    const end = Math.min(body.length, idx + 44);
    const context = body.slice(start, end).replace(/\s+/g, ' ');
    // Skip false positives like "annulled"
    if (body.slice(idx - 1, idx + 5).match(/\bnull\b/)) {
      nullMatches.push(context);
    }
    idx++;
  }

  // Find empty links
  const links = document.querySelectorAll('a[href^="http"]');
  const emptyLinks = [];
  links.forEach(l => {
    if (!l.textContent?.trim()) {
      emptyLinks.push({ href: l.getAttribute('href')?.slice(0, 80), html: l.innerHTML.slice(0, 120) });
    }
  });

  return { undefinedMatches, nullMatches, emptyLinks };
});

console.log('=== "undefined" in text ===');
results.undefinedMatches.forEach((m, i) => console.log(`  ${i+1}. ...${m}...`));

console.log('\n=== "null" in text ===');
results.nullMatches.forEach((m, i) => console.log(`  ${i+1}. ...${m}...`));

console.log('\n=== Empty links ===');
results.emptyLinks.forEach((l, i) => console.log(`  ${i+1}. href=${l.href}\n     html=${l.html}`));

await browser.close();
