// node e2e/shot.mjs <hash> <out.png> [--dark] [--light] [--full] [--width N] [--height N] [--y N] [--sel css] [--click text]...
import { open, goto } from './browser.mjs';
const args = process.argv.slice(2);
const [hash, out] = args;
const flag = (n) => args.includes('--' + n);
const val = (n, d) => (args.includes('--' + n) ? Number(args[args.indexOf('--' + n) + 1]) : d);
const { browser, page, errors } = await open({ width: val('width', 1440), height: val('height', 900), dark: flag('dark') });
// Looking at a page must not change the reader's progress file.
await page.setRequestInterception(true);
page.on('request', (r) =>
  r.method() === 'PUT' && r.url().includes('/api/progress')
    ? r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' })
    : r.continue(),
);
await goto(page, hash);
if (val('y', 0)) await page.evaluate((y) => window.scrollTo(0, y), val('y', 0));
// --sel "<css>": scroll that element to the top of the viewport; --light: force the light theme.
const selAt = args.indexOf('--sel');
if (selAt >= 0) await page.evaluate((s) => document.querySelector(s)?.scrollIntoView({ block: 'start' }), args[selAt + 1]);
if (flag('light')) await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
// --click "<button text>" (repeatable): press buttons before the shot.
for (let i = 0; i < args.length; i++)
  if (args[i] === '--click') {
    await page.evaluate((t) => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === t)?.click(), args[i + 1]);
    await new Promise((r) => setTimeout(r, 80));
  }
await new Promise((r) => setTimeout(r, 300));
await page.screenshot({ path: out, fullPage: flag('full') });
console.log(JSON.stringify({ errors, height: await page.evaluate(() => document.documentElement.scrollHeight) }));
await browser.close();
