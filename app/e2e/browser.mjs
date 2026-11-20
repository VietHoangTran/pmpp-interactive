// Shared helper: drive the running app in the Chrome installed on this machine.
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';

const CANDIDATES = [
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter(Boolean);

export const BASE = process.env.APP_URL || 'http://localhost:5173';

export async function open({ width = 1440, height = 900, dark = false } = {}) {
  const executablePath = CANDIDATES.find((p) => fs.existsSync(p));
  if (!executablePath) throw new Error('No Chrome found. Set CHROME_PATH.');
  const browser = await puppeteer.launch({ executablePath, headless: 'new', args: ['--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  if (dark) await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  page.on('requestfailed', (r) => errors.push('requestfailed: ' + r.url()));
  return { browser, page, errors };
}

export async function goto(page, hash) {
  // The dev server keeps a websocket open, so wait for the app rather than for a quiet network.
  await page.goto(BASE + '/#/' + hash, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.page h1', { timeout: 20000 });
  await page.evaluate(() => document.fonts.ready);
  await new Promise((r) => setTimeout(r, 400));
}
