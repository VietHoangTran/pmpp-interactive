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

/**
 * Opens Chrome on a new page. With `progress: true` (or a seed object) the progress file is
 * stubbed (see stubProgress) and the result also has `puts`, the saves the page attempted.
 */
export async function open({ width = 1440, height = 900, dark = false, progress = false } = {}) {
  const executablePath = CANDIDATES.find((p) => fs.existsSync(p));
  if (!executablePath) throw new Error('No Chrome found. Set CHROME_PATH.');
  const browser = await puppeteer.launch({ executablePath, headless: 'new', args: ['--autoplay-policy=no-user-gesture-required'] });
  const page = await browser.newPage();
  // Checks click choice options by their written index; the app shows readers a shuffled
  // order (components/Ask.tsx displayOrder), so ask it for the written order here.
  await page.evaluateOnNewDocument(() => { window.__PMPP_WRITTEN_ORDER__ = true; });
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  if (dark) await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  page.on('requestfailed', (r) => errors.push('requestfailed: ' + r.url()));
  if (progress) {
    const puts = await stubProgress(page, progress === true ? {} : { seed: progress });
    return { browser, page, errors, puts };
  }
  return { browser, page, errors };
}

export async function goto(page, hash) {
  // The dev server keeps a websocket open, so wait for the app rather than for a quiet network.
  await page.goto(BASE + '/#/' + hash, { waitUntil: 'domcontentloaded' });
  // data-ready appears once the page's own code (a lazily loaded section, the review queue) has rendered.
  await page.waitForSelector('.page[data-ready]', { timeout: 30000 });
  await page.evaluate(() => document.fonts.ready);
  await new Promise((r) => setTimeout(r, 400));
}

// ---------------------------------------------------------------- helpers for chapter checks

/**
 * Serve an empty (or given) progress file and swallow every save, so a check never reads or
 * writes the reader's progress/progress.json. Call before the first goto. Returns the list
 * the saved bodies are pushed onto, newest last. Other requests go through untouched (or to
 * `onRequest(request)`, which must return true when it answered the request itself).
 */
export async function stubProgress(page, { seed = { version: 1 }, onRequest } = {}) {
  const puts = [];
  await page.setRequestInterception(true);
  page.on('request', (r) => {
    if (r.isInterceptResolutionHandled?.()) return;
    if (r.url().includes('/api/progress')) {
      if (r.method() === 'PUT') {
        try {
          puts.push(JSON.parse(r.postData() || 'null'));
        } catch {
          puts.push(null);
        }
        return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
      }
      return r.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(seed) });
    }
    if (onRequest && onRequest(r)) return;
    return r.continue();
  });
  return puts;
}

/**
 * Set an <input> (a range slider, a number box) or a <select> the way a person would, so React
 * sees the change: through the element's native value setter, then input and change events.
 * Returns the value the element holds afterwards (a range clamps and steps it).
 */
export async function setRange(page, selector, value) {
  return page.$eval(
    selector,
    (el, v) => {
      const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, String(v));
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return el.value;
    },
    value,
  );
}
/** The same as setRange, for any input or select. */
export const setValue = setRange;

/**
 * Answer the question (or prediction) with progress id `id` and press its button.
 * - A choice question or a choice prediction: `answer` is the option's written index (its
 *   place in the source, whatever order the page shows it in).
 * - A numeric question or a number prediction: `answer` is the value (a number or a string).
 * Returns { kind, correct, verdict }: `correct` is true or false for a graded question and
 * null for a prediction; `verdict` is the text shown after the button. Throws when the page
 * has no such question, so a renamed id fails the check loudly.
 */
export async function answerQuestion(page, id, answer, { timeout = 5000 } = {}) {
  const sel = `.q[data-question="${CSS_escape(id)}"]`;
  await page.waitForSelector(sel, { timeout });
  const q = await page.$(sel);
  const kind = await q.evaluate((e) => e.dataset.questionKind);
  if (kind === 'choice' || kind === 'predict-choice') {
    const radio = await q.$(`input[type=radio][data-written-index="${Number(answer)}"]`);
    if (!radio) throw new Error(`question ${id} has no option with written index ${answer}`);
    await radio.click();
  } else {
    const input = await q.$('input[type=number]');
    if (!input) throw new Error(`question ${id} has no number box (already committed?)`);
    await input.evaluate((el, v) => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(v));
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }, answer);
  }
  await q.$eval('button.primary', (b) => b.click());
  await page.waitForFunction((s) => !!document.querySelector(s + ' .verdict'), { timeout }, sel);
  return q.evaluate((e, k) => {
    const v = e.querySelector('.verdict');
    const graded = k === 'choice' || k === 'numeric';
    return { kind: k, correct: graded ? v.classList.contains('right') : null, verdict: v.textContent };
  }, kind);
}

// CSS.escape is a browser API; ids are plain (letters, digits, / . - _), so quoting is enough.
const CSS_escape = (s) => String(s).replace(/["\\]/g, '\\$&');
