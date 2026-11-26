// Browser check for spaced review, the summary page, theme switching and every route.
// The progress file is faked in both directions, so the real one is untouched.
// Usage: node e2e/app-review.mjs   (CHAPTERS=ch02,ch05 limits the route sweep)
//
// Every step waits for what it needs (a button, a saved progress file) instead of sleeping a
// fixed time, and a step that cannot run is reported as a failed check, never a crash, so the
// route sweep always runs to the end.
import { open, goto, answerQuestion, stubProgress } from './browser.mjs';

const day = (offset) => {
  const d = new Date();
  d.setDate(d.getDate() + offset);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};
// Two questions due (one overdue), one not yet due.
const seeded = {
  version: 1,
  sections: { 'ch05/5.1': { visited: '2026-01-01T00:00:00.000Z', done: true } },
  answers: {},
  review: {
    'ch05/start/q-rowmajor': { box: 1, due: day(-2), seen: 2, lapses: 0 },
    'ch05/start/q-barrier': { box: 0, due: day(0), seen: 1, lapses: 1 },
    'ch05/5.1/q-regime': { box: 2, due: day(5), seen: 3, lapses: 0 },
  },
};

const { browser, page: firstPage, errors } = await open({ width: 1440, height: 1000 });
let page = firstPage;
let puts = await stubProgress(page, { seed: seeded });

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const body = () => page.$eval('.page', (e) => e.textContent).catch(() => '');
/** Runs one part of the check; an exception becomes a failed check and the script goes on. */
async function step(name, fn) {
  try {
    await fn();
  } catch (e) {
    check(`${name}: the step ran to the end`, false, String(e?.message ?? e).split('\n')[0] + ` | page: ${(await body()).slice(0, 160)}`);
  }
}
/** Waits until a button with exactly this text is there and enabled, then clicks it. */
async function clickButton(text, timeout = 10000) {
  await page.waitForFunction(
    (t) => [...document.querySelectorAll('button')].some((b) => b.textContent.trim() === t && !b.disabled),
    { timeout },
    text,
  );
  await page.evaluate((t) => [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === t && !b.disabled).click(), text);
}
/** Waits for a save of the progress file that satisfies `ok`; returns it, or the last one after the timeout. */
async function savedWhere(ok, timeout = 5000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    const hit = [...puts].reverse().find((p) => p && ok(p));
    if (hit) return hit;
    await new Promise((r) => setTimeout(r, 50));
  }
  return puts.at(-1);
}
const waitText = (text, timeout = 10000) =>
  page.waitForFunction((t) => document.querySelector('.page')?.textContent.includes(t), { timeout }, text);

// ---- rail reflects progress ----
await step('rail', async () => {
  await goto(page, 'ch05/5.1');
  check('rail shows how many reviews are due', (await page.$eval('.rail', (e) => e.textContent)).includes('2 due'));
  check('rail ticks a finished section', (await page.$eval('.rail nav a[aria-current="page"]', (e) => e.textContent)).includes('✓'));
});

// ---- review session ----
await step('review session', async () => {
  await goto(page, 'review');
  check('review starts with the most overdue question', (await body()).includes('Question 1 of 2') && (await body()).includes('By how many array elements'));
  check('review does not prefill or reveal earlier answers', !(await body()).includes('Correct.') && (await page.$eval('.q input[type=number]', (e) => e.value)) === '');
  await answerQuestion(page, 'ch05/start/q-rowmajor', 6);
  const r1 = (await savedWhere((p) => p.review?.['ch05/start/q-rowmajor']?.box === 2))?.review?.['ch05/start/q-rowmajor'];
  check('a right answer moves the item up a box and pushes it out a week', r1?.box === 2 && r1.due === day(7), JSON.stringify(r1));
  await clickButton('Next question');
  await waitText('Question 2 of 2');
  check('second due question appears', (await body()).includes('Question 2 of 2') && (await body()).includes('__syncthreads()'));
  await answerQuestion(page, 'ch05/start/q-barrier', 2);
  const last = await savedWhere((p) => p.review?.['ch05/start/q-barrier']?.seen === 2);
  const r2 = last?.review?.['ch05/start/q-barrier'];
  check('a wrong answer sends the item back to box 0, due tomorrow, and counts a lapse', r2?.box === 0 && r2.due === day(1) && r2.lapses === 2, JSON.stringify(r2));
  check('the item not yet due was left alone', last?.review?.['ch05/5.1/q-regime']?.due === day(5), JSON.stringify(last?.review?.['ch05/5.1/q-regime']));
  await clickButton('Finish');
  await waitText('That was the last one due today.').catch(() => undefined);
  check('the session ends cleanly', (await body()).includes('That was the last one due today.'), (await body()).slice(0, 200));
});

// ---- summary page ----
await step('summary page', async () => {
  await goto(page, 'ch05/wrap');
  const r = await answerQuestion(page, 'ch05/wrap/q-transpose', 2);
  check('summary: the transpose defect question accepts "only B = 1"', r.correct === true, r.verdict.slice(0, 120));
  check('summary maps all twelve book exercises', (await body()).includes("The book's exercises") && (await page.$$('.page table.data tbody tr')).length >= 10);
});

// ---- theme ----
await step('theme', async () => {
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  const before = await bg();
  await page.evaluate(() => [...document.querySelectorAll('.rail button')].find((b) => / theme$/.test(b.textContent)).click());
  await page.waitForFunction((b) => getComputedStyle(document.body).backgroundColor !== b, { timeout: 3000 }, before).catch(() => undefined);
  check('theme button switches between light and dark', (await bg()) !== before);
});

// ---- language, persisted preference, and progress shared with the English lesson ----
await step('language', async () => {
  const languagePicker = '.rail select';
  await goto(page, '');
  check('English is the default on the home page', (await body()).includes('From tensors to kernels'));
  await page.select(languagePicker, 'vi');
  await waitText('Từ tensor đến kernel');
  check('Vietnamese mode translates the course map', (await body()).includes('Lộ trình toàn khóa') && (await body()).includes('Kernel lượng tử hóa'));

  await page.reload({ waitUntil: 'domcontentloaded' });
  await waitText('Lộ trình toàn khóa');
  check('Vietnamese mode survives reload', await page.$eval('html', (el) => el.lang) === 'vi');

  await goto(page, 'ch01/start');
  await waitText('Vì sao có hai hướng thiết kế bộ xử lý');
  const quiz = '[data-question="ch01/start/q-batch"]';
  await page.click(quiz + ' input[data-written-index="0"]');
  await page.click(quiz + ' button.primary');
  await waitText('Đúng.');
  check('the first Vietnamese quiz uses the original answer key',
    (await savedWhere((p) => p.answers?.['ch01/start/q-batch']?.correct === true))?.answers?.['ch01/start/q-batch']?.value === 0);

  const captionToggle = await page.$$('.scene .check input[type=checkbox]');
  await captionToggle[1].click();
  await waitText('Bắt đầu với một ngân sách công suất cố định');
  check('the first animation offers Vietnamese captions', await page.$eval('.scene .caption', (el) => el.lang) === 'vi');

  await page.setViewport({ width: 390, height: 900, deviceScaleFactor: 1 });
  check('the Vietnamese introductory lesson fits a phone viewport', await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
  await page.select(languagePicker, 'en');
  await waitText('Why there are two kinds of processor');
  check('returning to English keeps the answer on the same page',
    (await page.$eval(quiz + ' .verdict', (el) => el.textContent)).includes('Correct.'));

  await page.select(languagePicker, 'vi');
  await goto(page, 'ch05/5.1');
  check('untranslated lessons still open with their progress and source text',
    (await body()).includes('Nội dung bài này hiện có bằng tiếng Anh.') &&
    (await page.$eval('.rail nav a[aria-current="page"]', (el) => el.textContent)).includes('✓'));
  await page.select(languagePicker, 'en');
});

// ---- every route renders without errors, at desktop and phone widths ----
// The routes come from the app's own chapter registry: every chapter that has sections.
// CHAPTERS=ch02,ch05 limits the sweep (useful while other chapters are being built).
const only = (process.env.CHAPTERS || '').split(',').filter(Boolean);
let built = [];
await step('chapter registry', async () => {
  built = await page.evaluate(async () => {
    const reg = await import('/src/chapters/registry.ts');
    const out = [];
    for (const id of reg.CHAPTER_IDS) {
      try {
        const ch = await reg.loadChapter(id);
        if (ch.sections.length) out.push([id, ch.sections.map((s) => s.id)]);
      } catch {
        out.push([id, ['start']]); // a chapter whose index does not load: its start route will say so
      }
    }
    return out;
  });
});
const routes = ['', 'review', ...built.filter(([id]) => !only.length || only.includes(id)).flatMap(([id, ss]) => ss.map((s) => `${id}/${s}`))];
console.log(`sweeping ${routes.length} routes in ${built.length} built chapters${only.length ? ` (limited to ${only.join(', ')})` : ''}`);

/** A renderer that crashed (out of memory on a long sweep) is replaced, so the sweep goes on. */
async function freshPage(width) {
  try {
    await page.close();
  } catch {
    /* already gone */
  }
  page = await browser.newPage();
  await page.evaluateOnNewDocument(() => {
    window.__PMPP_WRITTEN_ORDER__ = true;
  });
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push('console: ' + m.text());
  });
  puts = await stubProgress(page, { seed: seeded });
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
}
const gone = (e) => /Target closed|Session closed|crash|detached|Protocol error/i.test(String(e?.message ?? e));

for (const width of [1440, 390]) {
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1 }).catch(() => freshPage(width));
  for (const route of routes) {
    const name = `route "${route || 'home'}" at ${width}px`;
    try {
      await goto(page, route);
    } catch (e) {
      const text = await page.$eval('.page', (el) => el.textContent.slice(0, 160)).catch(() => '');
      check(`${name}: loads`, false, text || String(e?.message ?? e).split('\n')[0]);
      if (gone(e) || page.isClosed()) await freshPage(width);
      continue;
    }
    try {
      const info = await page.evaluate(() => ({
        placeholder: document.querySelector('.page').textContent.includes('This section is being written'),
        overflow: document.documentElement.scrollWidth - window.innerWidth,
        h1: document.querySelector('.page h1')?.textContent,
      }));
      check(`${name}: has content, no sideways page scroll`, !info.placeholder && !!info.h1 && info.overflow <= 1, JSON.stringify(info));
    } catch (e) {
      check(`${name}: can be inspected`, false, String(e?.message ?? e).split('\n')[0]);
      if (gone(e) || page.isClosed()) await freshPage(width);
    }
  }
}

check('no console or page errors', errors.length === 0, errors.slice(0, 12).join(' | ') + (errors.length > 12 ? ` | and ${errors.length - 12} more` : ''));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
