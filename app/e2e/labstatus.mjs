// Shared helper for lesson pages that show a lab's files: checks that each listing's status
// is LabKit's solutionTabStatus (components/LabKit.tsx), read from the reports, and that a
// failed run reads as failed. Two passes, each in its own browser: one with this machine's
// reports/runs, one with a fixture in which the listed labs' reference solutions failed on
// Metal and on the emulator. LabKit.tsx is JSX, so its wording is restated here; keep the two
// in step.
import { open, goto, stubProgress, BASE } from './browser.mjs';

const FAILED = 'failed on this machine';
const NO_RUN = 'no run on this machine';
const NAME = { cuda: 'CUDA', hip: 'HIP' };
const WHERE = { cuda: 'CUDA', hip: 'HIP', emu: 'the emulator' };
const and = (xs) => (xs.length < 3 ? xs.join(' and ') : xs.slice(0, -1).join(', ') + ' and ' + xs.at(-1));

/** lib/reports.ts labStatus for a reference solution. */
export function labState(reports, backend, lab) {
  const rs = (reports[backend]?.results ?? []).filter((x) => x.lab === lab && x.variant === 'solution' && x.tests.length);
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
}

/** LabKit's solutionTabStatus(reports, lab, file); `frontend` is GET /api/frontend-check. */
export function tabStatus(reports, lab, file, frontend = {}) {
  const st = (b) => labState(reports, b, lab);
  if (file !== 'kernel.cu') return st('metal') === 'pass' ? 'verified on Metal' : `unverified: ${st('metal') === 'fail' ? FAILED : NO_RUN}`;
  const fam = ['cuda', 'hip', 'emu'];
  const passed = fam.filter((b) => st(b) === 'pass').map((b) => (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`));
  const failed = fam.filter((b) => st(b) === 'fail').map((b) => WHERE[b]);
  const none = ['cuda', 'hip'].filter((b) => st(b) === 'none').map((b) => NAME[b]);
  const fe = st('cuda') === 'pass' ? 'none' : (frontend[`labs/${lab}/solution/kernel.cu`] ?? 'none');
  if (fe === 'ok') passed.push('front-end-checked');
  if (!passed.length && !failed.length && fe === 'none') return `unverified: ${NO_RUN}`;
  return [
    passed.join(', '),
    failed.length ? `unverified on ${and(failed)}: ${FAILED}` : '',
    fe === 'fail' ? 'failed the CUDA front-end check on this machine' : '',
    none.length ? `unverified on ${none.join(' and ')}` : '',
  ]
    .filter(Boolean)
    .join('; ');
}

/** A copy of `reports` in which every lab in `labs` has a failing reference solution on Metal and on the emulator. */
export function failedFixture(reports, labs) {
  const out = JSON.parse(JSON.stringify(reports));
  for (const b of ['metal', 'emu']) {
    out[b] ??= { backend: b, device: { kind: 'device', backend: b, available: true, name: 'e2e fixture' }, results: [], work_state: {} };
    out[b].results = out[b].results.filter((x) => !(labs.includes(x.lab) && x.variant === 'solution'));
    for (const lab of labs)
      out[b].results.push({ lab, variant: 'solution', built: true, test_rc: 1, tests: [{ case: 'e2e fixture: a failed run', pass: false }], bench: [] });
  }
  return out;
}

/** Every tab of every code figure on the page: { label, status, caption }. */
async function codeTabs(page) {
  const out = [];
  const figs = await page.$$('figure.code');
  for (const fig of figs) {
    const n = await fig.evaluate((f) => f.querySelectorAll('.tabs button').length);
    const caption = await fig.evaluate((f) => f.querySelector('figcaption')?.textContent ?? '');
    if (n === 0) {
      const status = await fig.evaluate((f) => f.querySelector('.status')?.textContent ?? '');
      out.push({ label: '', status, caption });
      continue;
    }
    for (let i = 0; i < n; i++) {
      await fig.evaluate((f, k) => f.querySelectorAll('.tabs button')[k].click(), i);
      await new Promise((r) => setTimeout(r, 30));
      out.push(
        await fig.evaluate(
          (f, k, cap) => ({ label: f.querySelectorAll('.tabs button')[k].textContent, status: f.querySelector('.status')?.textContent ?? '', caption: cap }),
          i,
          caption,
        ),
      );
    }
    // Leave the figure on its first tab, as the reader finds it.
    await fig.evaluate((f) => f.querySelector('.tabs button')?.click());
  }
  return out;
}

/**
 * For each expected tab { label, lab, file, prefix?, where? } (`where`: text the figure's
 * caption contains, to tell apart tabs with the same label), checks that every matching tab
 * on the page shows prefix + tabStatus(...), first with this machine's reports and then with
 * the failed fixture, where it must also say "failed on this machine". `reveal` lists the
 * labels of buttons that show a listing the page hides until asked (pressed when present).
 */
export async function checkLabStatuses(check, hash, expected, { reveal = [] } = {}) {
  const real = await (await fetch(BASE + '/api/reports')).json();
  const frontend = await (await fetch(BASE + '/api/frontend-check')).json();
  const labs = [...new Set(expected.map((e) => e.lab))];
  for (const [name, reports] of [
    ["this machine's reports", real],
    ['a fixture where the reference failed on Metal and the emulator', failedFixture(real, labs)],
  ]) {
    const b2 = await openWithReports(reports);
    try {
      await goto(b2.page, hash);
      if (reveal.length) {
        await b2.page.evaluate((labels) => {
          for (const b of document.querySelectorAll('button')) if (labels.includes(b.textContent.trim())) b.click();
        }, reveal);
        await new Promise((r) => setTimeout(r, 500));
      }
      const want = (e) => (e.prefix ?? '') + tabStatus(reports, e.lab, e.file, frontend);
      const matches = (tabs, e) => tabs.filter((t) => t.label === e.label && (!e.where || t.caption.includes(e.where)));
      // Statuses appear once the page has read the reports; give it a few tries.
      let tabs = [];
      for (let i = 0; i < 15; i++) {
        tabs = await codeTabs(b2.page);
        if (expected.every((e) => { const m = matches(tabs, e); return m.length && m.every((t) => t.status === want(e)); })) break;
        await new Promise((r) => setTimeout(r, 300));
      }
      for (const e of expected) {
        const m = matches(tabs, e);
        const w = want(e);
        const failedToo = name.startsWith('a fixture') ? w.includes(FAILED) : true;
        check(
          `lab status: "${e.label}"${e.where ? ` (${e.where})` : ''} with ${name} reads "${w}"`,
          m.length > 0 && m.every((t) => t.status === w) && failedToo,
          JSON.stringify(m.map((t) => t.status)),
        );
      }
      check(`lab status: no console or page errors with ${name}`, b2.errors.length === 0, b2.errors.join(' | '));
    } finally {
      await b2.browser.close();
    }
  }
}

/** Chrome on a new page whose /api/reports answers with `reports`; progress is stubbed as in every check. */
async function openWithReports(reports) {
  const body = JSON.stringify(reports);
  const { browser, page, errors } = await open();
  await stubProgress(page, {
    onRequest: (r) => {
      if (!r.url().includes('/api/reports')) return false;
      r.respond({ status: 200, contentType: 'application/json', body });
      return true;
    },
  });
  return { browser, page, errors };
}
