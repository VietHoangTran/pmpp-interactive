// Browser check for chapter 20, "Before you start": the KV-cache prediction and its
// reveal, the milestone readiness table with this machine's reports and with none, the
// road map, and the seven prerequisite questions.
// Usage: node e2e/ch20-start.mjs [screenshot-prefix]
import { open, goto, BASE } from './browser.mjs';

const out = process.argv[2];
const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const READY = 'section.bench[aria-label="What lab 20.1 builds on"]';
const MILESTONES = { M2: 'ch05/lab3_gemm', M3: 'ch15/lab6_linear', M4: 'ch10/lab3_softmax_argmax' };
const NAMES = { metal: 'Metal', cuda: 'CUDA', hip: 'HIP' };

// The oracle for the table: lib/reports.ts labStatus, restated.
function labStatus(run, lab, variant) {
  if (variant === 'work' && run?.work_state?.[lab] === 'untouched') return 'none';
  const rs = run?.results.filter((x) => x.lab === lab && x.variant === variant && x.tests.length) ?? [];
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
}
function expected(reports, b, lab) {
  const run = reports[b];
  if (!run) return { work: 'no run on this machine', ref: '' };
  const w = labStatus(run, lab, 'work');
  const r = labStatus(run, lab, 'solution');
  const work = w === 'pass' ? 'yours passes' : w === 'fail' ? 'yours fails' : run.work_state?.[lab] === 'untouched' ? 'not started' : 'no run of yours';
  const ref = r === 'pass' ? `reference verified on ${NAMES[b]}` : r === 'fail' ? 'reference failed here' : 'reference not run';
  return { work, ref };
}
const cells = (page) =>
  page.$$eval(READY + ' td[data-milestone]', (tds) => tds.map((t) => ({ m: t.dataset.milestone, b: t.dataset.backend, work: t.dataset.work, text: t.textContent })));

async function session({ noReports }) {
  const { browser, page, errors } = await open({ width: 1440, height: 1000 });
  let puts = 0;
  await page.setRequestInterception(true);
  page.on('request', (r) => {
    if (r.method() === 'PUT' && r.url().includes('/api/progress')) {
      puts++;
      return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
    }
    if (noReports && r.method() === 'GET' && r.url().includes('/api/reports'))
      return r.respond({ status: 200, contentType: 'application/json', body: '{}' });
    return r.continue();
  });
  await goto(page, 'ch20/start');
  await page.waitForSelector(READY);
  await page.waitForFunction((sel) => !document.querySelector(sel + ' [data-readiness="loading"]'), {}, READY);
  return { browser, page, errors, puts: () => puts };
}

// ---- with this machine's reports
const reports = await fetch(BASE + '/api/reports').then((r) => r.json()).catch(() => ({}));
{
  const { browser, page, errors, puts } = await session({ noReports: false });
  const shown = await cells(page);
  check('the readiness table has three milestones on three backends', shown.length === 9, String(shown.length));
  const bad = shown.filter((c) => {
    const e = expected(reports, c.b, MILESTONES[c.m]);
    return c.work !== e.work || !c.text.includes(e.ref);
  });
  check('every cell matches this machine\'s reports (your work, then the reference)', bad.length === 0, JSON.stringify(bad));
  const any = ['metal', 'cuda', 'hip'].some((b) => reports[b]);
  const status = await page.$eval(READY + ' [data-readiness]', (e) => e.dataset.readiness + '|' + e.textContent);
  check('the status line says where lab 20.1 builds from', status.startsWith(any ? 'runs|' : 'none|') && status.includes('M3 (Q K^T) and M2 (P V)'), status);
  const links = await page.$$eval(READY + ' td a', (as) => as.map((a) => a.getAttribute('href')));
  check('each milestone links to its chapter\'s labs page', JSON.stringify(links) === JSON.stringify(['#/ch05/labs', '#/ch15/labs', '#/ch10/labs']), links.join(' '));
  if (out) await page.screenshot({ path: out + '-start.png' });

  // ---- road map
  const road = await page.$$eval('table[data-table="roadmap"] tbody tr', (rs) => rs.map((r) => r.textContent));
  check('the road map lists the five steps and the five labs', road.length === 5 && ['20.1', '20.2', '20.3', '20.4', '20.5'].every((l) => road.join(' ').includes('Lab ' + l) || road.join(' ').includes('Labs ' + l) || road.join(' ').includes('and ' + l)), road.join(' | '));

  // ---- prediction
  const pred = await page.evaluateHandle(() => [...document.querySelectorAll('.q')].find((q) => q.textContent.includes('KV cache')));
  await (await pred.$('input[type=number]')).type('3');
  await (await pred.$('button.primary')).click();
  await pause(200);
  const reveal = await pred.evaluate((e) => e.textContent);
  check('the reveal gives Eq. 20.8\'s 2 GiB for Llama 2 7B at 4,096 tokens', reveal.includes('1 × 4096 × 2 × 32 × 4096 × 2 B') && reveal.includes('2 GiB') && reveal.includes('512 KiB per token'), reveal);
  check('the reveal judges a guess of 3 GiB in range and links to 20.6', reveal.includes('You were in the right range') && (await pred.$('a[href="#/ch20/20.6"]')) !== null);
  check('committing the prediction writes progress (intercepted)', puts() >= 1, String(puts()));

  // ---- questions
  const q = (part) => page.evaluateHandle((part) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(part)), part);
  const typeIn = async (part, v) => {
    const h = await q(part);
    await (await h.$('input[type=number]')).type(String(v));
    await (await h.$('button.primary')).click();
    await pause(150);
    return h.evaluate((e) => e.textContent);
  };
  const pick = async (part, label) => {
    const h = await q(part);
    await h.evaluate((e, label) => [...e.querySelectorAll('.opts label')].find((l) => l.textContent.trim().startsWith(label)).querySelector('input').click(), label);
    await (await h.$('button.primary')).click();
    await pause(150);
    return h.evaluate((e) => e.textContent);
  };
  const qs = await page.$$eval('.q .kind', (ks) => ks.map((k) => k.textContent).filter((t) => t.startsWith('From chapter')));
  check('seven prerequisite questions', qs.length === 7, qs.join('|'));
  let t = await typeIn('4096 × 4096 weight', 0.5);
  check('q-gemv-ratio: 0.5 FLOP/B is right', t.includes('Correct.') && t.includes('ch. 5 §5.1'), t.slice(0, 80));
  t = await pick('all equal to 100', 'NaN in every entry');
  check('q-overflow: NaN is right', t.includes('Correct.'));
  t = await pick('AMD gfx9', '32: HIP reports');
  check('q-wave64: "32 everywhere" is wrong, with the reason', t.includes('Not quite.') && t.includes('64 on gfx9'), t.slice(0, 120));
  t = await pick('S = Q K^T', "Lab 15.6's launchLinear");
  check('q-qkt-linear: launchLinear is right', t.includes('Correct.'));
  const hrefs = await page.$$eval('.q .verdict a', (as) => as.map((a) => a.getAttribute('href')));
  check('answered explanations link to where the idea was taught', hrefs.includes('#/ch05/5.1') && hrefs.includes('#/ch15/15.1') && hrefs.includes('#/ch04/4.4'), hrefs.join(' '));

  check('no console or page errors (with reports)', errors.length === 0, errors.join(' | '));
  await browser.close();
}

// ---- with no reports at all
{
  const { browser, page, errors } = await session({ noReports: true });
  const shown = await cells(page);
  check('without reports every cell says "no run on this machine"', shown.length === 9 && shown.every((c) => c.work === 'no run on this machine'), JSON.stringify(shown.map((c) => c.work)));
  const status = await page.$eval(READY + ' [data-readiness]', (e) => e.dataset.readiness + '|' + e.textContent);
  check('without reports the status line says there are no runs', status.startsWith('none|No runs on this machine yet.'), status);
  if (out) await page.screenshot({ path: out + '-start-noreports.png' });
  check('no console or page errors (no reports)', errors.length === 0, errors.join(' | '));
  await browser.close();
}

console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
