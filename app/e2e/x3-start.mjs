// Browser check for module X3's start page: the snapshot table against sim/x3sources.ts, the tour
// table against sim/backendmap.ts, the attention-home prediction and its reveal, the excerpt
// format, the nine reused prerequisite questions loaded from their chapters, and two questions.
// Usage: APP_URL=http://localhost:6721 node e2e/x3-start.mjs [screenshot-prefix]
import { open, goto } from './browser.mjs';

const out = process.argv[2];
const { browser, page, errors } = await open({ width: 1440, height: 1000 });
// Leave no trace in the real progress file, and start from empty progress so the
// prediction can be committed whatever the reader has already answered.
await page.setRequestInterception(true);
page.on('request', (r) => {
  if (!r.url().includes('/api/progress')) return r.continue();
  if (r.method() === 'PUT') return r.respond({ status: 200, contentType: 'application/json', body: '{"ok":true}' });
  return r.respond({ status: 200, contentType: 'application/json', body: '{"version":1,"sections":{},"answers":{},"review":{}}' });
});

const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const qBox = (needle) => page.evaluateHandle((n) => [...document.querySelectorAll('.q')].find((e) => e.textContent.includes(n)), needle);
async function answerChoice(needle, index) {
  const q = await qBox(needle);
  const radios = await q.$$('input[type=radio]');
  await radios[index].click();
  await (await q.$('button.primary')).click();
  await pause(150);
  return q.evaluate((e) => e.textContent);
}

await goto(page, 'x3/start');
const M = await page.evaluate(async () => {
  const s = await import('/src/sim/x3sources.ts');
  const b = await import('/src/sim/backendmap.ts');
  return {
    pinned: s.PINNED.map((p) => ({ id: p.id, commit: p.commit, short: s.shortCommit(p.id), date: p.date })),
    tours: [...new Set(b.OPS.map((o) => o.tour))],
    home: Object.fromEntries(b.PROJECTS.map((p) => [p, b.attentionHome(p)])),
    torchCommit: s.pinned('pytorch').commit,
  };
});

// The snapshot table is PINNED, row for row.
const rows = await page.$$eval('table[data-part=snapshot] tbody tr', (rs) =>
  rs.map((r) => ({ id: r.dataset.project, commit: r.querySelector('[data-commit]').dataset.commit, short: r.querySelector('[data-commit]').textContent, href: r.querySelector('a').href, text: r.textContent })),
);
check('snapshot table has every pinned project', rows.length === M.pinned.length && rows.length === 8, String(rows.length));
check('each row shows the pinned commit, short and full', rows.every((r, i) => r.id === M.pinned[i].id && r.commit === M.pinned[i].commit && r.short === M.pinned[i].short));
check('each project link is a tree link at its commit', rows.every((r, i) => r.href.includes('/tree/' + M.pinned[i].commit)));
check('dates are shown', rows.every((r, i) => (M.pinned[i].date ? r.text.includes(M.pinned[i].date) : r.text.includes('—'))));
check('llama.cpp copyright line is printed', rows[0].text.includes('Copyright (c) 2023-2026 The ggml authors'));

// The tour table: the eight tour groups, each with its section link.
const tour = await page.$$eval('table[data-part=tour] tbody tr', (rs) => rs.map((r) => ({ t: r.dataset.tour, links: [...r.querySelectorAll('a')].map((a) => a.getAttribute('href')) })));
check('tour table lists the eight tour groups of the op map', tour.length === 8 && tour.every((r, i) => r.t === M.tours[i]), JSON.stringify(tour.map((r) => r.t)));
check('every tour row links to a lab and to an X3 section', tour.every((r) => r.links.some((h) => h.endsWith('/labs')) && r.links.some((h) => /#\/x3\/X3\.\d/.test(h))));

// The excerpt carries the pinned permalink and the licence.
const ex = await page.$eval('figure[data-excerpt=x30-ufunc-add]', (f) => ({ href: f.querySelector('a[data-permalink]').href, text: f.textContent }));
check('excerpt links at the pinned PyTorch commit', ex.href.includes(M.torchCommit));
check('excerpt shows its licence', ex.text.includes('BSD-style') && ex.text.includes('return self + alpha * other;'));

// The prediction: commit one answer per project, then reveal.
const P = '[data-part=predict-home]';
const choice = { 'llama.cpp': 'own', mlx: 'own', pytorch: 'submodule' };
for (const [p, h] of Object.entries(choice)) await page.click(`${P} fieldset[data-project="${p}"] input[data-home=${h}]`);
await page.evaluate((P) => [...document.querySelectorAll(P + ' button')].find((b) => b.textContent.includes('Commit')).click(), P);
await pause(200);
check('prediction is committed and echoed', (await page.$eval(`${P} [data-readout=home-committed]`, (e) => e.textContent)).includes('You predicted'));
await page.evaluate((P) => [...document.querySelectorAll(P + ' button')].find((b) => b.textContent.includes('Show me now')).click(), P);
await pause(150);
const reveal = await page.$$eval(`${P} [data-readout=home-reveal] p[data-project]`, (ps) => ps.map((p) => ({ p: p.dataset.project, right: p.dataset.right })));
check('reveal marks each answer against the model', reveal.length === 3 && reveal.every((r) => (r.right === '1') === (choice[r.p] === M.home[r.p])), JSON.stringify(reveal));
check('the model says own, mixed, submodule', M.home['llama.cpp'] === 'own' && M.home.mlx === 'mixed' && M.home.pytorch === 'submodule');
if (out) await page.screenshot({ path: out + '-predict.png' });

// The nine reused questions load from their own chapters.
await page.waitForFunction(() => document.querySelectorAll('[data-prior] .q').length === 9, { timeout: 30000 }).catch(() => undefined);
const prior = await page.$$eval('[data-prior]', (ds) => ds.map((d) => ({ id: d.dataset.prior, q: !!d.querySelector('.q'), missing: d.hasAttribute('data-missing') })));
check('nine prerequisite questions are shown', prior.length === 9 && prior.every((p) => p.q && !p.missing), JSON.stringify(prior));
const t = await answerChoice('A GPU argmax combines (value, index) pairs', 0);
check('a reused question can be answered', t.includes('Correct.'));

// The page's own questions.
check('q-pinned: the commit answer is right', (await answerChoice('point at a commit hash', 0)).includes('Correct.'));
check('q-hip-same: a wrong answer is explained', (await answerChoice('Where is the HIP softmax kernel?', 1)).includes('Not quite.'));
check('q-hip-same: the right answer', (await answerChoice('Where is the HIP softmax kernel?', 0)).includes('Correct.'));

if (out) await page.screenshot({ path: out + '-end.png', fullPage: true });
check('no console or page errors', errors.length === 0, errors.join(' | '));
await browser.close();
console.log(fail.length ? `\n${fail.length} check(s) failed` : '\nall checks passed');
process.exit(fail.length ? 1 : 0);
