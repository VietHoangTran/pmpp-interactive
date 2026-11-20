import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { chapter } from './index';

// Chapter 9's contracts between groups of files, checked in one place:
//  - sections 9.2, 9.4, 9.5 and 9.6 show regions of lab files marked `// region: <name>`
//    and `// endregion`; each region must exist, once, in both the CUDA and the Metal file,
//    and hold the kernel the section says it shows;
//  - the predictions in 9.3, 9.4, 9.5 and on the labs page read benchmark rows by name and
//    key; the lab drivers must emit those names and keys;
//  - the index lists every section once, each with a file that exists, and the lab READMEs'
//    prediction ids are asked for on the labs page.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const here = (f: string) => fs.readFileSync(path.join(__dirname, f), 'utf8');

/** The lines strictly between `// region: name` and the next `// endregion`, or null. */
function region(src: string, name: string): string | null {
  const lines = src.split('\n');
  const starts = lines.flatMap((l, i) => (l.trim() === `// region: ${name}` ? [i] : []));
  if (starts.length !== 1) return null;
  const end = lines.findIndex((l, i) => i > starts[0] && l.trim() === '// endregion');
  return end < 0 ? null : lines.slice(starts[0] + 1, end).join('\n');
}

const REGIONS: { section: string; file: string; lab: string; name: string; cuda: string; metal: string }[] = [
  { section: 'S92.tsx', file: 'solution/kernel', lab: 'lab1_racehunt', name: 'basic', cuda: 'histogramBasic', metal: 'histogram_basic' },
  { section: 'S94.tsx', file: 'solution/kernel', lab: 'lab1_racehunt', name: 'pool', cuda: 'histogramPool', metal: 'histogram_pool' },
  { section: 'S94.tsx', file: 'solution/kernel', lab: 'lab2_privatised', name: 'shared', cuda: 'histogramShared', metal: 'histogram_shared' },
  { section: 'S95.tsx', file: 'solution/kernel', lab: 'lab2_privatised', name: 'shared', cuda: 'histogramShared', metal: 'histogram_shared' },
  { section: 'S95.tsx', file: 'provided/ref_kernels', lab: 'lab2_privatised', name: 'contiguous', cuda: 'refHistogramContiguous', metal: 'ref_histogram_contiguous' },
  { section: 'S96.tsx', file: 'solution/kernel', lab: 'lab2_privatised', name: 'runs', cuda: 'histogramRuns', metal: 'histogram_runs' },
];

describe('the lab regions the sections show', () => {
  for (const r of REGIONS) {
    it(`${r.section} shows region "${r.name}" of labs/ch09/${r.lab}/${r.file}.cu and .metal, which hold ${r.cuda} and ${r.metal}`, () => {
      const cu = region(read(`labs/ch09/${r.lab}/${r.file}.cu`), r.name);
      const metal = region(read(`labs/ch09/${r.lab}/${r.file}.metal`), r.name);
      expect(cu).not.toBeNull();
      expect(metal).not.toBeNull();
      expect(cu).toMatch(new RegExp(`__global__ void ${r.cuda}\\(`));
      expect(metal).toMatch(new RegExp(`kernel void ${r.metal}\\(`));
      // The section names the lab and the region it loads.
      const src = here(r.section);
      expect(src.includes(`'${r.name}'`) || src.includes(`name="${r.name}"`)).toBe(true);
      expect(src).toContain(r.lab);
    });
  }
  it('the regions of a solution use the book\'s API: atomic_ref with an explicit scope and relaxed order, never a bare ++ on a bin', () => {
    for (const lab of ['lab1_racehunt', 'lab2_privatised']) {
      const src = read(`labs/ch09/${lab}/solution/kernel.cu`).replace(/\/\/.*$/gm, '');
      expect(src).toMatch(/cuda::atomic_ref<unsigned int, cuda::thread_scope_(block|device)>/);
      expect(src).not.toMatch(/\+\+bins\[/);
      expect(src).not.toMatch(/fetch_add\([^,)]*\)/); // every fetch_add passes an order
    }
  });
});

describe('the benchmark rows the predictions read', () => {
  it('lab 9.1 emits basic and pool rows with data, pixels, seconds and gpixels_per_s', () => {
    const main = read('labs/ch09/lab1_racehunt/main.cpp');
    expect(main).toContain('bench_row(a, ki == 0 ? "basic" : "pool")');
    for (const k of ['"data"', '"pixels"', '"seconds"', '"gpixels_per_s"']) expect(main).toContain(k);
    expect(main).toMatch(/"uniform"/);
    expect(main).toMatch(/"constant"/);
  });
  it('lab 9.2 emits rows with data, coarse, pixels, seconds and gpixels_per_s, for basic, contiguous, shared and runs, on uniform, constant and sky', () => {
    const main = read('labs/ch09/lab2_privatised/main.cpp');
    for (const k of ['"data"', '"coarse"', '"pixels"', '"seconds"', '"gpixels_per_s"']) expect(main).toContain(k);
    for (const n of ['"basic"', '"contiguous"', '"shared"', '"runs"', '"uniform"', '"constant"', '"sky"']) expect(main).toContain(n);
  });
  it('lab 9.3 emits count, count_pool, append and penalties rows with data, batch, len, vocab and seconds, on uniform and zipf ids', () => {
    const main = read('labs/ch09/lab3_tokencounts/main.cpp');
    for (const k of ['"data"', '"batch"', '"len"', '"vocab"', '"seconds"', '"tokens_per_s"', '"elements_per_s"', '"count_pool"', '"append"', '"penalties"'])
      expect(main).toContain(k);
    expect(main).toMatch(/"zipf"/);
  });
  it('the predictions on the sections and the labs page read exactly those names', () => {
    const labs = here('Labs.tsx');
    for (const s of ["'basic'", "'pool'", "'shared'", "'contiguous'", "'runs'", "'count_pool'", "'append'", "'sky'", "'zipf'"]) expect(labs).toContain(s);
    expect(here('S93.tsx')).toContain("r.name === 'basic'");
    expect(here('S94.tsx')).toMatch(/r\.name === 'shared' && num\(r\.coarse\) === 16/);
    expect(here('S95.tsx')).toMatch(/num\(r\.coarse\) === 16 && r\.data === 'uniform'/);
  });
});

describe('the chapter\'s wiring', () => {
  it('lists every section once, in the book\'s order, each with a file that exists', () => {
    expect(chapter.sections.map((s) => s.id)).toEqual(['start', '9.1', '9.2', '9.3', '9.4', '9.5', '9.6', 'labs', 'wrap']);
    const index = here('index.tsx');
    expect(index).not.toMatch(/Temporary|TEMPORARY/);
    for (const m of index.matchAll(/import\('\.\/(\w+)'\)/g)) expect(fs.existsSync(path.join(__dirname, m[1] + '.tsx'))).toBe(true);
    expect(chapter.sections.find((s) => s.id === 'labs')!.pages).toBeNull();
    expect(chapter.sections.find((s) => s.id === 'wrap')!.num).toBe('9.7');
  });
  it('the labs page asks for every prediction id the lab READMEs name, and for the section predictions they point to', () => {
    const labs = here('Labs.tsx');
    for (const dir of ['lab1_racehunt', 'lab2_privatised', 'lab3_tokencounts']) {
      const ids = [...read(`labs/ch09/${dir}/README.md`).matchAll(/`(ch09\/[\w.]+\/p-[\w-]+)`/g)].map((m) => m[1]);
      for (const id of ids) expect(labs).toContain(`id="${id}"`);
    }
    // The three predictions shared with sections use the sections' ids, so one commit counts on both pages.
    for (const [file, id] of [['S93.tsx', 'ch09/9.3/p-skew-slowdown'], ['S94.tsx', 'ch09/9.4/p-shared-speedup'], ['S95.tsx', 'ch09/9.5/p-interleaved-gain']]) {
      expect(here(file)).toContain(`id="${id}"`);
      expect(labs).toContain(`id="${id}"`);
    }
  });
  it('no page of the chapter says it is being written', () => {
    for (const f of fs.readdirSync(__dirname).filter((f) => f.endsWith('.tsx'))) expect(here(f)).not.toMatch(/being written/i);
  });
});
