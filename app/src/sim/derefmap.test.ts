import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  ERAS,
  ERA_ORDER,
  ROWS,
  ROW_LABEL,
  SITES,
  VERDICT_LABEL,
  cell,
  changed,
  citedItems,
  era,
  firstEra,
  grid,
  tone,
  works,
  type Platform,
} from './derefmap';

const repo = path.resolve(__dirname, '../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const facts = {
  appC: read('notes/today/appC-facts.md'),
  ch02: read('notes/today/ch02-facts.md'),
  ch05: read('notes/today/ch05-facts.md'),
};
const notes = read('notes/appC.md');

describe("the notes' table \"Who can dereference what\" (notes/appC.md)", () => {
  it('cudaMalloc pointer: host cannot dereference, kernels can (p. 625)', () => {
    for (const id of ERA_ORDER) {
      expect(cell('cuda', id, 'device', 'host').verdict).toBe('no');
      expect(cell('cuda', id, 'device', 'arg').verdict).toBe('local');
    }
  });
  it('cudaHostAlloc, CUDA 2.2: host yes; kernel only through the cudaHostGetDevicePointer value (p. 624)', () => {
    expect(cell('cuda', 'zerocopy', 'pinned', 'host').verdict).toBe('local');
    expect(cell('cuda', 'zerocopy', 'pinned', 'arg').verdict).toBe('other-ptr');
    expect(cell('cuda', 'zerocopy', 'pinned', 'arg').cost).toContain('cudaHostGetDevicePointer');
  });
  it('the same under UVAS: passed directly; pointers read from memory safe only into cudaHostAlloc memory (p. 625)', () => {
    expect(cell('cuda', 'uva', 'pinned', 'arg').verdict).toBe('remote');
    expect(cell('cuda', 'uva', 'pinned', 'chased').verdict).toBe('remote');
    expect(cell('cuda', 'uva', 'malloc', 'chased').verdict).toBe('no');
    expect(cell('cuda', 'uva', 'malloc', 'arg').verdict).toBe('no');
  });
  it('managed memory, CUDA 6 on Kepler/Maxwell: host yes but not at the same time; flushed at launch (pp. 626–627)', () => {
    expect(cell('cuda', 'um-limited', 'managed', 'host').verdict).toBe('not-while-running');
    expect(cell('cuda', 'um-limited', 'managed', 'arg').verdict).toBe('bulk');
    expect(cell('cuda', 'um-limited', 'managed', 'arg').cost).toContain('cannot exceed GPU memory');
  });
  it('managed memory, Pascal and later: host may be slower; pages fault in; host-built structures can be followed (pp. 627–629)', () => {
    expect(cell('cuda', 'pascal', 'managed', 'host').verdict).toBe('fault');
    expect(cell('cuda', 'pascal', 'managed', 'arg').verdict).toBe('fault');
    expect(cell('cuda', 'pascal', 'managed', 'chased').verdict).toBe('fault');
  });
  it("another GPU's memory, Kepler and later: kernels yes, once access is enabled (p. 626; A8 adds the enabling)", () => {
    expect(cell('cuda', 'uva', 'peer', 'arg').verdict).toBe('no');
    expect(cell('cuda', 'kepler', 'peer', 'arg').verdict).toBe('after-enable');
    expect(cell('cuda', 'kepler', 'peer', 'arg').source).toContain('A8');
  });
  it('the notes table really has these rows', () => {
    for (const s of ['`cudaMalloc()` device pointer', 'same, under UVAS (CUDA 4)', 'Pascal and later', 'another GPU']) expect(notes).toContain(s);
  });
});

describe('the facts table "brought up to date" (notes/today/appC-facts.md)', () => {
  it('CUDA today: cudaMalloc stays host-inaccessible even with hardware coherence (A12)', () => {
    expect(cell('cuda', 'hmm', 'device', 'host').verdict).toBe('no');
    expect(cell('cuda', 'hmm', 'device', 'host').source).toBe('A12');
  });
  it('CUDA today: plain malloc only with HMM or ATS (A12); registered memory through a distinct device address (A3)', () => {
    expect(cell('cuda', 'pascal', 'malloc', 'arg').verdict).toBe('no');
    expect(cell('cuda', 'hmm', 'malloc', 'arg').verdict).toBe('fault');
    expect(cell('cuda', 'uva', 'registered', 'arg').verdict).toBe('other-ptr');
  });
  it('HIP: hipMalloc host-readable as zero copy; hipHostMalloc same pointer; managed without XNACK behaves as pinned (B2, B3, B5)', () => {
    expect(cell('hip', 'hmm', 'device', 'host').verdict).toBe('remote');
    expect(cell('hip', 'hmm', 'pinned', 'arg').verdict).toBe('remote');
    expect(cell('hip', 'hmm', 'managed', 'arg').verdict).toBe('remote');
    expect(cell('hip', 'hmm', 'managed', 'arg', { xnack: true }).verdict).toBe('fault');
    expect(cell('hip', 'hmm', 'malloc', 'arg').verdict).toBe('no');
    expect(cell('hip', 'hmm', 'malloc', 'arg', { xnack: true }).verdict).toBe('fault');
    expect(cell('hip', 'hmm', 'registered', 'arg').verdict).toBe('unknown');
    expect(cell('hip', 'hmm', 'peer', 'arg').verdict).toBe('after-enable');
  });
  it('Metal: private buffer not CPU-visible; shared buffer used in place; no peers; malloc not GPU-visible until wrapped (C2, C4, ch05 C6)', () => {
    expect(cell('metal', 'hmm', 'device', 'host').verdict).toBe('no');
    expect(cell('metal', 'hmm', 'pinned', 'arg').verdict).toBe('local');
    expect(cell('metal', 'hmm', 'peer', 'arg').verdict).toBe('no');
    expect(cell('metal', 'hmm', 'malloc', 'arg').verdict).toBe('no');
  });
  it('the facts table has those rows', () => {
    for (const s of ['`cudaMalloc` / `hipMalloc`', 'plain `malloc` / `new` / `mmap`', 'a peer GPU']) expect(facts.appC).toContain(s);
  });
});

describe('what the map is for: three discoveries', () => {
  it('UVA unified the addresses (C.3) several eras before page faulting unified access (C.8)', () => {
    expect(firstEra('managed', 'arg', ['fault'])).toBe('pascal');
    const steps = ERA_ORDER.indexOf('pascal') - ERA_ORDER.indexOf('uva');
    expect(steps).toBeGreaterThanOrEqual(3);
    // At UVA nothing about host memory became reachable except what C.2 already allowed.
    expect(changed('uva').filter((x) => x.to !== 'no' && x.row === 'malloc')).toEqual([]);
  });
  it('plain malloc memory reaches kernels only with HMM or ATS, not with Pascal (A12)', () => {
    expect(firstEra('malloc', 'arg', ['fault', 'remote', 'local'])).toBe('hmm');
    expect(firstEra('malloc', 'chased', ['fault', 'remote', 'local'])).toBe('hmm');
  });
  it('cudaMalloc memory never becomes host-dereferenceable on CUDA; HIP documents slow host reads (B3)', () => {
    expect(firstEra('device', 'host', ['local', 'remote', 'fault', 'bulk'])).toBeNull();
    expect(cell('hip', 'hmm', 'device', 'host').verdict).toBe('remote');
  });
});

describe('eras: dates as the book gives them and as our sources correct them', () => {
  it('nine eras: the starting point, seven book steps, one step beyond the book', () => {
    expect(ERA_ORDER).toEqual(['copies', 'fermi', 'zerocopy', 'uva', 'kepler', 'um-limited', 'pascal', 'vmm', 'hmm']);
    expect(ERAS.filter((e) => e.book !== null)).toHaveLength(8);
    expect(era('hmm').strip).toBe('beyond the book');
    expect(ERAS.map((e) => e.strip).slice(1, 8)).toEqual(['C.1', 'C.2', 'C.3', 'C.4–C.5', 'C.6', 'C.7–C.8', 'C.9']);
  });
  it('Fermi: whitepaper 2009, products March 2010 (A2)', () => {
    expect(era('fermi').book).toBe('2009');
    expect(era('fermi').corrected).toEqual({ date: expect.stringContaining('March 2010'), item: 'A2' });
  });
  it('Kepler: 2012, not 2013, and not the first with 64-bit addresses (A6)', () => {
    expect(era('kepler').book).toBe('2013');
    expect(era('kepler').corrected!.date).toMatch(/2012.*Fermi already had 64-bit/);
    expect(era('kepler').corrected!.item).toBe('A6');
  });
  it('CUDA 6: announced November 2013, shipped April 2014 (A9)', () => {
    expect(era('um-limited').corrected).toEqual({ date: 'announced November 2013, shipped April 2014', item: 'A9' });
  });
  it('VMM: CUDA 10.2, not CUDA 11 (A15)', () => {
    expect(era('vmm').book).toBe('CUDA 11, no year');
    expect(era('vmm').corrected).toEqual({ date: 'CUDA 10.2, November 2019', item: 'A15' });
  });
  it('Pascal: CUDA 8 is dated by the toolkit archive; the book gives only the year for Pascal (A11)', () => {
    expect(era('pascal').book).toBe('2016');
    expect(era('pascal').corrected).toEqual({ date: 'CUDA 8 shipped in September 2016; the book gives 2016 for Pascal', item: 'A11' });
    expect(facts.appC).toContain('CUDA 8 is "Sept 2016" in the toolkit archive');
  });
  it('the limited era carries today\'s Windows, WSL and Tegra (A10)', () => {
    expect(era('um-limited').note).toContain('Windows, WSL and some Tegra');
  });
});

describe('every cell is complete and cites something that exists', () => {
  const platforms: [Platform, boolean][] = [['cuda', false], ['hip', false], ['hip', true], ['metal', false]];
  it('every platform, era, row and site has a verdict with a label, a sentence and a source', () => {
    for (const [p, x] of platforms)
      for (const id of ERA_ORDER) {
        const g = grid(p, id, { xnack: x });
        for (const r of ROWS)
          for (const s of SITES) {
            const c = g[r][s];
            expect(VERDICT_LABEL[c.verdict]).toBeTruthy();
            expect(c.cost.length).toBeGreaterThan(10);
            expect(c.cost.endsWith('.')).toBe(true);
            expect(c.source.length).toBeGreaterThan(1);
            expect(c.cost).not.toMatch(/—/);
          }
      }
  });
  it('every facts item cited exists in the facts file it names', () => {
    const items = citedItems().flatMap((src) => {
      const out: [string, string][] = [];
      for (const part of src.split(/[;,]/)) {
        const m = /(?:(ch02|ch05)-facts\s+)?\b([A-D]\d{1,2})\b/.exec(part.trim());
        if (m) out.push([m[1] ?? 'appC', m[2]]);
        // "A3, A5": a bare item after a comma belongs to the same file as the one before it.
      }
      return out;
    });
    expect(items.length).toBeGreaterThan(40);
    for (const [file, id] of items) expect(facts[file as 'appC' | 'ch02' | 'ch05']).toContain(`### ${id}.`);
  });
  it('what our sources do not state is not given a verdict or a book page as if they did', () => {
    // CUDA 6 on Kepler/Maxwell: the book dates linked-structure traversal to Pascal, so the cell is our inference.
    const chased = cell('cuda', 'um-limited', 'managed', 'chased');
    expect(chased.source.startsWith('ours, inferred from p. 626')).toBe(true);
    expect(chased.cost.startsWith('Our inference')).toBe(true);
    // HIP with XNACK: B5 documents GPU-side faulting only; the CPU side is not spelled out.
    expect(cell('hip', 'hmm', 'managed', 'host', { xnack: true }).verdict).toBe('unknown');
    // Metal: no pinned/pageable distinction is documented (C2, infra C12), which is not the same as "none".
    expect(ROW_LABEL.metal.pinned).toBe('a shared buffer (no pinned memory documented for Metal)');
  });
  it('colour: success and failure only; slow verdicts are neutral', () => {
    expect(tone('local')).toBe('ok');
    expect(tone('no')).toBe('fail');
    for (const v of ['remote', 'fault', 'bulk', 'other-ptr', 'after-enable', 'not-while-running', 'unknown'] as const) expect(tone(v)).toBe('neutral');
    expect(works('bulk') && works('fault') && !works('other-ptr')).toBe(true);
  });
  it('Metal and HIP do not depend on the era', () => {
    expect(grid('metal', 'copies')).toEqual(grid('metal', 'hmm'));
    expect(grid('hip', 'copies')).toEqual(grid('hip', 'pascal'));
  });
});
