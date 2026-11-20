import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { INDEX, arrayLimit } from '../../sim/addrwidth';
import { breakEven, lifetimeTables } from '../../sim/zerocopy';

// The summary page shows labs/appC/snippets/wrap_layer_table.cu, which ./verify.sh
// front-end-checks. This test fails if the page's copy drifts from the file, and
// pins the numbers its two numeric questions state.
const repo = path.resolve(__dirname, '../../../..');
const read = (rel: string) => fs.readFileSync(path.join(repo, rel), 'utf8');
const page = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const facts = read('notes/today/appC-facts.md');
const flat = (s: string) => s.replace(/\s+/g, ' ');

describe('the layer-table program', () => {
  it('is labs/appC/snippets/wrap_layer_table.cu, character for character', () => {
    const m = /export const LAYER_TABLE = String\.raw`([\s\S]*?)`;/.exec(page);
    expect(m).not.toBeNull();
    expect(m![1].trim()).toBe(read('labs/appC/snippets/wrap_layer_table.cu').trim());
  });
  it('keeps the table managed and the keys in device memory, as the question says', () => {
    const file = read('labs/appC/snippets/wrap_layer_table.cu');
    expect(file).toContain('cudaMallocManaged((void**)&layers');
    expect(file).toContain('cudaMalloc((void**)&layers[l].k');
    expect(file).toContain('printf("%f\\n", layers[0].k[0]);    // line B');
  });
  it('rests on the facts it cites', () => {
    expect(flat(facts)).toContain('Hardware coherency does not enable access to GPU-only allocations such as those made with cudaMalloc from the host.');
    expect(flat(facts)).toContain('CPUs can access memory allocated via hipMalloc()');
  });
});

describe('the KV offset question', () => {
  it('overflows int first at layer 16', async () => {
    const { KV, KV_PER_LAYER, KV_FIRST_BAD } = await import('./Wrap');
    expect(KV_PER_LAYER).toBe(2 ** 27);
    expect(INDEX.int32.log2Count).toBe(31);
    expect(KV_FIRST_BAD).toBe(16);
    // layer 15 fits in an int, layer 16 does not
    expect(15 * KV_PER_LAYER).toBeLessThan(2 ** 31);
    expect(16 * KV_PER_LAYER).toBe(2 ** 31);
    // the whole cache of half-precision keys is 8 GiB
    expect(KV.layers * KV_PER_LAYER * 2).toBe(8 * 2 ** 30);
    // and an int index over 2-byte elements stops at 4 GiB, which this cache passes
    expect(arrayLimit('int32', 2).bytes).toBe(4 * 2 ** 30);
  });
});

describe('the embedding question', () => {
  it('breaks even after vocab / 0.9 tokens at the book\'s 10%', async () => {
    const { VOCAB, EMBED_BREAK_EVEN } = await import('./Wrap');
    expect(VOCAB).toBe(128000);
    expect(EMBED_BREAK_EVEN).toBeCloseTo(VOCAB / 0.9, 6);
    expect(EMBED_BREAK_EVEN).toBeCloseTo(breakEven(1 / VOCAB, 0.1), 9);
    expect(lifetimeTables(0.1)).toBeCloseTo(1.111, 3);
    expect(Math.round(EMBED_BREAK_EVEN)).toBe(142222);
  });
  it('quotes the facts it cites for mapped atomics', () => {
    expect(flat(facts)).toContain('are not atomic from the point of view of the host or other GPUs.');
    expect(flat(facts)).toContain('does not guarantee that all types of access, for example atomics, are supported on all systems.');
  });
});
