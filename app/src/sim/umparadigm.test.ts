import { describe, expect, it } from 'vitest';
import { ROWS, checklist, granularityText, isLabC1, paradigm, paradigmFromRun, parseParadigmRow, type Attrs, type RunLike } from './umparadigm';

const A = (c: number, p: number, h: number): Attrs => ({
  concurrentManagedAccess: !!c,
  pageableMemoryAccess: !!p,
  usesHostPageTables: !!h,
});
const answers = (a: Attrs) => Object.fromEntries(checklist(a).map((c) => [c.key, c.answer]));

describe("NVIDIA's table of paradigms (CUDA guide 2.6.2.1; appC-facts A14)", () => {
  it('has four rows in the guide\'s order', () => {
    expect(ROWS.map((r) => r.key)).toEqual(['limited', 'managed', 'hmm', 'ats']);
  });
  it('ConcurrentManagedAccess = 0 is the limited paradigm', () => {
    expect(paradigm(A(0, 0, 0))).toMatchObject({ row: { key: 'limited' }, listed: true });
  });
  it('1, 0, 0 is full support for managed allocations only', () => {
    expect(paradigm(A(1, 0, 0))).toMatchObject({ row: { key: 'managed' }, listed: true });
  });
  it('1, 1, 0 is HMM; 1, 1, 1 is ATS', () => {
    expect(paradigm(A(1, 1, 0)).row.key).toBe('hmm');
    expect(paradigm(A(1, 1, 1)).row.key).toBe('ats');
  });
  it('combinations the table does not list are flagged, not guessed', () => {
    expect(paradigm(A(1, 0, 1))).toMatchObject({ row: { key: 'managed' }, listed: false });
    expect(paradigm(A(0, 1, 0))).toMatchObject({ row: { key: 'limited' }, listed: false });
    expect(paradigm(A(0, 1, 1)).listed).toBe(false);
  });
});

describe('what works on each paradigm', () => {
  it('limited: no malloc pointers, no concurrent access, no oversubscription, bulk moves (A10)', () => {
    expect(answers(A(0, 0, 0))).toEqual({
      malloc: 'no',
      concurrent: 'no',
      oversubscribe: 'no',
      fault: 'no',
      'cudamalloc-host': 'no',
    });
    expect(granularityText(A(0, 0, 0))).toMatch(/bulk/);
  });
  it('managed only (the C.7 question): concurrent access, faults and oversubscription, but no malloc pointers (A12, A14)', () => {
    expect(answers(A(1, 0, 0))).toEqual({
      malloc: 'no',
      concurrent: 'yes',
      oversubscribe: 'yes',
      fault: 'yes',
      'cudamalloc-host': 'no',
    });
    expect(granularityText(A(1, 0, 0))).toMatch(/^pages/);
  });
  it('HMM and ATS: malloc pointers work; coherence by pages or by cache lines (A11, A12)', () => {
    expect(answers(A(1, 1, 0)).malloc).toBe('yes');
    expect(answers(A(1, 1, 1)).malloc).toBe('yes');
    expect(granularityText(A(1, 1, 0))).toMatch(/^pages/);
    expect(granularityText(A(1, 1, 1))).toMatch(/^cache lines/);
  });
  it('the host never dereferences cudaMalloc memory, on any paradigm (A12)', () => {
    for (const a of [A(0, 0, 0), A(1, 0, 0), A(1, 1, 0), A(1, 1, 1)]) expect(answers(a)['cudamalloc-host']).toBe('no');
  });
  it('an unlisted combination with ConcurrentManagedAccess = 0 says "not stated" for malloc', () => {
    expect(answers(A(0, 1, 0)).malloc).toBe('not stated');
  });
});

describe("lab C.1's 'paradigm' bench row", () => {
  it("reads the row lab C.1 writes (labs/appC/lab1_feeding/main.cpp)", () => {
    expect(parseParadigmRow({ name: 'paradigm', concurrent_managed: 1, pageable_access: 0, uses_host_page_tables: 0, paradigm: 'full, managed allocations only' })).toEqual(A(1, 0, 0));
    expect(parseParadigmRow({ name: 'paradigm', concurrent_managed: 1, pageable_access: 1, uses_host_page_tables: 1, paradigm: 'x' })).toEqual(A(1, 1, 1));
  });
  it('reads the attributes under several spellings', () => {
    expect(parseParadigmRow({ name: 'paradigm', concurrentManagedAccess: 1, pageableMemoryAccess: 0, pageableMemoryAccessUsesHostPageTables: 0 })).toEqual(A(1, 0, 0));
    expect(
      parseParadigmRow({
        name: 'paradigm',
        cudaDevAttrConcurrentManagedAccess: '1',
        cudaDevAttrPageableMemoryAccess: '1',
        cudaDevAttrPageableMemoryAccessUsesHostPageTables: '0',
      }),
    ).toEqual(A(1, 1, 0));
    expect(
      parseParadigmRow({
        name: 'paradigm',
        hipDeviceAttributeConcurrentManagedAccess: 'true',
        hipDeviceAttributePageableMemoryAccess: 'false',
        hipDeviceAttributePageableMemoryAccessUsesHostPageTables: 'false',
      }),
    ).toEqual(A(1, 0, 0));
    expect(parseParadigmRow({ name: 'paradigm', concurrent_managed_access: 0, pageable_memory_access: 0, pageable_memory_access_uses_host_page_tables: 0 })).toEqual(A(0, 0, 0));
  });
  it('returns null when a field is missing or unreadable', () => {
    expect(parseParadigmRow({ name: 'paradigm', concurrentManagedAccess: 1, pageableMemoryAccess: 0 })).toBeNull();
    expect(parseParadigmRow({ name: 'paradigm', concurrentManagedAccess: 'n/a', pageableMemoryAccess: 0, pageableMemoryAccessUsesHostPageTables: 0 })).toBeNull();
  });
  it('finds the row in a run, preferring the solution', () => {
    const run: RunLike = {
      results: [
        { lab: 'ch05/lab2_tiled', variant: 'solution', bench: [{ name: 'paradigm', concurrentManagedAccess: 0, pageableMemoryAccess: 0, pageableMemoryAccessUsesHostPageTables: 0 }] },
        { lab: 'appC/lab1_feed', variant: 'work', bench: [{ name: 'paradigm', concurrentManagedAccess: 0, pageableMemoryAccess: 0, pageableMemoryAccessUsesHostPageTables: 0 }] },
        { lab: 'appC/lab1_feed', variant: 'solution', bench: [{ name: 'dense' }, { name: 'paradigm', concurrentManagedAccess: 1, pageableMemoryAccess: 1, pageableMemoryAccessUsesHostPageTables: 1 }] },
      ],
    };
    expect(paradigmFromRun(run)).toEqual(A(1, 1, 1));
    expect(paradigmFromRun({ results: [] })).toBeNull();
    expect(paradigmFromRun(undefined)).toBeNull();
  });
  it('recognises lab C.1 by its directory', () => {
    expect(isLabC1('appC/lab1_three_ways')).toBe(true);
    expect(isLabC1('appC/lab10_x')).toBe(false);
    expect(isLabC1('ch05/lab1_roofline')).toBe(false);
  });
});
