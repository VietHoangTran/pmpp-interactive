import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { numericOk, type ChoiceQ, type NumericQ } from '../../components/Ask';
import { DEFAULTS, simulate } from '../../sim/chunkpipe';
import { PP, PP_RESULT, REREC, questionsWrap } from './Wrap';

// The summary page shows one host sequence of its own, for a question. Its checked copy is the
// region [shown] of labs/ch23/snippets/rerecord.cu, which the CUDA front-end check parses. This
// test fails if the page's copy and that file drift apart, runs the front-end check on the file
// (skipped without clang++), and pins the pipeline question's numbers to the stepper's model.
const repo = path.resolve(__dirname, '../../../..');
const SNIP = path.join(repo, 'labs/ch23/snippets');
const region = (file: string, name: string) => {
  const text = fs.readFileSync(path.join(SNIP, file), 'utf8');
  const m = new RegExp(`// \\[${name}\\]\\n([\\s\\S]*?)\\n[ \\t]*// \\[/${name}\\]`).exec(text);
  if (!m) throw new Error(`no region ${name} in ${file}`);
  const lines = m[1].split('\n');
  const ind = Math.min(...lines.filter((l) => l.trim()).map((l) => (/^ */.exec(l) as RegExpExecArray)[0].length));
  return lines.map((l) => l.slice(ind)).join('\n').trim();
};
const correct = (q: ChoiceQ) => q.options.findIndex((o) => o.correct);
const text = (v: unknown) => (typeof v === 'string' ? v : '');

describe('summary page of chapter 23', () => {
  it('has five questions under ch23/wrap, each with a source and one way to be right', () => {
    const ids = Object.keys(questionsWrap);
    expect(ids).toEqual(['ch23/wrap/q-rerecord', 'ch23/wrap/q-decode', 'ch23/wrap/q-pp', 'ch23/wrap/q-ex2', 'ch23/wrap/q-ex3']);
    for (const id of ids) {
      const q = questionsWrap[id];
      expect(q.id).toBe(id);
      expect(q.source).toBeTruthy();
      if (q.kind === 'choice') {
        expect(q.options.filter((o) => o.correct)).toHaveLength(1);
        expect(q.options.filter((o) => !o.correct).every((o) => !!o.why)).toBe(true);
      }
    }
  });
  it('no right answer is the longest option (string labels)', () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind !== 'choice') continue;
      const lengths = q.options.map((o) => text(o.label).length);
      const right = lengths[correct(q)];
      expect(lengths.some((l, i) => i !== correct(q) && l >= right), q.id).toBe(true);
    }
  });
});

describe("the event question's sequence is the checked file", () => {
  it('the page shows region [shown] of rerecord.cu, line for line', () => {
    expect(REREC.trim()).toBe(region('rerecord.cu', 'shown'));
  });
  it('the wait comes between the two records, and consumeC is in the other stream', () => {
    const lines = REREC.split('\n');
    expect(lines[1]).toContain('cudaEventRecord(done, s1)');
    expect(lines[2]).toContain('cudaStreamWaitEvent(s2, done, 0)');
    expect(lines[4]).toContain('cudaEventRecord(done, s1)');
    expect(lines[5]).toContain('consumeC<<<grid, block, 0, s2>>>');
  });
});

describe('the pipeline question is the stepper model with its lanes renamed', () => {
  it(`gives ${PP_RESULT.total} against ${PP_RESULT.serial}, with a bubble of ${PP_RESULT.bubble}`, () => {
    const s = simulate({ ...DEFAULTS, a: 32, b: 8, c: 32, n: 8, streams: 8, engines: 2, order: 'depth', overhead: 0, host: 'pinned', queues: Infinity, view: 'cuda' });
    expect(s.total).toBe(37);
    expect(s.serial).toBe(72);
    expect(s.bubble).toBe(5);
    expect(PP_RESULT).toEqual({ total: 37, serial: 72, bubble: 5, fill: 9 });
    // E1: a + b + c + (n - 1) max(a, b, c), per micro-batch
    expect(PP.gpu + PP.link + PP.gpu + (PP.micro - 1) * Math.max(PP.gpu, PP.link)).toBe(37);
  });
  it('the numeric answers are exact', () => {
    const pp = questionsWrap['ch23/wrap/q-pp'] as NumericQ;
    expect(numericOk(pp, 37)).toBe(true);
    expect(numericOk(pp, 72)).toBe(false);
    const ex2 = questionsWrap['ch23/wrap/q-ex2'] as NumericQ;
    expect(ex2.answer).toBe(4000 / 1000);
  });
});

const clang = process.env.PMPP_CLANG || ['/usr/bin/clang++', '/opt/homebrew/opt/llvm/bin/clang++', '/usr/local/bin/clang++'].find((p) => fs.existsSync(p));
const inc = path.join(repo, 'labs/common/include');
describe.skipIf(!clang)('the CUDA front-end check (clang, the course stub headers)', () => {
  it('passes on rerecord.cu', () => {
    const r = spawnSync(clang as string, ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only', '-std=c++17',
      '-I', inc, '-I', path.join(inc, 'pmpp', 'emu', 'shim'), '-include', 'pmpp/check/cuda_stub.h', '-DTILE_WIDTH=16',
      path.join(SNIP, 'rerecord.cu')], { encoding: 'utf8' });
    expect(r.status, r.stderr).toBe(0);
  });
});
