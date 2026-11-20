import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { FIG_HEAD, PROGRAM_FILE, figLines, type Program } from '../../sim/managedmem';

// Section C.6 shows Fig. C.1 and our explicit-copy port as the model's lines
// (src/sim/managedmem.ts, figLines), so that the stepper can run them. The checked files
// are labs/appC/snippets/c6_sortfile_*.cu, which the CUDA front-end check parses. This
// test fails if the lines the page shows and the files drift apart, character for character.
const repo = path.resolve(__dirname, '../../../..');
const snippet = (name: string) => fs.readFileSync(path.join(repo, 'labs/appC/snippets', name), 'utf8');

/** The body of sortfile in a file: the lines between its opening line and the closing brace. */
function body(src: string): string[] {
  const lines = src.split('\n');
  const start = lines.indexOf(FIG_HEAD);
  if (start < 0) throw new Error('no sortfile in the file');
  const end = lines.indexOf('}', start);
  return lines.slice(start + 1, end);
}

describe('the listings in section C.6 are the checked snippet files', () => {
  for (const p of ['cpu', 'explicit', 'managed'] as Program[])
    it(`${p}: ${PROGRAM_FILE[p]}`, () => {
      expect(figLines(p).map((l) => l.text)).toEqual(body(snippet(PROGRAM_FILE[p])));
    });

  it('the managed port differs from the CPU original in exactly the four lines Fig. C.1 marks', () => {
    const cpu = body(snippet(PROGRAM_FILE.cpu));
    const managed = body(snippet(PROGRAM_FILE.managed));
    expect(cpu).toHaveLength(managed.length);
    const diff = managed.filter((l, i) => l !== cpu[i]).map((l) => l.trim());
    expect(diff).toEqual(['cudaMallocManaged(&data, N);', 'qsort_char<<<grid, block>>>(data, N, 1);', 'cudaDeviceSynchronize();', 'cudaFree(data);']);
  });

  it('the files say where they differ from the page', () => {
    const m = snippet(PROGRAM_FILE.managed);
    expect(m).toContain('<<<...>>>');
    expect(m).toContain('without its semicolon');
    expect(m).toContain('static const dim3 grid(1), block(1);');
    expect(snippet(PROGRAM_FILE.explicit)).toContain('Ours, not the book');
  });

  it('the deleted synchronise is the managed file\'s line, struck out, not removed', () => {
    const ls = figLines('managed', { noSync: true });
    expect(ls.map((l) => l.text)).toEqual(body(snippet(PROGRAM_FILE.managed)));
    expect(ls.filter((l) => l.deleted).map((l) => l.text.trim())).toEqual(['cudaDeviceSynchronize();']);
  });
});
