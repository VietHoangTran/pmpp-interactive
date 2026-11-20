import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BOOK, latticeOf, stepsGrayscale, type GrayOp } from '../../sim/mapping';

// Code shown in section 3.2 must be code that is checked. The kernels live in
// labs/ch03/snippets, where the CUDA front-end check parses the .cu files (and
// grayscale_kernel.cu was also run on the CPU emulator). This test fails if the
// copies in S32.tsx drift from those files, if the stepper's line numbers stop
// pointing at the statements they describe, or if an erratum-hunt spot is no
// longer on the line it claims.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'S32.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${constant} in S32.tsx`);
  return m[1].trim();
};
const snippet = (name: string) => fs.readFileSync(path.join(repo, 'labs/ch03/snippets', name), 'utf8').trim();

describe('section 3.2 code matches its checked files', () => {
  it('the corrected CUDA kernel and launch', () => {
    expect(lesson('GRAY_CU')).toBe(snippet('grayscale_kernel.cu'));
  });
  it('the Metal kernel', () => {
    expect(lesson('GRAY_METAL')).toBe(snippet('grayscale_kernel.metal'));
  });
  it('the kernel as printed (the part after the marker line)', () => {
    const parts = snippet('grayscale_as_printed.cu').split('// ---- as printed ----\n');
    expect(parts.length).toBe(2);
    expect(lesson('PRINTED')).toBe(parts[1].trim());
  });
  it('the printed listing keeps the figure\'s 21 lines, with the store on line 19', () => {
    const lines = lesson('PRINTED').split('\n');
    expect(lines.length).toBe(21);
    expect(lines[18]).toContain('Pout[grayOffset] = (unsigned char) 0.21f*r+0.71f*g+0.07f*b;');
    expect(lines[7]).toContain('if (col < width && row < height)');
    expect(lines[9]).toContain('int grayOffset = row*width + col;');
    expect(lines[12]).toContain('int rgbOffset = grayOffset*CHANNELS;');
  });
});

describe('the stepper highlights the statement each step describes', () => {
  const m = /const LINE: Record<GrayOp, number> = \{([\s\S]*?)\};/.exec(src);
  if (!m) throw new Error('no LINE in S32.tsx');
  const LINE = Object.fromEntries([...m[1].matchAll(/(\w+): (\d+)/g)].map((x) => [x[1], Number(x[2])])) as Record<GrayOp, number>;
  const lines = lesson('GRAY_CU').split('\n');
  const expectOn: Record<GrayOp, string> = {
    col: 'int col = blockIdx.x * blockDim.x + threadIdx.x;',
    row: 'int row = blockIdx.y * blockDim.y + threadIdx.y;',
    guard: 'if (col < width && row < height) {',
    grayOffset: 'int grayOffset = row * width + col;',
    rgbOffset: 'int rgbOffset = grayOffset * CHANNELS;',
    r: 'unsigned char r = Pin[rgbOffset];',
    g: 'unsigned char g = Pin[rgbOffset + 1];',
    b: 'unsigned char b = Pin[rgbOffset + 2];',
    sum: '0.299f * r + 0.587f * g + 0.114f * b',
    cast: '(unsigned char)(',
    store: 'Pout[grayOffset] =',
    done: '}',
  };
  it('every op of the model has a line', () => {
    const p = latticeOf(BOOK, 0, 1, 0, 0);
    for (const s of stepsGrayscale(BOOK, p.x, p.y)) expect(LINE[s.op], s.op).toBeGreaterThan(0);
  });
  for (const op of Object.keys(expectOn) as GrayOp[])
    it(`${op} is on line ${LINE[op]}`, () => {
      expect(lines[LINE[op] - 1]).toContain(expectOn[op]);
    });
  it('the stepper shows lines 3 to 21: the kernel, without the launch', () => {
    expect(lines[2].startsWith('// One thread per pixel')).toBe(true);
    expect(lines[20]).toBe('}');
    expect(lines[21]).toBe('');
  });
  it('the launch passes the output first, as the kernel declares it', () => {
    expect(lesson('GRAY_CU')).toContain('colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, m, n);');
    expect(lesson('GRAY_CU')).toContain('colorToGrayscaleConversion(unsigned char* Pout,');
  });
});

describe('erratum-hunt spots sit on the lines they name', () => {
  const printed = lesson('PRINTED').split('\n');
  const hostM = /const PRINTED_HOST = \[([\s\S]*?)\];/.exec(src);
  if (!hostM) throw new Error('no PRINTED_HOST');
  const host = [...hostM[1].matchAll(/'([^']*)'/g)].map((x) => x[1]);
  const spotsM = /const SPOTS: Spot\[\] = \[([\s\S]*?)\n\];/.exec(src);
  if (!spotsM) throw new Error('no SPOTS');
  const spots = [...spotsM[1].matchAll(/\{ id: '(\w+)', where: '([kh])', line: (\d+), text: '([^']*)'(?:, finding: '(\w+)')?/g)].map((x) => ({
    id: x[1], where: x[2], line: Number(x[3]), text: x[4], finding: x[5],
  }));
  it('there are fourteen spots, with unique ids', () => {
    expect(spots.length).toBe(14);
    expect(new Set(spots.map((s) => s.id)).size).toBe(14);
  });
  for (const s of spots)
    it(`${s.id}: "${s.text}" is on ${s.where === 'k' ? 'kernel' : 'host'} line ${s.line}`, () => {
      const line = (s.where === 'k' ? printed : host)[s.line - 1];
      expect(line).toContain(s.text);
    });
  it('three scored defects and the spelling are all reachable', () => {
    for (const f of ['cast', 'constants', 'args', 'spelling']) expect(spots.some((s) => s.finding === f), f).toBe(true);
    // both defects of line 19 are separate spots on that line
    expect(spots.filter((s) => s.where === 'k' && s.line === 19).map((s) => s.finding).sort()).toEqual(['cast', 'constants']);
  });
  it('the printed host call is the one on p. 50', () => {
    expect(host).toEqual([
      'dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);',
      'dim3 dimBlock(16, 16, 1);',
      'colorToGrayscaleConversion<<<dimGrid,dimBlock>>>(Pin_d, Pout_d, m, n);',
    ]);
  });
});
