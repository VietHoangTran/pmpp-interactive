import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { CONVERGENT_SPEC, FIG_VALUES, MIRROR_SPEC, OWNED, PARTNERS, TESTS, WRITERS, runSpec } from '../../sim/reduction';
import { functionSource } from '../../sim/toolchain';

// "Mirror it" (section 10.4, book exercise 3) assembles a kernel from four parts and
// shows labs/ch10/snippets/s104_mirrored.cu as the reference answer. The CUDA front-end
// check parses that file, and src/sim/reduction.test.ts runs it on the emulator. This test
// ties the parts the page offers, and the loop it prints around them, to that file and to
// Fig. 10.8 as labs/ch10/snippets/s103_inplace.cu holds it, so none of them can drift.
const repo = path.resolve(__dirname, '../../../..');
const snippet = (name: string) => fs.readFileSync(path.join(repo, 'labs/ch10/snippets', name), 'utf8');
const page = fs.readFileSync(path.join(__dirname, 'S104.tsx'), 'utf8');
const mirrored = functionSource(snippet('s104_mirrored.cu'), 'MirroredSumReductionKernel')!;
const convergent = functionSource(snippet('s103_inplace.cu'), 'ConvergentSumReductionKernel')!;
const LOOP = 'for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {';

describe('Mirror it and its reference answer', () => {
  it('the page loads the reference answer from the snippet file', () => {
    expect(page).toContain("useLabFile('ch10/snippets/s104_mirrored.cu')");
  });
  it('the loop the page prints is Fig. 10.8\'s, and the answer keeps it', () => {
    expect(page).toContain(`'    ${LOOP}'`);
    expect(convergent.text).toContain(LOOP);
    expect(mirrored.text).toContain(LOOP);
    expect(mirrored.text).toContain('__syncthreads();');
  });
  it('the mirrored parts are the expressions of the answer', () => {
    expect(mirrored.text).toContain(`unsigned int i = ${OWNED[MIRROR_SPEC.owned].code};`);
    expect(mirrored.text).toContain(`if (${TESTS[MIRROR_SPEC.test].code}) {`);
    expect(mirrored.text).toContain(`input[i] += ${PARTNERS[MIRROR_SPEC.partner].code};`);
    const [cond, store] = WRITERS[MIRROR_SPEC.writer].code.split(') ');
    expect(mirrored.text).toContain(`${cond}) {`);
    expect(mirrored.text).toContain(store);
  });
  it('Fig. 10.8\'s parts, the page\'s starting point, are the expressions of the book\'s kernel', () => {
    expect(convergent.text).toContain(`unsigned int i = ${OWNED[CONVERGENT_SPEC.owned].code};`);
    expect(convergent.text).toContain(`if (${TESTS[CONVERGENT_SPEC.test].code}) {`);
    expect(convergent.text).toContain(`input[i] += ${PARTNERS[CONVERGENT_SPEC.partner].code};`);
    expect(convergent.text).toContain('*output = input[0];');
  });
  it('the model runs the answer to the total in the last element, the starting point to the total in element 0', () => {
    const m = runSpec(MIRROR_SPEC, FIG_VALUES);
    expect(m.correct && m.totalInLast).toBe(true);
    const c = runSpec(CONVERGENT_SPEC, FIG_VALUES);
    expect(c.correct).toBe(true);
    expect(c.totalInLast).toBe(false);
  });
});
