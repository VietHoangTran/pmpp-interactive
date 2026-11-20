import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Lab X1.2 (group 12) links three earlier reference solutions into one program. On CUDA,
// HIP and the emulators it compiles copies kept in labs/x1/shared-g12/, because the
// originals all define laneIdx and warpIdx (and lab 10.3 warpsInBlock) with external
// linkage, and the emulator links every .cu of a lab into one executable. Each copy is the
// original with a note on top and `static` in front of those helpers, nothing else. These
// tests fail the moment an original changes and its copy does not, or the other way round.
// (On Metal the lab loads the originals' kernel.metal files by path and links their launch.mm
// files, so there is nothing to copy there.)
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const MARKER = '// ---- end of the note; the original file follows ----\n';

const COPIES = [
  { copy: 'labs/x1/shared-g12/sortrows_g12.cu', original: 'labs/ch14/lab3_sortrows/solution/kernel.cu', helpers: ['laneIdx', 'warpIdx'] },
  { copy: 'labs/x1/shared-g12/softmax_g12.cu', original: 'labs/ch10/lab3_softmax_argmax/solution/kernel.cu', helpers: ['warpIdx', 'laneIdx', 'warpsInBlock'] },
  { copy: 'labs/x1/shared-g12/topp_g12.cu', original: 'labs/ch11/lab6_topp/solution/kernel.cu', helpers: ['warpIdx', 'laneIdx'] },
];

describe('lab X1.2: the provided copies of labs 14.3, 10.3 and 11.6', () => {
  for (const c of COPIES) {
    describe(path.basename(c.copy), () => {
      const text = read(c.copy);
      const at = text.indexOf(MARKER);
      const body = at < 0 ? '' : text.slice(at + MARKER.length);

      it('starts with a note that names its original', () => {
        expect(at).toBeGreaterThan(0);
        expect(text.slice(0, at)).toContain(c.original);
      });

      it('makes exactly the named helpers static', () => {
        const made = [...body.matchAll(/^static (__device__[^\n]*?\b(\w+)\(\))/gm)].map((m) => m[2]);
        expect(made.sort()).toEqual([...c.helpers].sort());
      });

      it('is otherwise the original, byte for byte', () => {
        const undone = body.replace(/^static (__device__[^\n]*?\b(?:laneIdx|warpIdx|warpsInBlock)\(\))/gm, '$1');
        expect(undone).toBe(read(c.original));
      });
    });
  }

  it('the lab compiles the copies, and the originals on Metal', () => {
    const cmake = read('labs/x1/lab2_filters/CMakeLists.txt');
    for (const c of COPIES) expect(cmake).toContain(path.basename(c.copy));
    for (const c of COPIES) expect(cmake).toContain(c.original.replace('labs/', '').replace('kernel.cu', 'launch.mm'));
    const main = read('labs/x1/lab2_filters/main.cpp');
    for (const c of COPIES) expect(main).toContain(c.original.replace('labs/', '').replace('kernel.cu', 'kernel.metal'));
  });
});
