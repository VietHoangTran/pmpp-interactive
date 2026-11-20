import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Section C.9 shows the virtual-memory declarations as quoted text, not as a lab file:
// nothing compiles them (the CUDA front-end stub has no Driver API). They are kept in
// labs/appC/snippets/c9_vmm_decls.h and c9_vmm_decls_hip.h. This test ties the lesson's
// copy to those files, and the files to the facts file the quotes were recorded in, so a
// declaration cannot be edited, or invented, in one place alone.
const repo = path.resolve(__dirname, '../../../..');
const src = fs.readFileSync(path.join(__dirname, 'SC9.tsx'), 'utf8');
const facts = fs.readFileSync(path.join(repo, 'notes/today/appC-facts.md'), 'utf8');
const file = (name: string) => fs.readFileSync(path.join(repo, 'labs/appC/snippets', name), 'utf8');
const squash = (s: string) => s.replace(/\s+/g, ' ').replace(/\s*([(),*])\s*/g, '$1').trim();
const lesson = (name: string) => {
  const m = new RegExp('export const ' + name + ' = `([\\s\\S]*?)`;').exec(src);
  if (!m) throw new Error(`no ${name} in SC9.tsx`);
  return m[1];
};
/** The declarations in a block: comment lines dropped, statements split at ';'. */
const declarations = (block: string) =>
  block
    .split('\n')
    .filter((l) => !l.trim().startsWith('//'))
    .join('\n')
    .split(';')
    .map(squash)
    .filter(Boolean);

describe('section C.9 shows the snippet files', () => {
  it('the CUDA block is labs/appC/snippets/c9_vmm_decls.h', () => {
    expect(lesson('SIGS_CUDA').trim()).toBe(file('c9_vmm_decls.h').trim());
  });
  it('the HIP block is labs/appC/snippets/c9_vmm_decls_hip.h', () => {
    expect(lesson('SIGS_HIP').trim()).toBe(file('c9_vmm_decls_hip.h').trim());
  });
  it('the caption names both files', () => {
    expect(src).toContain('labs/appC/snippets/c9_vmm_decls.h and c9_vmm_decls_hip.h');
  });
});

describe('the snippet files quote their declarations from notes/today/appC-facts.md', () => {
  it('the four CUDA Driver API declarations (A15)', () => {
    const d = declarations(file('c9_vmm_decls.h'));
    expect(d).toHaveLength(4);
    expect(d.map((x) => /cuMem\w+/.exec(x)?.[0])).toEqual(['cuMemAddressReserve', 'cuMemCreate', 'cuMemMap', 'cuMemSetAccess']);
    for (const x of d) expect(squash(facts)).toContain(x);
  });
  it('the two HIP declarations (B7)', () => {
    const d = declarations(file('c9_vmm_decls_hip.h'));
    expect(d).toHaveLength(2);
    for (const x of d) expect(squash(facts)).toContain(x);
  });
  it('the teardown order named in the comment is the documented one (A15)', () => {
    expect(file('c9_vmm_decls.h')).toContain('cuMemUnmap, cuMemRelease, cuMemAddressFree');
    expect(facts).toContain('Teardown order: `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`');
  });
});
