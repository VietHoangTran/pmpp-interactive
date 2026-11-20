import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  PINNED,
  attribution,
  excerptFragments,
  excerptInFacts,
  linesText,
  permalink,
  pinned,
  shortCommit,
} from './x3sources';

const repo = path.resolve(__dirname, '../../..');
const facts = fs.readFileSync(path.join(repo, 'notes/today/x3-facts.md'), 'utf8');

/** The snapshot table of the facts header: | Project | Branch | Commit read (committed) | Licence |. */
function headerRows() {
  const start = facts.indexOf('| Project | Branch | Commit read');
  const lines = facts.slice(start).split('\n');
  const rows: { project: string; branch: string; commit: string; date: string | null; licence: string }[] = [];
  for (const line of lines.slice(2)) {
    if (!line.startsWith('|')) break;
    const cells = line.split('|').slice(1, -1).map((c) => c.trim());
    const commit = /`([0-9a-f]{40})`/.exec(cells[2])![1];
    const date = /\((\d{4}-\d{2}-\d{2})/.exec(cells[2]);
    rows.push({ project: cells[0], branch: cells[1].replace(/`/g, ''), commit, date: date ? date[1] : null, licence: cells[3] });
  }
  return rows;
}

describe('PINNED is the facts header, row for row', () => {
  const rows = headerRows();
  it('has the same eight projects in the same order', () => {
    expect(rows.length).toBe(8);
    expect(PINNED.length).toBe(rows.length);
  });
  PINNED.forEach((p, i) => {
    it(`${p.id}: repo, branch, commit, date, licence`, () => {
      const r = rows[i];
      expect(r.project).toContain('`' + p.repo + '`');
      expect(r.branch).toBe(p.branch);
      expect(r.commit).toBe(p.commit);
      expect(r.date).toBe(p.date);
      expect(r.licence.startsWith(p.licence)).toBe(true);
      if (p.copyright) expect(r.licence).toContain(p.copyright);
    });
  });
  it('the short commits are the ones the lessons print', () => {
    expect(shortCommit('llama.cpp')).toBe('9197945');
    expect(shortCommit('mlx')).toBe('bd2d1fb');
    expect(shortCommit('pytorch')).toBe('a6ec1fa');
    expect(shortCommit('mlx-lm')).toBe('2d4df8d');
    expect(shortCommit('transformers')).toBe('4d9ef62');
  });
  it('copyright lines match A4', () => {
    expect(facts).toContain(`"${pinned('llama.cpp').copyright}"`);
    expect(facts).toContain(`"${pinned('mlx').copyright}"`);
    expect(attribution('pytorch')).toContain('NOASSERTION');
    expect(facts).toContain('GitHub\'s API reports `NOASSERTION`');
  });
});

describe('permalinks use the pinned commit', () => {
  it('a file with a line range', () => {
    expect(permalink('llama.cpp', 'ggml/src/ggml-cuda/unary.cu', [116, 127])).toBe(
      'https://github.com/ggml-org/llama.cpp/blob/919794500475862f6768c353255469dc661cb734/ggml/src/ggml-cuda/unary.cu#L116-L127',
    );
  });
  it('a single line and a directory', () => {
    expect(permalink('mlx', 'mlx/backend/cuda/softmax.cu', 108)).toMatch(/\/blob\/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a\/mlx\/backend\/cuda\/softmax\.cu#L108$/);
    expect(permalink('mlx', 'mlx/backend', null, true)).toBe('https://github.com/ml-explore/mlx/tree/bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a/mlx/backend');
  });
  it('every permalink the facts file gives for a tour project is one permalink() can build', () => {
    // Each https://github.com/<repo>/blob/<commit>/<path> in the facts file, for the pinned
    // repos, carries exactly the pinned commit.
    const re = /https:\/\/github\.com\/([\w.-]+\/[\w.-]+)\/(?:blob|tree)\/([0-9a-f]{40})\//g;
    let m: RegExpExecArray | null;
    let n = 0;
    while ((m = re.exec(facts))) {
      const p = PINNED.find((x) => x.repo === m![1]);
      expect(p, m[1]).toBeDefined();
      expect(m[2]).toBe(p!.commit);
      n++;
    }
    expect(n).toBeGreaterThan(40);
  });
  it('line text', () => {
    expect(linesText([116, 127])).toBe('lines 116 to 127');
    expect(linesText(154)).toBe('line 154');
    expect(linesText(null)).toBe('');
  });
});

describe('excerpt matching', () => {
  it('splits on elision lines', () => {
    expect(excerptFragments('a\nb\n...\nc')).toEqual(['a\nb', 'c']);
  });
  it('matches a reflowed inline quote and rejects an edited one', () => {
    const q = 'if (FC_OP == OP_UNARY_NUM_SILU) {\n    dst_ptr[i0] = ggml_metal_unary_cast<T>(x / (1 + exp(-x)));\n}';
    expect(excerptInFacts(q, facts)).toBe(true);
    expect(excerptInFacts(q.replace('1 + exp', '1.0f + exp'), facts)).toBe(false);
  });
  it('reads the facts file\'s escaped quotes as plain quotes', () => {
    expect(excerptInFacts('[[host_name("{0}")]]', facts)).toBe(true);
  });
});
