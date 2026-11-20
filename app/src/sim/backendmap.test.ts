import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BACKENDS,
  DIFFER_ROWS,
  ENTRY_POINTS,
  METAL_NORM_NAMES,
  ONE_CALL,
  OPS,
  PROJECTS,
  ROUTES,
  ROUTE_CHOICES,
  TAGS,
  VARIANT_MECHANISMS,
  cell,
  cellLink,
  factsFile,
  fuseGraph,
  gradeSort,
  metalNormName,
  route,
  scoreWalk,
  whereLink,
  attentionHome,
  hopOrder,
  type Cell,
  type Project,
} from './backendmap';
import { ATTENTION_HOME } from './attnroute';
import { excerptInFacts, pinned } from './x3sources';

const repo = path.resolve(__dirname, '../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const x3 = read('notes/today/x3-facts.md');
const hasItem = (ref: string) => {
  const { file, item } = factsFile(ref);
  return fs.existsSync(path.join(repo, file)) && read(file).includes(`### ${item}.`);
};
const allCells = () =>
  OPS.flatMap((o) => PROJECTS.flatMap((p) => BACKENDS.map((b) => ({ op: o.id, p, b, c: o.cells[p][b] as Cell }))));
const numbers = (lines: number | [number, number]) => (typeof lines === 'number' ? [lines] : lines);
const base = (p: string) => p.split('/').pop()!;

describe('every cell is grounded in a facts item', () => {
  it('nine ops, three projects, three backends', () => {
    expect(OPS.length).toBe(9);
    expect(allCells().length).toBe(81);
  });
  for (const { op, p, b, c } of allCells()) {
    it(`${op} / ${p} / ${b}`, () => {
      expect(c.facts.length).toBeGreaterThan(0);
      for (const f of c.facts) expect(hasItem(f), f).toBe(true);
      for (const t of c.tags) expect(TAGS).toContain(t);
      expect(c.tags.length).toBeGreaterThan(0);
      if (c.readAt === 'master') {
        // Read at master in another chapter's facts pass: link that item's URL, no pinned lines.
        expect(c.url, 'master cells carry their URL').toBeTruthy();
        expect(c.lines).toBeUndefined();
        const cited = c.facts.filter((f) => factsFile(f).file !== 'notes/today/x3-facts.md');
        expect(cited.some((f) => read(factsFile(f).file).includes(c.url!)), c.url).toBe(true);
      } else if (c.file) {
        // A pinned file was read in the x3 pass: its name occurs there, and so do its line numbers.
        expect(x3.includes(base(c.file)), c.file).toBe(true);
        if (c.lines) for (const n of numbers(c.lines)) expect(x3.includes(String(n)), `${c.file} ${n}`).toBe(true);
        expect(cellLink(p as Project, c)).toContain(pinned(p as Project).commit);
      }
      for (const w of c.also ?? []) {
        if (w.lines) for (const n of numbers(w.lines)) expect(x3.includes(String(n)), `${w.path} ${n}`).toBe(true);
        expect(whereLink(p as Project, w)).toContain(pinned(w.project ?? (p as Project)).commit);
      }
    });
  }
});

describe('the HIP column', () => {
  it('llama.cpp: every HIP cell is the CUDA file', () => {
    for (const o of OPS) {
      const cu = o.cells['llama.cpp'].cuda;
      const hip = o.cells['llama.cpp'].hip;
      expect(hip.file).toBe(cu.file);
      expect(hip.kernel).toBe(cu.kernel);
      expect(hip.tags).toContain('same source as CUDA');
      expect(hip.facts).toContain('B9');
    }
  });
  it('PyTorch: the CUDA file hipified, except add (not read), matmul and attention (other libraries)', () => {
    const differ = new Set(['add', 'matmul', 'attention']);
    for (const o of OPS) {
      const cu = o.cells.pytorch.cuda;
      const hip = o.cells.pytorch.hip;
      if (o.id === 'rope') {
        expect(hip.tags).toEqual(['composite of other ops']);
        continue;
      }
      if (differ.has(o.id)) {
        expect(hip.kernel).not.toBe(cu.kernel);
        continue;
      }
      expect(hip.file).toBe(cu.file);
      expect(hip.tags).toContain('hipified at build time');
    }
    expect(cell('add', 'pytorch', 'hip').tags).toContain('not read in the facts pass');
    expect(cell('matmul', 'pytorch', 'hip').kernel).toContain('Composable Kernel');
    expect(cell('attention', 'pytorch', 'hip').kernel).toContain('AOTriton');
  });
  it('MLX: no backend in every HIP cell', () => {
    for (const o of OPS) {
      const hip = o.cells.mlx.hip;
      expect(hip.tags).toEqual(['no backend']);
      expect(hip.file).toBeNull();
      expect(hip.facts).toEqual(['B9']);
    }
  });
});

describe('the cells the lessons lean on', () => {
  it('PyTorch add on CUDA is generated at build time and has no source file', () => {
    const c = cell('add', 'pytorch', 'cuda');
    expect(c.tags).toEqual(['generated at build time']);
    expect(c.file).toBeNull();
    expect(x3).toContain('torchgen writes `UfuncCUDA_{name}.cu` at build time');
  });
  it('MLX SiLU is generated at run time on both backends, with no file', () => {
    for (const b of ['cuda', 'metal'] as const) {
      expect(cell('silu', 'mlx', b).tags).toContain('generated at run time (mx.compile)');
      expect(cell('silu', 'mlx', b).file).toBeNull();
    }
  });
  it('llama.cpp RMSNorm is fused before dispatch on CUDA, HIP and Metal', () => {
    for (const b of BACKENDS) expect(cell('rmsnorm', 'llama.cpp', b).tags).toContain('fused before dispatch');
  });
  it('PyTorch attention on CUDA comes from a submodule', () => {
    expect(cell('attention', 'pytorch', 'cuda').tags).toContain('third-party submodule');
    expect(x3).toContain('third_party/flash-attention');
  });
  it('every llama.cpp Metal cell finds its kernel by a name string or names one', () => {
    for (const o of OPS) {
      const m = o.cells['llama.cpp'].metal;
      if (m.readAt === 'pinned') expect(m.tags).toContain('found by name string');
    }
  });
});

describe('course labs', () => {
  it('every lab that exists is a directory under labs/, every planned lab has none', () => {
    for (const o of OPS)
      for (const l of o.course) {
        if (l.planned) expect(l.dir).toBeNull();
        else expect(fs.existsSync(path.join(repo, 'labs', l.dir!)), l.dir!).toBe(true);
      }
  });
  it('each op belongs to one module section and one of the eight tour groups', () => {
    const groups = new Set(OPS.map((o) => o.tour));
    expect(groups.size).toBe(8);
    for (const o of OPS) expect(o.section).toMatch(/^X3\.[2-7]$/);
  });
});

describe('entry points and the differences table', () => {
  it('every entry point cites existing items', () => {
    for (const e of ENTRY_POINTS) for (const f of e.facts) expect(hasItem(f), f).toBe(true);
  });
  it('the three differences rows are copied from the facts table', () => {
    for (const r of DIFFER_ROWS) expect(x3).toContain(`| ${r.feature} | ${r.cuda} | ${r.hip} | ${r.metal} |`);
  });
});

describe('routes for "Follow the call"', () => {
  it('each hop has exactly one right option and cites existing items', () => {
    for (const r of ROUTES)
      for (const h of r.hops) {
        expect(h.options.filter((o) => o.correct).length, `${r.id}: ${h.prompt}`).toBe(1);
        for (const o of h.options) if (!o.correct) expect(o.why, `${r.id}: ${o.label}`).toBeTruthy();
        for (const f of h.facts) expect(hasItem(f), f).toBe(true);
        expect(x3.includes(base(h.path)), h.path).toBe(true);
        if (h.lines) for (const n of numbers(h.lines)) expect(x3.includes(String(n)), `${h.path} ${n}`).toBe(true);
      }
  });
  it('five choices, the RMS_NORM one with a fusion switch', () => {
    expect(ROUTE_CHOICES.length).toBe(5);
    for (const c of ROUTE_CHOICES) {
      if (c.variants) {
        expect(route(c.variants.on).hops[0].found).toContain('ggml_cuda_can_fuse');
        expect(route(c.variants.off).start).toContain('GGML_CUDA_DISABLE_FUSION');
      } else expect(route(c.id).id).toBe(c.id);
    }
  });
  it('scores the first pick of each hop', () => {
    const r = route('torch-add-cuda');
    const right = r.hops.map((h) => h.options.findIndex((o) => o.correct));
    expect(scoreWalk(r, right)).toEqual({ answered: 4, right: 4 });
    expect(scoreWalk(r, [right[0] === 0 ? 1 : 0, null, null, null])).toEqual({ answered: 1, right: 0 });
  });
});

describe('variant mechanisms', () => {
  it('every item is quoted verbatim in x3-facts', () => {
    for (const v of VARIANT_MECHANISMS) expect(excerptInFacts(v.code, x3), v.code).toBe(true);
  });
  it('grades a sort', () => {
    const all = Object.fromEntries(VARIANT_MECHANISMS.map((v) => [v.id, v.mechanism]));
    expect(gradeSort(all)).toEqual({ right: VARIANT_MECHANISMS.length, total: VARIANT_MECHANISMS.length, wrong: [] });
    expect(gradeSort({ ...all, 'llama-fc': 'compile-time' }).wrong).toEqual(['llama-fc']);
  });
  it('uses all six bins', () => {
    expect(new Set(VARIANT_MECHANISMS.map((v) => v.mechanism)).size).toBe(6);
  });
});

describe('the "One RMSNorm call" scene model', () => {
  it('with fusion on, RMS_NORM and MUL become one call; the plain host function is never reached', () => {
    const d = fuseGraph(ONE_CALL.graph, true);
    expect(d.map((x) => x.nodes)).toEqual([[0], [1, 2], [3]]);
    expect(d[1]).toEqual({ nodes: [1, 2], call: 'ggml_cuda_op_rms_norm_fused', fused: true });
    expect(d.some((x) => x.call === 'ggml_cuda_op_rms_norm')).toBe(false);
  });
  it('with fusion off, each node is its own call', () => {
    const d = fuseGraph(ONE_CALL.graph, false);
    expect(d.map((x) => x.call)).toEqual(['ggml_cuda_op_get_rows', 'ggml_cuda_op_rms_norm', 'binary op (binbcast.cu)', 'ggml_cuda_mul_mat']);
  });
  it('the longest rule wins when an ADD follows', () => {
    expect(fuseGraph(['RMS_NORM', 'MUL', 'ADD'], true)).toEqual([{ nodes: [0, 1, 2], call: 'the fused RMS_NORM, MUL, ADD call', fused: true }]);
  });
  it('on Metal the norm consumes two nodes and asks for kernel_rms_norm_mul_f32, a host_name in norm.metal', () => {
    expect(metalNormName(ONE_CALL.graph, 1)).toEqual({ F: 2, consumed: 2, name: 'kernel_rms_norm_mul_f32' });
    expect(x3).toContain('template [[host_name("kernel_rms_norm_mul_f32")]] kernel kernel_rms_norm_fuse_t kernel_rms_norm_fuse_impl<float, 2>;');
    for (const m of METAL_NORM_NAMES) expect(x3).toContain('`' + m.name + '`');
  });
  it('the scene\'s line numbers are the facts file\'s', () => {
    expect(x3).toContain('`rms_norm_f32` (lines 127 to 272)');
    expect(x3).toContain('static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx)` (`ggml/src/ggml-metal/ggml-metal-ops.cpp` line 187');
    expect(x3).toContain('`GGML_OP_RMS_NORM` → `ggml_metal_op_norm` (414)');
  });
});

describe('the start page prediction: where CUDA attention kernels live', () => {
  it('llama.cpp its own, MLX mixed, PyTorch a submodule first', () => {
    expect(PROJECTS.map((p) => attentionHome(p))).toEqual(['own', 'mixed', 'submodule']);
  });
  it('agrees with X3.5\'s summary table (attnroute.ts ATTENTION_HOME, CUDA rows)', () => {
    const cudaRows = ATTENTION_HOME.filter((r) => r.backend.includes('CUDA'));
    const byName: Record<string, Project> = { 'llama.cpp': 'llama.cpp', MLX: 'mlx', PyTorch: 'pytorch' };
    for (const r of cudaRows) expect(attentionHome(byName[r.project]) === 'own').toBe(r.allOwn);
    expect(cudaRows.length).toBe(3);
  });
});

describe('hop option order', () => {
  it('is a permutation, stable, and does not always put the right option first', () => {
    let firstRight = 0;
    let hops = 0;
    for (const r of ROUTES)
      r.hops.forEach((h, i) => {
        const o = hopOrder(r.id, i, h.options.length);
        expect([...o].sort()).toEqual(h.options.map((_, k) => k));
        expect(hopOrder(r.id, i, h.options.length)).toEqual(o);
        if (h.options[o[0]].correct) firstRight++;
        hops++;
      });
    expect(firstRight).toBeLessThan(hops / 2);
  });
});
