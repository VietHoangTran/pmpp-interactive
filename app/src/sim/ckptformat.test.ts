import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CHECKPOINTS,
  HEADER_BYTES,
  LAB_EMU,
  LAB_GPU,
  SMOL_HEADER_LEN,
  STORIES_15M,
  STORIES_260K,
  bf16ToF32,
  bf16ToF32Bits,
  checkSize,
  checklist,
  encodeHeader,
  encodeSafetensors,
  fieldBytes,
  fileBytes,
  fitOf,
  floatCount,
  hex,
  layout,
  lengthBytes,
  parseHeader,
  parseSafetensors,
  safetensorsBytes,
  shapeOf,
  smolTensors,
  type Llama2cHeader,
} from './ckptformat';

// Numbers the X3.8 lesson states, pinned to notes/today/x3-facts.md (D1 to D4, E1, E2, appA A2).

describe('llama2.c header (D1)', () => {
  it('is 28 bytes of little-endian int32 that round-trip', () => {
    const b = encodeHeader(STORIES_15M);
    expect(b.length).toBe(HEADER_BYTES);
    expect(parseHeader(b)).toEqual(STORIES_15M);
    const neg: Llama2cHeader = { ...STORIES_260K, vocab_size: -512 };
    expect(parseHeader(encodeHeader(neg))).toEqual(neg);
  });
  it("stories15M's bytes are the ones the stepper shows", () => {
    expect(hex(encodeHeader(STORIES_15M)).join(' ')).toBe(
      '20 01 00 00 00 03 00 00 06 00 00 00 06 00 00 00 06 00 00 00 00 7D 00 00 00 01 00 00',
    );
    expect(fieldBytes(STORIES_15M, 5)).toEqual(['00', '7D', '00', '00']); // 32000
    expect(hex(encodeHeader({ ...STORIES_15M, vocab_size: -32000 }).slice(20, 24))).toEqual(['00', '83', 'FF', 'FF']);
  });
  it('rejects a short buffer and a non-int32 field', () => {
    expect(() => parseHeader(new Uint8Array(27))).toThrow();
    expect(() => encodeHeader({ ...STORIES_15M, dim: 2.5 })).toThrow();
  });
});

describe('E1: file size from the header', () => {
  it('reproduces the Hub sizes of stories15M and stories260K exactly', () => {
    expect(floatCount(STORIES_15M)).toBe(15204000);
    expect(fileBytes(STORIES_15M)).toBe(60816028);
    expect(floatCount(STORIES_260K)).toBe(264128);
    expect(fileBytes(STORIES_260K)).toBe(1056540);
    for (const c of CHECKPOINTS.filter((x) => x.header && x.hubBytes != null)) expect(fileBytes(c.header!)).toBe(c.hubBytes);
  });
  it('lists the blocks in run.c order, the classifier only when vocab_size is negative', () => {
    const names = layout(STORIES_15M).map((b) => b.name);
    expect(names).toEqual([
      'token_embedding_table', 'rms_att_weight', 'wq', 'wk', 'wv', 'wo', 'rms_ffn_weight', 'w1', 'w2', 'w3',
      'rms_final_weight', 'freq_cis_real', 'freq_cis_imag',
    ]);
    const un = layout({ ...STORIES_15M, vocab_size: -32000 });
    expect(un.at(-1)!.name).toBe('wcls');
    expect(fileBytes({ ...STORIES_15M, vocab_size: -32000 }) - fileBytes(STORIES_15M)).toBe(4 * 32000 * 288);
  });
  it('a GQA header shrinks wk and wv (stories260K: 8 heads over 4)', () => {
    const b = layout(STORIES_260K);
    expect(b.find((x) => x.name === 'wk')!.floats).toBe(5 * 32 * 64);
    expect(b.find((x) => x.name === 'wq')!.floats).toBe(5 * 64 * 64);
  });
  it('checks a download by its size', () => {
    expect(checkSize(STORIES_15M, 60816028).ok).toBe(true);
    const short = checkSize(STORIES_15M, 60816028 - 4096);
    expect(short.ok).toBe(false);
    expect(short.hint).toContain('4,096 bytes short');
    const flipped = checkSize(STORIES_15M, 60816028 + 4 * 32000 * 288);
    expect(flipped.hint).toContain('negative vocab_size');
  });
});

describe("Lab X3.2's synthetic models (M10)", () => {
  // The explorer offers them as presets; they must be the shapes the lab's CMake generates.
  const cmake = path.resolve(__dirname, '../../../labs/x3/lab2_tinyllama/provided/pieces.cmake');
  const shape = (name: string): Llama2cHeader | null => {
    if (!fs.existsSync(cmake)) return null;
    const m = new RegExp('set\\(' + name + ' ([^)]*)\\)').exec(fs.readFileSync(cmake, 'utf8'));
    if (!m) return null;
    const arg = (k: string) => Number(new RegExp('--' + k + ' (\\d+)').exec(m[1])![1]);
    return { dim: arg('dim'), hidden_dim: arg('hidden'), n_layers: arg('layers'), n_heads: arg('heads'), n_kv_heads: arg('kv-heads'), vocab_size: arg('vocab'), seq_len: arg('seq-len') };
  };
  it.skipIf(!fs.existsSync(cmake))('match provided/pieces.cmake', () => {
    expect(shape('_x3_emu_shape')).toEqual(LAB_EMU);
    expect(shape('_x3_gpu_shape')).toEqual(LAB_GPU);
  });
  it('have head size 128, so M7 fits them as written', () => {
    for (const h of [LAB_EMU, LAB_GPU]) expect(fitOf({ dim: h.dim, heads: h.n_heads, kvHeads: h.n_kv_heads })).toMatchObject({ headSize: 128, m7: true, perLane: 4 });
  });
});

describe('E2: head sizes', () => {
  it('8, 48 and 64; none is M7’s 128; only 64 is a multiple of 32', () => {
    const fits = Object.fromEntries(CHECKPOINTS.map((c) => [c.id, fitOf(c.arch)]));
    expect(fits.stories260K).toEqual({ headSize: 8, qPerKv: 2, m7: false, perLane: null });
    expect(fits.stories15M).toEqual({ headSize: 48, qPerKv: 1, m7: false, perLane: null });
    expect(fits.smollm2).toEqual({ headSize: 64, qPerKv: 3, m7: false, perLane: 2 });
    expect(fitOf({ dim: 4096, heads: 32, kvHeads: 8 })).toEqual({ headSize: 128, qPerKv: 4, m7: true, perLane: 4 });
  });
  it('reports unusable headers instead of dividing', () => {
    expect(shapeOf({ dim: 100, n_heads: 3, n_kv_heads: 1 }).problems[0]).toContain('not a multiple of n_heads');
    expect(shapeOf({ dim: 96, n_heads: 6, n_kv_heads: 4 }).problems[0]).toContain('not a multiple of n_kv_heads');
  });
});

describe('bfloat16 (appA A2, D4)', () => {
  it('0x3F80 becomes 0x3F800000 = 1.0, exactly', () => {
    expect(bf16ToF32Bits(0x3f80)).toBe(0x3f800000);
    expect(bf16ToF32(0x3f80)).toBe(1);
    expect(bf16ToF32(0xc040)).toBe(-3);
    expect(bf16ToF32Bits(0xffff)).toBe(0xffff0000);
  });
});

describe('safetensors (D3)', () => {
  const specs = [
    { name: 'a', dtype: 'F32' as const, shape: [2, 3] },
    { name: 'b', dtype: 'BF16' as const, shape: [4] },
  ];
  it('parses a synthetic file: N, JSON, offsets relative to the byte buffer', () => {
    const data = new Uint8Array(24);
    new DataView(data.buffer).setFloat32(20, 7.5, true);
    const buf = encodeSafetensors(specs, { metadata: { format: 'pt' }, data: { a: data } });
    const st = parseSafetensors(buf, { whole: true });
    expect(st.metadata).toEqual({ format: 'pt' });
    expect(st.tensors.map((t) => [t.name, t.begin, t.end])).toEqual([['a', 0, 24], ['b', 24, 32]]);
    expect(buf.length).toBe(safetensorsBytes(st.headerLen, specs));
    // The last float of "a" sits at dataStart + its offset, not at the offset itself.
    expect(new DataView(buf.buffer).getFloat32(st.dataStart + 20, true)).toBe(7.5);
  });
  it('reads a header-only range request', () => {
    const buf = encodeSafetensors(specs);
    const st = parseSafetensors(buf);
    expect(parseSafetensors(buf.slice(0, st.dataStart)).tensors.length).toBe(2);
    expect(() => parseSafetensors(buf.slice(0, st.dataStart), { whole: true })).toThrow(/past the end/);
  });
  it("rejects a header that does not start with '{'", () => {
    const buf = encodeSafetensors(specs);
    buf[8] = 0x20; // a space
    expect(() => parseSafetensors(buf)).toThrow(/does not start with/);
  });
  it('rejects a length that runs past the buffer and offsets that disagree with the shape', () => {
    const buf = encodeSafetensors(specs);
    new DataView(buf.buffer).setBigUint64(0, BigInt(buf.length), true);
    expect(() => parseSafetensors(buf)).toThrow(/runs past/);
    const head = JSON.stringify({ x: { dtype: 'F32', shape: [3], data_offsets: [0, 8] } });
    const enc = new TextEncoder().encode(head);
    const b2 = new Uint8Array(8 + enc.length + 12);
    new DataView(b2.buffer).setBigUint64(0, BigInt(enc.length), true);
    b2.set(enc, 8);
    expect(() => parseSafetensors(b2)).toThrow(/span 8 bytes/);
  });
  it('first bytes of SmolLM2-135M: N = 30,528, then the buffer starts at 30,536', () => {
    expect(lengthBytes(SMOL_HEADER_LEN).join(' ')).toBe('40 77 00 00 00 00 00 00');
  });
});

describe('SmolLM2-135M (D4)', () => {
  it('272 tensors, no lm_head, and the Hub size to the byte from the shapes and N', () => {
    const t = smolTensors();
    expect(t.length).toBe(272);
    expect(t.some((x) => x.name.includes('lm_head'))).toBe(false);
    expect(t.find((x) => x.name === 'model.layers.0.self_attn.k_proj.weight')!.shape).toEqual([192, 576]);
    expect(t.reduce((s, x) => s + x.shape.reduce((a, b) => a * b, 1), 0)).toBe(134515008);
    expect(safetensorsBytes(SMOL_HEADER_LEN, t)).toBe(269060552);
    expect(CHECKPOINTS.find((c) => c.id === 'smollm2')!.hubBytes).toBe(269060552);
  });
  it('needs the four conversions; the llama2.c files need none of them', () => {
    const smol = checklist(CHECKPOINTS.find((c) => c.id === 'smollm2')!.arch);
    expect(smol.filter((i) => i.todo).map((i) => i.id)).toEqual(['dtype', 'rope', 'gqa', 'constants', 'classifier', 'head']);
    const s15 = checklist(CHECKPOINTS.find((c) => c.id === 'stories15M')!.arch);
    expect(s15.filter((i) => i.todo).map((i) => i.id)).toEqual(['head']);
    const s260 = checklist(CHECKPOINTS.find((c) => c.id === 'stories260K')!.arch);
    expect(s260.filter((i) => i.todo).map((i) => i.id)).toEqual(['gqa', 'head']);
  });
});

// The C++ check shown in the lesson, compiled and run against files built from the model.
const repo = path.resolve(__dirname, '../../..');
const cxx = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!cxx)('labs/x3/snippets/ckpt_check.cpp', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'x3-ckpt-'));
  const exe = path.join(dir, 'ckpt_check');
  const build = spawnSync(cxx!, ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', path.join(repo, 'labs/x3/snippets/ckpt_check.cpp'), '-o', exe], { encoding: 'utf8' });
  // A file with the right header and the right length (sparse: the weights are not needed).
  const make = (name: string, h: Llama2cHeader, size: number) => {
    const p = path.join(dir, name);
    fs.writeFileSync(p, encodeHeader(h));
    fs.truncateSync(p, size);
    return p;
  };
  const run = (p: string) => spawnSync(exe, [p], { encoding: 'utf8' });
  it('builds without warnings', () => expect(build.status, build.stderr).toBe(0));
  it('agrees with the model on both published headers', () => {
    for (const h of [STORIES_15M, STORIES_260K]) {
      const r = run(make('ok.bin', h, fileBytes(h)));
      expect(r.status).toBe(0);
      expect(r.stdout).toContain(`expected ${fileBytes(h)} bytes, file has ${fileBytes(h)}: ok`);
      const f = fitOf(CHECKPOINTS.find((c) => c.header === h)!.arch);
      expect(r.stdout).toContain(`head size ${f.headSize}, ${f.qPerKv} query heads per KV head, classifier shared`);
    }
  });
  it('fails a truncated download and counts the unshared classifier', () => {
    const r = run(make('short.bin', STORIES_15M, fileBytes(STORIES_15M) - 4096));
    expect(r.status).toBe(1);
    expect(r.stdout).toContain('MISMATCH');
    const un = { ...STORIES_260K, vocab_size: -512 };
    const u = run(make('un.bin', un, fileBytes(un)));
    expect(u.status).toBe(0);
    expect(u.stdout).toContain('classifier unshared');
    const tiny = path.join(dir, 'tiny.bin');
    fs.writeFileSync(tiny, new Uint8Array(10));
    expect(run(tiny).stdout).toContain('shorter than a header');
  });
});
