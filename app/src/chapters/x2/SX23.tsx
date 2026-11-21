import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  LAYOUT_LABEL,
  LAYOUT_WEIGHTS,
  SCENE_BLOCK,
  blockAddress,
  demoBlock,
  isAligned,
  packMlx4,
  packQ4K,
  packQ4_0,
  placeOf,
  qsLoads,
  quantTrace,
  quantizeQ4_0,
  quantizeQ4_1,
  sceneActivations,
  unpackTrace,
  weightsInByte,
  type Layout,
  type UnpackVariant,
} from '../../sim/quant';
import { PackedBlockScene } from './PackedBlockScene';

const FACTS = 'notes/today/x2-facts.md';
const CHECKED = (items: string) => `2026-11-20 in the facts pass (${FACTS}, ${items})`;
const SNIPPET = 'x2/snippets/q4_0_ref.h';
const hex = (v: number, digits: number) => (v >>> 0).toString(16).toUpperCase().padStart(digits, '0');

// ---------------------------------------------------------------- questions
export const questionsSX23: Record<string, Question> = {
  'x2/X2.3/q-byte9': {
    kind: 'numeric',
    id: 'x2/X2.3/q-byte9',
    prompt: <>In a llama.cpp Q4_0 block, which weight (0 to 31) is in the high nibble of quant byte 9?</>,
    answer: weightsInByte('q4_0', 9)[1],
    explain: <p>Byte j holds weight j low and weight j + 16 high, so byte 9 holds weights 9 and 25.</p>,
    source: `${FACTS} D1 (quantize_row_q4_0_ref, dequantize_row_q4_0)`,
  },
  'x2/X2.3/q-mlx13': {
    kind: 'choice',
    id: 'x2/X2.3/q-mlx13',
    prompt: <>MLX 4-bit: where is weight 13 of a group?</>,
    options: [
      { label: 'uint32 word 1, bits 20 to 23', correct: true },
      { label: 'uint32 word 1, bits 8 to 11', why: 'Weight 13 is the sixth of word 1 (13 - 8 = 5), and 5 × 4 = 20.' },
      { label: 'byte 13, low nibble', why: 'That would be one code per byte. MLX packs 8 codes per 32-bit word.' },
      { label: 'uint32 word 0, bits 28 to 31', why: 'Word 0 holds weights 0 to 7 only.' },
    ],
    explain: <p>8 codes per uint32, the first in the 4 least significant bits: word 13 / 8 = 1, position 13 mod 8 = 5, bits 4 × 5 = 20 to 23. Neighbours stay neighbours.</p>,
    source: `${FACTS} D5 (mlx.core.quantize docs)`,
  },
  'x2/X2.3/q-why-j16': {
    kind: 'choice',
    id: 'x2/X2.3/q-why-j16',
    prompt: <>Why does llama.cpp put weight j + 16, rather than weight j + 1, beside weight j in a byte?</>,
    options: [
      {
        label: 'One mask then gives codes j..j+3 in activation order, ready for dp4a.',
        correct: true,
      },
      { label: 'It compresses better: nibbles 16 apart tend to be similar in size.', why: 'Both orders store exactly 4 bits per weight.' },
      { label: 'It keeps every block\'s codes 2-byte aligned within the row.', why: 'Alignment is set by the FP16 d and the block size, not by the nibble order.' },
      { label: 'GPUs load the high nibble of a byte first, so j + 16 comes next.', why: 'No load reads nibbles; loads read bytes and words.' },
    ],
    explain: (
      <p>
        One 32-bit mask yields codes j..j+3, which meet four consecutive int8 activations in one dp4a; the shifted word
        yields j+16..j+19, which meet the activations 16 further on. With neighbours in a byte (MLX's order), masking a
        word gives codes j, j+2, j+4, j+6 in the low nibbles and
        their odd neighbours in the high ones, so a byte dot product would need the activations rearranged. With j and
        j + 16 the masked bytes are already in activation order. The layout is chosen for the kernel that reads it.
      </p>
    ),
    source: `${FACTS} D3 (vecdotq.cuh); the comparison with neighbour packing is ours`,
  },
  'x2/X2.3/q-align': {
    kind: 'choice',
    id: 'x2/X2.3/q-align',
    prompt: <>A row of Q4_0 blocks starts at a 16-byte aligned address. Whose quants can be read as <code>int</code> (4-byte loads): blocks 0, 1, 2, 3?</>,
    options: [
      { label: 'Blocks 1 and 3', correct: true },
      { label: 'All four', why: 'The quants start 2 bytes into an 18-byte block; 18i + 2 is a multiple of 4 only for odd i.' },
      { label: 'Blocks 0 and 2', why: 'Block 0’s quants start at byte 2, which is 2 mod 4.' },
      { label: 'None', why: 'Block 1’s quants start at 18 + 2 = 20, a multiple of 4.' },
    ],
    explain: (
      <p>
        The quants of block i start at 18i + 2: offsets 2, 20, 38, 56. Only 20 and 56 are multiples of 4. A kernel that
        reads every block's quants through <code>int*</code> is undefined for half of them; llama.cpp reads two 16-bit
        halves instead.
      </p>
    ),
    source: `${FACTS} A5, C5`,
  },
  'x2/X2.3/q-int4b': {
    kind: 'choice',
    id: 'x2/X2.3/q-int4b',
    prompt: <>MSL now has <code>uint4b_format</code> and <code>unpack</code>. Why does lab X2.2's Metal kernel unpack with masks instead?</>,
    options: [
      {
        label: 'Their nibble order is undocumented, and so are the GPU families they run on.',
        correct: true,
      },
      { label: 'The packed types are slower than masking the bytes by hand.', why: 'Nobody has published a measurement either way, and nothing here claims one.' },
      { label: 'They work only on values held in threadgroup memory.', why: 'The spec does not restrict them so.' },
      { label: 'They are signed only, so Q4_0\'s unsigned codes cannot use them.', why: 'Both int4b_format and uint4b_format exist.' },
    ],
    explain: (
      <p>
        The types also need Metal 4 with SDK 26.4 (and 4.1 for <code>unpack</code>). A kernel that must decode
        llama.cpp's bytes needs to know which nibble is element 0; the spec does not say, and
        the Feature Set Tables have no row for packed numeric types. Neither MLX nor llama.cpp uses them (a code search,
        not a proof). A lab that tried would have to test the nibble order on the device against a CPU pack first.
      </p>
    ),
    source: `${FACTS} C2`,
  },
};

// ---------------------------------------------------------------- explorer: where each weight lives
const codesFor = (layout: Layout): number[] => {
  if (layout === 'q4_0') return quantizeQ4_0(SCENE_BLOCK).codes;
  if (layout === 'mlx4') return quantTrace('mlx-code', SCENE_BLOCK).codes;
  return [...quantizeQ4_1(SCENE_BLOCK).codes, ...quantizeQ4_1(demoBlock('normal', 0, '-', 8)).codes];
};
const bytesFor = (layout: Layout, codes: number[]): number[] => {
  if (layout === 'q4_0') return packQ4_0(codes);
  if (layout === 'q4_k') return packQ4K(codes);
  return packMlx4(codes).flatMap((w) => [w & 255, (w >>> 8) & 255, (w >>> 16) & 255, w >>> 24]);
};

type Sel = { kind: 'w'; j: number } | { kind: 'b'; b: number };

function WhereLives() {
  const [layout, setLayout] = useState<Layout>('q4_0');
  const [sel, setSel] = useState<Sel>({ kind: 'w', j: 5 });
  const codes = useMemo(() => codesFor(layout), [layout]);
  const bytes = useMemo(() => bytesFor(layout, codes), [layout, codes]);
  const n = LAYOUT_WEIGHTS[layout];
  const pair: [number, number] = sel.kind === 'w' ? (() => {
    const p = placeOf(layout, Math.min(sel.j, n - 1));
    return weightsInByte(layout, p.byte);
  })() : weightsInByte(layout, sel.b);
  const byte = sel.kind === 'w' ? placeOf(layout, Math.min(sel.j, n - 1)).byte : sel.b;
  const focusW = sel.kind === 'w' ? Math.min(sel.j, n - 1) : null;
  const pl = focusW != null ? placeOf(layout, focusW) : null;
  const readout =
    focusW != null && pl
      ? `Weight ${focusW} (code ${codes[focusW]}) is in byte ${pl.byte}, ${pl.high ? 'high' : 'low'} nibble${pl.word != null ? `: uint32 word ${pl.word}, bits ${pl.bits![0]} to ${pl.bits![1]}` : ''}. That byte also holds weight ${pl.high ? pair[0] : pair[1]}.`
      : `Byte ${byte} = 0x${hex(bytes[byte], 2)}: weight ${pair[0]} in the low nibble (${codes[pair[0]]}), weight ${pair[1]} in the high nibble (${codes[pair[1]]}).`;
  return (
    <Bench
      title="Where each weight lives"
      task="Click weight 5 in each layout, then click a byte. Which layouts keep neighbours together?"
      foot={
        <>
          Codes from the seeded block of the X2.2 stepper: Q4_0's rule for Q4_0, MLX's coded affine rule for MLX, and
          Q4_1's rule on two blocks for the Q4_K chunk (Q4_K's own choice of scales is not modelled; only its order is).
          MLX bytes assume a little-endian machine. Orders: {`${FACTS}`} D1, D2, D5.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Layout
          <span className="seg" role="group" aria-label="layout">
            {(['q4_0', 'mlx4', 'q4_k'] as Layout[]).map((l) => (
              <button key={l} aria-pressed={layout === l} onClick={() => setLayout(l)}>{l === 'q4_0' ? 'llama.cpp Q4_0' : l === 'mlx4' ? 'MLX 4-bit' : 'llama.cpp Q4_K chunk'}</button>
            ))}
          </span>
        </label>
        <span className="muted">{LAYOUT_LABEL[layout]}</span>
      </div>
      <div className="mx-label">weights (click one)</div>
      <div data-part="weights" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(2.4rem, 1fr))', gap: '3px', marginBottom: '0.9rem' }}>
        {codes.map((c, j) => {
          const on = pair.includes(j);
          return (
            <button key={j} data-w={j} aria-pressed={on} onClick={() => setSel({ kind: 'w', j })}
              style={{ padding: '0.15rem 0', fontVariantNumeric: 'tabular-nums', lineHeight: 1.15 }}>
              <span style={{ fontSize: '0.68rem', opacity: 0.75 }}>w{j}</span>
              <br />
              {c}
            </button>
          );
        })}
      </div>
      <div className="mx-label t-global">bytes in memory (click one)</div>
      <div data-part="bytes" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(3.3rem, 1fr))', gap: '3px' }}>
        {bytes.map((b, i) => {
          const on = i === byte;
          const [lo, hi] = weightsInByte(layout, i);
          return (
            <button key={i} data-b={i} aria-pressed={on} onClick={() => setSel({ kind: 'b', b: i })}
              style={{ padding: '0.15rem 0', fontVariantNumeric: 'tabular-nums', lineHeight: 1.15, borderColor: 'var(--global)' }}>
              <span style={{ fontSize: '0.68rem', opacity: 0.75 }}>{layout === 'mlx4' && i % 4 === 0 ? `w${i / 4}·` : ''}b{i}</span>
              <br />
              {hex(b, 2)}
              <br />
              <span style={{ fontSize: '0.66rem', opacity: 0.75 }}>{hi}|{lo}</span>
            </button>
          );
        })}
      </div>
      <div className="narrate" role="status" data-part="readout" style={{ marginTop: '0.9rem' }}>{readout}</div>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: unpack one word
function BitRow({ word, width, fields }: { word: number; width: 16 | 32; fields: { lo: number; hi: number; weight: number }[] }) {
  return (
    <div data-part="bits" style={{ display: 'grid', gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))`, gap: '1px', maxWidth: width === 32 ? '46rem' : '26rem' }}>
      {Array.from({ length: width }, (_, k) => {
        const bit = width - 1 - k;
        const f = fields.find((x) => bit >= x.lo && bit <= x.hi);
        const v = (word >>> bit) & 1;
        return (
          <span key={bit} data-bit={bit} title={`bit ${bit}`}
            style={{
              textAlign: 'center', fontVariantNumeric: 'tabular-nums', fontSize: '0.74rem', padding: '0.18rem 0',
              background: f ? 'var(--reg-wash)' : 'var(--cell)', borderStyle: 'solid', borderColor: f ? 'var(--reg)' : 'var(--rule)',
              borderWidth: `1px 1px 1px ${bit % 4 === 3 ? 2 : 1}px`,
            }}>
            {v}
          </span>
        );
      })}
    </div>
  );
}

function Unpack() {
  const [variant, setVariant] = useState<UnpackVariant>('cuda32');
  const [wi, setWi] = useState(1);
  const [k, setK] = useState(0);
  const qs = useMemo(() => quantizeQ4_0(SCENE_BLOCK).bytes.slice(2), []);
  const y = useMemo(() => sceneActivations(), []);
  const words = variant === 'cuda32' ? 4 : 8;
  const steps = useMemo(() => unpackTrace(variant, qs, Math.min(wi, words - 1), y), [variant, qs, wi, words, y]);
  const st = steps[Math.min(k, steps.length - 1)];
  const pick = (v: UnpackVariant) => { setVariant(v); setK(0); setWi(v === 'cuda32' ? 1 : 2); };
  return (
    <Bench
      title="Unpack one word"
      task="Step through both variants on the word that holds weights 5 and 21. Which one never shifts?"
      foot={
        <>
          The block is the stepper's, quantised by Q4_0's reference rule; activations are seeded floats. CUDA variant:
          llama.cpp's <code>get_int_b2</code>, <code>&amp; 0x0F0F0F0F</code> and <code>&gt;&gt; 4</code> (D3, A5). Metal
          variant: llama.cpp's <code>block_q_n_dot_y</code> masks with activations pre-divided (D4); MLX and PyTorch MPS do
          the same (D5, D6). Highlighted bits: what this step singles out.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Kernel idiom
          <span className="seg" role="group" aria-label="variant">
            <button aria-pressed={variant === 'cuda32'} onClick={() => pick('cuda32')}>CUDA: 32-bit word</button>
            <button aria-pressed={variant === 'metal16'} onClick={() => pick('metal16')}>Metal: 16-bit word</button>
          </span>
        </label>
        <label className="field">
          Word
          <select aria-label="word" value={Math.min(wi, words - 1)} onChange={(e) => { setWi(+e.target.value); setK(0); }}>
            {Array.from({ length: words }, (_, i) => (
              <option key={i} value={i}>{variant === 'cuda32' ? `${i}: bytes ${4 * i}–${4 * i + 3}` : `${i}: bytes ${2 * i}–${2 * i + 1}`}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}>Restart</button>
        <button onClick={() => setK(k - 1)} disabled={k === 0}>Back</button>
        <button className="primary" onClick={() => setK(k + 1)} disabled={k >= steps.length - 1}>Step</button>
        <span className="muted" aria-live="polite">step {k + 1} of {steps.length}: {st.op}</span>
      </div>
      <div className="narrate" role="status" data-op={st.op} style={{ marginBottom: '0.8rem' }}>{st.text}</div>
      <BitRow word={st.word} width={st.width} fields={st.fields} />
      {st.out.length ? (
        <table className="data" data-part="out" style={{ marginTop: '0.8rem', maxWidth: '34rem' }}>
          <thead><tr><th>weight</th><th className="num">value</th><th>from</th></tr></thead>
          <tbody>
            {st.out.map((o) => (
              <tr key={o.weight} data-weight={o.weight} data-value={o.value}>
                <td>{o.weight}</td>
                <td className="num t-reg">{Number.isInteger(o.value) ? o.value : o.value.toPrecision(5)}</td>
                <td className="muted">{o.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: blocks end to end
const WIDTHS = [1, 2, 4, 16];
function Alignment() {
  const [i, setI] = useState(0);
  const [width, setWidth] = useState(4);
  const blocks = 8;
  const B = 4.6; // px per byte
  const x0 = 6;
  const W = x0 * 2 + blocks * 18 * B;
  const a = blockAddress(i);
  const loads = qsLoads(i, width);
  const bad = loads.filter((l) => !l.ok).length;
  const okBlocks = Array.from({ length: blocks }, (_, k) => qsLoads(k, width).every((l) => l.ok));
  return (
    <Bench
      title="Blocks end to end"
      task="Drag through the blocks at a 4-byte load width, then try 2 and 16."
      foot={
        <>
          An array of 18-byte Q4_0 blocks from a 16-byte aligned start: a 2-byte d (solid), then 16 bytes of quants.
          Ticks every 4 bytes. A load of w bytes is aligned when its address is a multiple of w (A5: <code>char4</code> and{' '}
          <code>int</code> are 4-byte aligned; C5: the same on Metal). Addresses only; no timing.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Block {i}
          <input type="range" aria-label="block index" min="0" max={blocks - 1} value={i} onChange={(e) => setI(+e.target.value)} />
        </label>
        <label className="field">
          Load width (bytes)
          <span className="seg" role="group" aria-label="load width">
            {WIDTHS.map((w) => <button key={w} aria-pressed={width === w} onClick={() => setWidth(w)}>{w}</button>)}
          </span>
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W} 92`} role="img" aria-label="Byte strip of eight blocks" style={{ width: '100%', minWidth: 600, display: 'block' }}>
        {Array.from({ length: blocks * 18 / 4 + 1 }, (_, k) => (
          <line key={k} x1={x0 + 4 * k * B} x2={x0 + 4 * k * B} y1={8} y2={44} stroke="var(--rule)" />
        ))}
        {Array.from({ length: blocks }, (_, k) => {
          const b = blockAddress(k);
          return (
            <g key={k} opacity={k === i ? 1 : 0.45}>
              <rect x={x0 + b.d * B} y={14} width={2 * B} height={24} fill="var(--global)" />
              <rect x={x0 + b.qs * B} y={14} width={16 * B} height={24} fill="var(--global-wash)" stroke="var(--global)" strokeWidth={0.8} />
              <text x={x0 + (b.start + 9) * B} y={30} fontSize="9" textAnchor="middle">{k}</text>
            </g>
          );
        })}
        {loads.map((l, k) => (
          <rect key={k} data-load={l.addr} data-ok={l.ok ? 'yes' : 'no'} x={x0 + l.addr * B + 0.5} y={52} width={width * B - 1} height={14}
            fill={l.ok ? 'var(--ok-wash)' : 'var(--fail-wash)'} stroke={l.ok ? 'var(--ok)' : 'var(--fail)'} />
        ))}
        <text x={x0} y={84} fontSize="10">loads of block {i}'s quants, {width} bytes each</text>
      </svg>
      </div>
      <div className={'narrate' + (bad ? ' bad' : '')} role="status" data-part="align" style={{ marginTop: '0.6rem' }}>
        Block {i}: d at byte {a.d} ({isAligned(a.d, 2) ? 'aligned for a 2-byte load' : 'misaligned'}), quants at byte {a.qs} = {a.qs % 4} mod 4.{' '}
        {bad ? `${bad} of ${loads.length} ${width}-byte loads are misaligned.` : `All ${loads.length} ${width}-byte loads are aligned.`}{' '}
        At this width the quants of blocks {okBlocks.flatMap((o, k) => (o ? [k] : [])).join(', ') || 'none'} of 0 to {blocks - 1} read cleanly.
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function SX23() {
  const q = questionsSX23;
  const src = useLabFile(SNIPPET);
  return (
    <>
      <p>
        A 4-bit code is half a byte, and no load instruction reads half a byte. So every quantised kernel unpacks: it
        loads bytes or words that hold several codes and pulls them apart with masks and shifts. Where each code sits in
        those bytes is a choice, and the three layouts you are most likely to read make three different ones. This
        section is about why, and about the one trap in reading llama.cpp's blocks.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The book names 4-bit data (appendix A, §A.5, p. 592) and nothing more. The orders, alignments and idioms here
          are from the projects' source and the vendors' references, checked in the facts pass.
        </p>
      </Beyond>
      <p>
        There is no 4-bit integer type in CUDA or HIP C++: you store codes in <code>uint8_t</code> or{' '}
        <code>uint32_t</code> and pack them by hand. Metal has recently added packed 4-bit types, which come at the end
        of this page.
      </p>
      <Beyond title="Below C++">
        <p>
          PTX does have <code>.s4</code> and <code>.u4</code>, but only inside matrix instructions (<code>mma</code> from
          sm_75 and sm_80), <code>ldmatrix</code>'s <code>.s8.s4</code> form (PTX 9.4; sm_90a and the sm_100, sm_110 and sm_120 families) and the
          packing conversion <code>cvt.pack</code>. Checked {CHECKED('A4')}.
        </p>
      </Beyond>

      <h2>Three orders</h2>
      <p>
        <b>llama.cpp Q4_0</b>: byte j of the 16 quant bytes holds weight j in its low nibble and weight j + 16 in its
        high nibble. The two codes in a byte are not neighbours. <b>MLX</b>: 8 codes per <code>uint32</code>, the first
        in the 4 least significant bits; neighbours stay neighbours. <b>llama.cpp Q4_K</b>: within each 64-weight chunk,
        the low nibbles of 32 bytes are the first 32 weights and the high nibbles the next 32: Q4_0's idea at a larger
        size.
      </p>
      <Predict
        id="x2/X2.3/p-byte5"
        prompt={<>Before you use the explorer: in a Q4_0 block, weight 5 shares its byte with which other weight?</>}
        unit="(weight index)"
      >
        {(v) => (
          <p data-part="p-byte5">
            Weight {weightsInByte('q4_0', 5)[1]}: byte 5 holds weights {weightsInByte('q4_0', 5).join(' and ')}.{' '}
            {v === weightsInByte('q4_0', 5)[1] ? 'As you said.' : v === 4 || v === 6 ? 'That is MLX’s order, where neighbours share a byte.' : ''}
          </p>
        )}
      </Predict>
      <WhereLives />
      <Ask q={q['x2/X2.3/q-byte9']} label="Work it out" />
      <Ask q={q['x2/X2.3/q-mlx13']} />

      <h2>Why j and j + 16</h2>
      <p>
        llama.cpp's CUDA decode kernel reads a 32-bit word of quants, four bytes, and computes{' '}
        <code>v &amp; 0x0F0F0F0F</code>. Each byte of the result is one low nibble: the codes of weights j to j + 3, in
        order. Those meet four consecutive int8 activations in one four-way byte dot product. Then{' '}
        <code>(v &gt;&gt; 4) &amp; 0x0F0F0F0F</code> gives weights j + 16 to j + 19, which meet the activations 16 further
        on. One mask, one shift, no shuffling of bytes: the layout serves the kernel. Section{' '}
        <a href={href('x2', 'X2.5')}>X2.5</a> takes that integer path further.
      </p>
      <p>
        The other idiom, used by llama.cpp's Metal kernels, MLX and PyTorch's MPS backend, never shifts. It masks a
        16-bit word four ways: <code>0x000F</code>, <code>0x0F00</code>, <code>0x00F0</code>, <code>0xF000</code>. A masked
        nibble left in place is the code times 1, 256, 16 or 4096, so the kernel divides the matching activations by
        the same powers of two when it loads them. Dividing a float by a power of two is exact (unless it underflows), so
        each product is exactly code × activation. Metal also has <code>extract_bits</code>, which sign-extends for signed
        types; these kernels do not need it.
      </p>
      <Today checked={CHECKED('D3, D4, D5, D6, C1, E4')}>
        <p>
          llama.cpp CUDA: <code>vi0 = (v[i] &gt;&gt; 0) &amp; 0x0F0F0F0F</code>, <code>vi1 = (v[i] &gt;&gt; 4) &amp;
          0x0F0F0F0F</code>. llama.cpp Metal: "we assume that the yl's have been multiplied with the appropriate scale
          factor that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)". MLX's <code>qdot</code> and
          PyTorch's <code>int4pack_mm</code> on MPS do the same.
        </p>
      </Today>
      <Unpack />
      <Ask q={q['x2/X2.3/q-why-j16']} />

      <h2>The alignment trap</h2>
      <p>
        A Q4_0 block is a 2-byte FP16 d followed by 16 bytes of quants: 18 bytes, and only 2-byte aligned. Lay blocks end
        to end and block i's quants start at 18i + 2. For even i that is 2 mod 4, so reading them through{' '}
        <code>int*</code> or <code>char4*</code> (both 4-byte aligned types) is a misaligned access, which is undefined
        behaviour in C++ on every backend. llama.cpp reads the quants as two 16-bit halves on CUDA and HIP (its{' '}
        <code>get_int_b2</code>) and as <code>uint16_t</code> words on Metal. MLX sidesteps the problem: scales, biases and
        codes live in separate arrays, so the codes stay 4-byte aligned.
      </p>
      <Simplification>
        <p>
          Addresses only: the explorer marks which loads are aligned, not what a misaligned load costs or does on a given
          GPU, which no vendor here documents as a single behaviour.
        </p>
      </Simplification>
      <Alignment />
      <Ask q={q['x2/X2.3/q-align']} />
      {src ? (
        <Code
          tabs={[{ label: 'q4_0_ref.h', code: src, status: 'host C++; compiled and compared bit for bit with the model by a unit test; not a GPU kernel', highlight: [75, 76, 84, 85, 86, 87, 88, 102] }]}
          caption={
            <>
              The same CPU reference as in section X2.2, with the unpacking marked: the dequantiser's mask and shift, the
              word built from two 16-bit reads, and the block's dot product with the 8 folded out. Loaded from{' '}
              <code>labs/x2/snippets/q4_0_ref.h</code>.
            </>
          }
        />
      ) : null}

      <h2>One block, end to end</h2>
      <p>
        The animation follows the stepper's block of 32 weights from floats to codes, into bytes, through memory and back
        out of one 16-bit word, and checks the block's dot product the way the Metal kernels compute it.
      </p>
      <Simplification>
        <p>
          The scene draws addresses and values only. It runs the reference rule from llama.cpp's CPU code; the GPU
          kernels read the same bytes but are not simulated, and nothing is timed.
        </p>
      </Simplification>
      <PackedBlockScene />

      <h2>What each backend gives you for 4-bit data</h2>
      <p>
        On CUDA and HIP: bytes, words and your own masks. On Metal, the same, and since recently a little more. MSL
        defines packed 4-bit integer formats with conversion to floating point, and MPSGraph has a 4-bit data type with a
        blockwise dequantise. The labs do not use either, for the reasons in the question below.
      </p>
      <Today checked={CHECKED('C2, C4')}>
        <p>
          MSL: <code>int4b_format</code> and <code>uint4b_format</code> from Metal 4.0 with SDK 26.4,{' '}
          <code>unpack</code> to float, half or bfloat from Metal 4.1. Which nibble of a byte is element 0, and which GPU
          families support the types, are not documented; neither MLX nor llama.cpp uses them (a code search).
          MPSGraph: <code>MPSDataTypeInt4</code> and <code>dequantizeTensor</code> with blockwise float scales from
          macOS 15.
        </p>
      </Today>
      <Ask q={q['x2/X2.3/q-int4b']} />
      <p>
        One bridge for goal three. The bytes of a Q4_0 tensor in a GGUF file are exactly these 18-byte blocks, and
        llama.cpp's <code>gguf-py</code> package quantises with the same rule in NumPy, so it is a usable oracle when you
        write your own tests. Lab X2.2 (on the <a href={href('x2', 'labs')}>labs page</a>) reads these blocks on the GPU.
      </p>
    </>
  );
}
