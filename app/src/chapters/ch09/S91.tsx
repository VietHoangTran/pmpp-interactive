import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import {
  PIXELS,
  PRESETS,
  SIDE,
  WIDTHS,
  binCount,
  binOf,
  binRange,
  flatIndex,
  histogram,
  hottest,
  preset,
  type PresetId,
} from '../../sim/histogram';

const SEQ_TIERS: Record<string, Tier> = { image: 'global', bins: 'global', i: 'reg', b: 'reg' };
const grey = (v: number) => `rgb(${v},${v},${v})`;
const range = (b: number, w: number) => {
  const [lo, hi] = binRange(b, w);
  return lo === hi ? String(lo) : `${lo}–${hi}`;
};

// ---------------------------------------------------------------- the explorer
function BarChart({ counts, width, mark }: { counts: number[]; width: number; mark: number }) {
  const W = 540;
  const H = 230;
  const pad = { l: 34, r: 8, t: 10, b: 40 };
  const n = counts.length;
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const bw = pw / n;
  const Y = (c: number) => pad.t + ph - (c / PIXELS) * ph;
  const ticks = [0, 16, 32, 48, 64];
  const labels = n <= 16 ? counts.map((_, b) => b) : [0, 64, 128, 192, 255];
  const busiest = counts.indexOf(Math.max(...counts));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Histogram of the image" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(t)} y2={Y(t)} stroke="var(--rule-soft)" />
          <text x={pad.l - 5} y={Y(t) + 4} fontSize="10.5" textAnchor="end">{t}</text>
        </g>
      ))}
      {counts.map((c, b) => (
        <rect
          key={b}
          data-bin={b}
          data-count={c}
          x={pad.l + b * bw + (n <= 16 ? 2 : 0)}
          y={Y(c)}
          width={Math.max(2.5, bw - (n <= 16 ? 4 : 0))}
          height={Y(0) - Y(c)}
          fill={b === mark ? 'var(--ink)' : 'var(--ink-3)'}
        />
      ))}
      {counts.map((c, b) =>
        c > 0 && (n <= 16 || b === mark || b === busiest) ? (
          <text key={'c' + b} x={Math.min(pad.l + (b + 0.5) * bw, W - pad.r - 4)} y={Y(c) - 4} fontSize="10.5"
            textAnchor={n <= 16 ? 'middle' : 'end'} fontWeight={b === mark ? 650 : 400}>
            {n <= 16 ? c : `${b}: ${c}`}
          </text>
        ) : null,
      )}
      {labels.map((b) => (
        <text
          key={'l' + b}
          x={pad.l + (b + 0.5) * bw}
          y={H - pad.b + 14}
          fontSize={n <= 8 ? 10.5 : 9}
          textAnchor="middle"
        >
          {n <= 8 ? range(b, width) : n <= 16 ? binRange(b, width)[0] : b}
        </text>
      ))}
      <text x={pad.l + pw / 2} y={H - 6} fontSize="11" textAnchor="middle">
        {n === 256 ? 'pixel value (one bin per value)' : `bin (interval of ${width} values)`}
      </text>
    </svg>
  );
}

function FlatStrip({ img, focus }: { img: number[]; focus: number }) {
  const C = 8;
  const lx = Math.min(Math.max(focus * C + C / 2, 30), PIXELS * C - 30);
  return (
    <svg viewBox={`0 0 ${PIXELS * C} 46`} role="img" aria-label="The image as one flat array" style={{ width: '100%', display: 'block' }}>
      {img.map((v, i) => (
        <rect key={i} x={i * C + 0.5} y={2} width={C - 1} height={20} fill={grey(v)} stroke={i === focus ? 'var(--ink)' : 'var(--rule)'} strokeWidth={i === focus ? 2 : 0.6} />
      ))}
      {Array.from({ length: SIDE }, (_, r) => (
        <line key={r} x1={r * SIDE * C} x2={r * SIDE * C} y1={0} y2={26} stroke="var(--ink-2)" strokeWidth={0.8} />
      ))}
      <path d={`M ${focus * C + C / 2} 24 l -4 6 h 8 z`} fill="var(--ink)" />
      <text x={lx} y={43} fontSize="13" textAnchor="middle" data-strip-index={focus}>index {focus}</text>
    </svg>
  );
}

function PaintAndBin() {
  const [pid, setPid] = useState<PresetId>('sky');
  const [img, setImg] = useState<number[]>(() => preset('sky').values.slice());
  const [width, setWidth] = useState(64);
  const [paint, setPaint] = useState(255);
  const [focus, setFocus] = useState(0);
  const [down, setDown] = useState(false);
  const counts = useMemo(() => histogram(img, width), [img, width]);
  const hot = hottest(counts);
  const sum = counts.reduce((s, c) => s + c, 0);
  const r = Math.floor(focus / SIDE);
  const c = focus % SIDE;
  const v = img[focus];
  const b = binOf(v, width);

  useEffect(() => {
    const up = () => setDown(false);
    window.addEventListener('pointerup', up);
    return () => window.removeEventListener('pointerup', up);
  }, []);

  const set = (i: number) => setImg((im) => (im[i] === paint ? im : im.map((x, k) => (k === i ? paint : x))));
  const choose = (id: PresetId) => {
    setPid(id);
    setImg(preset(id).values.slice());
  };

  return (
    <Bench
      title="Paint and bin"
      task="Paint some sky, then switch to one bin per value: which bar grows, and how many threads would collide on it?"
      foot={
        <>
          Our own 8 × 8 images, not the book's. Click or drag on the image to paint with the chosen value; point at a
          pixel to see where it lands in the flat array and which bin it updates. The bar chart's scale is fixed at 64,
          the whole image.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Image
          <span className="seg" role="group" aria-label="Image">
            {PRESETS.map((p) => (
              <button key={p.id} aria-pressed={pid === p.id} onClick={() => choose(p.id)}>{p.label}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Interval width
          <span className="seg" role="group" aria-label="Interval width">
            {WIDTHS.map((w) => (
              <button key={w} aria-pressed={width === w} onClick={() => setWidth(w)}>
                {w} ({binCount(w)} bins)
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          <span>
            Paint value = <span data-paint>{paint}</span>
          </span>
          <span className="row" style={{ gap: '0.5rem' }}>
            <input type="range" min="0" max="255" value={paint} aria-label="Paint value" onChange={(e) => setPaint(+e.target.value)} />
            <i style={{ display: 'inline-block', width: '1.4rem', height: '1.4rem', background: grey(paint), border: '1px solid var(--rule)', borderRadius: 2 }} />
          </span>
        </label>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div
            role="grid"
            aria-label="The image"
            data-part="image"
            style={{ display: 'grid', gridTemplateColumns: `repeat(${SIDE}, 1fr)`, gap: 2, maxWidth: 300, touchAction: 'none' }}
          >
            {img.map((val, i) => (
              <button
                key={i}
                data-index={i}
                aria-label={`row ${Math.floor(i / SIDE)}, column ${i % SIDE}, value ${val}`}
                onPointerDown={(e) => {
                  e.preventDefault();
                  setDown(true);
                  setFocus(i);
                  set(i);
                }}
                onPointerEnter={() => {
                  setFocus(i);
                  if (down) set(i);
                }}
                onFocus={() => setFocus(i)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    set(i);
                  }
                }}
                style={{
                  aspectRatio: '1',
                  padding: 0,
                  minWidth: 0,
                  background: grey(val),
                  border: i === focus ? '2px solid var(--ink)' : '1px solid var(--rule)',
                  borderRadius: 2,
                }}
              />
            ))}
          </div>
          <p className="muted" style={{ margin: '0.8rem 0 0.2rem' }}>The same 64 bytes as the kernel sees them, row after row:</p>
          <FlatStrip img={img} focus={focus} />
        </div>
        <div>
          <div className="narrate" role="status" data-part="readout" style={{ marginBottom: '0.8rem' }}>
            Pixel at row {r}, column {c}: flat index {r} × {SIDE} + {c} = <b data-flat>{flatIndex(r, c, SIDE)}</b>. Value{' '}
            <b>{v}</b>, so it updates bin <b data-bin-of>{b}</b> = ⌊{v} / {width}⌋
            {width > 1 ? <>, which holds the values {range(b, width)}</> : null}.
          </div>
          <BarChart counts={counts} width={width} mark={b} />
          <p style={{ margin: '0.6rem 0 0' }} data-part="hot" data-hot-count={hot.count}>
            The bars add up to <b data-sum>{sum}</b>, one count per pixel. Busiest bin:{' '}
            <b>{width === 1 ? `value ${hot.bin}` : `${range(hot.bin, width)}`}</b>, with {hot.count} pixels. With one thread
            per pixel, <b>{hot.count} threads</b> would update that one bin.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS91: Record<string, Question> = {
  'ch09/9.1/q-owner': {
    kind: 'choice',
    id: 'ch09/9.1/q-owner',
    prompt: <>Which of these breaks the owner-computes rule?</>,
    options: [
      { label: 'Colour to grayscale, one thread per pixel.', why: 'Each thread writes its own output pixel and nothing else.' },
      { label: 'Matrix multiplication, one thread per output element.', why: 'Many threads read the same inputs, but each output has one writer.' },
      { label: 'For every token id in a batch, add 1 to that token\'s count.', correct: true },
      {
        label: 'One block normalising one row of a softmax.',
        why: 'The block shares a row sum, but each output element is still written by exactly one thread. Computing the shared sum is a reduction, chapter 10.',
      },
    ],
    explain: (
      <p>
        In the counting kernel, which element a thread writes depends on the value it read: two threads that read the
        same token id write the same count. Nothing in the launch configuration tells you which threads will collide;
        only the data does. That is the defining property of the histogram pattern.
      </p>
    ),
    source: 'ch. 9, p. 201',
  },
  'ch09/9.1/q-contention': {
    kind: 'numeric',
    id: 'ch09/9.1/q-contention',
    prompt: (
      <>
        An image has 1,000,000 pixels, and 40% of them are pure white sky. With one thread per pixel and one bin per
        value, how many threads update bin 255?
      </>
    ),
    answer: 400000,
    unit: 'threads',
    explain: (
      <p>
        400,000: the height of a bar is the number of threads that contend for that bin. On the CPU it is only a count.
        On the GPU it is a queue, and section 9.3 turns it into time.
      </p>
    ),
    source: 'our arithmetic on the §9.1 setup (pp. 202–203)',
  },
  'ch09/9.1/q-cpu': {
    kind: 'choice',
    id: 'ch09/9.1/q-cpu',
    prompt: <>Why is the sequential loop limited by memory on a CPU, rather than by its increments?</>,
    options: [
      { label: 'Each pixel needs an integer division by the interval width, and integer division is slow on every CPU.', why: 'The listing uses one bin per value and divides by nothing. Even with a division, it is a few cycles of arithmetic.' },
      { label: 'The loop branch mispredicts often, because the pixel values it reads are unpredictable.', why: 'The loop has one branch, the loop test, and it does not depend on the pixel values.' },
      { label: 'The bins sit in L1, so increments are cheap; streaming the image in from DRAM sets the pace.', correct: true },
      { label: 'The increments miss in the cache, because neighbouring pixels usually update bins that are far apart.', why: '256 counters of 4 bytes fit easily in a level-one cache.' },
    ],
    explain: (
      <p>
        The book's argument: the image is read once, sequentially, so every cache line fetched is fully used; the bins
        are small enough to sit in the CPU's level-one data cache. What is left is the rate at which DRAM can deliver
        the image.
      </p>
    ),
    source: 'ch. 9 §9.1, p. 203',
  },
};

export function S91() {
  const seq = useLabFile('ch09/snippets/histogram_sequential.cpp');
  return (
    <>
      <p>
        A histogram counts how many values in a data set fall in each interval and draws one bar per interval, its
        height the count. For an 8-bit grayscale image the values are pixel intensities from 0 (black) to 255 (white).
        Cut that range into four intervals of 64 and you get four bars: how much of the picture is near black, dark
        grey, light grey and near white.
      </p>
      <Source section="9.1" pages="202" />
      <p>
        The shape of the bars says something about the data as a whole, which is why the book calls it a feature of the
        data set: a histogram crowded at the right is an overexposed photo. The book lists uses from credit-card fraud
        detection and computer vision to speech recognition, purchase recommendations and astronomy. Wherever a lot of
        data has to be summarised, a histogram is likely to be one of the first computations.
      </p>
      <Source section="9.1" pages="202–203" />

      <h2>Paint and bin</h2>
      <p>
        The explorer below starts with a picture of a bright sky over dark ground. Paint on it, change the interval
        width, and watch two things: the number under each pixel's bin, and the busiest bar.
      </p>
      <Simplification>
        <p>
          8-bit grayscale only, and 64 pixels so that every pixel can be seen and painted. A real image has millions,
          which only makes the bars taller.
        </p>
      </Simplification>
      <PaintAndBin />

      <h2>One division, one flat array</h2>
      <p>
        Binning is one integer division: a pixel of value <i>v</i> goes in bin ⌊<i>v</i> / <i>w</i>⌋ for intervals of
        width <i>w</i>. The book's code takes the simplest case, one bin per value (<i>w</i> = 1, 256 bins), so the pixel
        value is itself the index into the bins array. The position of a pixel never matters, only its value, so the
        code treats the image as one flat array of <code>width*height</code> bytes, laid out row after row as in{' '}
        <a href={href('ch03', '3.2')}>chapter 3</a>.
      </p>
      <Source section="9.1" pages="203" />
      <p>The whole sequential algorithm is one loop:</p>
      {seq ? (
        <Code
          tabs={[{ label: 'histogram_sequential.cpp', code: seq, tiers: SEQ_TIERS }]}
          highlight={[6, 7]}
          caption={
            <>
              Fig. 9.2, loaded from <code>labs/ch09/snippets/histogram_sequential.cpp</code>. It differs from the page
              only in <code>const</code> on the image. A unit test compiles it and checks that it counts every image in the
              explorer exactly as the explorer does.
            </>
          }
        />
      ) : null}
      <p>
        Read it with the bins in mind: the function never sets them to zero. Like the book's listing, it assumes the
        caller has done that. Every GPU version in this chapter makes the same assumption, and the labs' host code
        clears the bins before each launch.
      </p>
      <Ask q={questionsS91['ch09/9.1/q-cpu']} label="Predict" />
      <p>
        On a CPU this loop is limited by memory, not by its arithmetic. It reads the image once, from start to end, so
        every cache line fetched from DRAM is used in full. The bins array, 256 counters of 4 bytes, sits in the
        level-one data cache, so each increment is cheap. What remains is how fast DRAM can deliver the image.
      </p>
      <Source section="9.1" pages="203" />

      <h2>Each bar is a queue</h2>
      <p>
        Now picture the obvious parallel version: one thread per pixel, each adding one to its pixel's bin. Then the
        height of each bar is exactly the number of threads that will try to update that bin. On a uniform image the
        threads spread over all the bins. On a real one they do not: in the sky picture, almost half of the threads
        want the same bin. Painting a large patch of one value, or choosing the flat image, makes one bar hold every
        thread there is.
      </p>
      <Beyond>
        <p>
          Reading bar heights as collision counts is our framing; the book makes the same point about skewed images on
          p. 210.
        </p>
      </Beyond>
      <Ask q={questionsS91['ch09/9.1/q-contention']} label="Work it out" />
      <Ask q={questionsS91['ch09/9.1/q-owner']} />

      <h2>Under torch.bincount</h2>
      <p>
        You have called this computation from Python. <code>torch.bincount</code> counts non-negative integers, and its
        output has one more bin than the largest value it saw; <code>torch.histc</code> is its sibling for a given
        number of bins over a range. Both run on the GPU. Feed <code>bincount</code> token ids and the bins are the
        vocabulary: from about fifty thousand to two hundred thousand of them in current models, far too many for the
        on-chip tricks that make a 256-bin image histogram fast. That limit
        comes back in section 9.4 and in <a href={href('ch09', 'labs')}>lab 9.3</a>.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md, C6, D1, D5, D7)">
        <p>
          Apple's library has an image histogram, <code>MPSImageHistogram</code>: bin counts must be a power of two,
          counts are 32-bit, and it works on textures. It does not serve a buffer of token ids.
        </p>
      </Today>
    </>
  );
}
