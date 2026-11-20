import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SCENE, kahanTrace, sceneValues, units25, type KahanStep } from '../../sim/fpsum';

// "Kahan's correction" (section A.6): an FP32 accumulator holding 1.0 receives 4,096 values
// of 2^-25, a quarter of the gap above 1. Every number drawn comes from fpsum.kahanTrace on
// sceneValues() (FP32, round to nearest even). Six beats, one narration clip each
// (content/appA/narration/kahan-N.txt).
const TRACE: KahanStep[] = kahanTrace(sceneValues(), 'fp32');
const step = (k: number) => TRACE[k]; // k = 1 is the first small value (TRACE[0] adds the 1.0)
const LAST = TRACE[TRACE.length - 1];
const ULP_UNITS = 4; // one FP32 ULP at 1 is 2^-23 = 4 units of 2^-25

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

/** A value near 1 as "1 + k × 2⁻²⁵", or a small one as "k × 2⁻²⁵". */
function u25(v: number, near1: boolean): string {
  const k = Math.round(units25(near1 ? v - 1 : v));
  const tail = (a: number) => (a === 1 ? '2⁻²⁵' : `${a} × 2⁻²⁵`);
  if (near1) return k === 0 ? '1' : `1 ${k > 0 ? '+' : '−'} ${tail(Math.abs(k))}`;
  return k === 0 ? '0' : `${k < 0 ? '−' : ''}${tail(Math.abs(k))}`;
}

// ---------------------------------------------------------------- the number line near 1
const LX = 60;
const LW = 600;
const LY = 92;
const UMIN = -1;
const UMAX = 11; // units of 2^-25 above 1
const xOf = (units: number) => LX + ((units - UMIN) / (UMAX - UMIN)) * LW;

function Line({ exact, plain, kahan, candidate }: { exact?: number; plain?: number; kahan?: number; candidate?: number }) {
  const ticks: ReactNode[] = [];
  for (let k = UMIN; k <= UMAX; k++) {
    const isFloat = k % ULP_UNITS === 0;
    ticks.push(
      <line key={k} x1={xOf(k)} x2={xOf(k)} y1={LY - (isFloat ? 9 : 4)} y2={LY + (isFloat ? 9 : 4)} stroke={isFloat ? 'var(--ink)' : 'var(--rule)'} strokeWidth={isFloat ? 2 : 1} />,
    );
  }
  const mark = (units: number) => xOf(Math.max(UMIN, Math.min(UMAX, units)));
  return (
    <g data-part="line">
      <text x={LX} y={LY - 52} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
        floats near 1: heavy ticks, one ULP (2⁻²³) apart; light ticks every 2⁻²⁵
      </text>
      <line x1={LX} x2={LX + LW} y1={LY} y2={LY} stroke="var(--rule)" />
      {ticks}
      <g fontSize="11" style={{ fill: 'var(--ink-2)' }}>
        <rect x={LX + 330} y={LY - 33} width="9" height="9" fill="var(--ink-3)" />
        <text x={LX + 344} y={LY - 25}>plain sum</text>
        <circle cx={LX + 420} cy={LY - 28.5} r="5" fill="var(--ink)" />
        <text x={LX + 430} y={LY - 25}>Kahan sum</text>
        <line x1={LX + 508} x2={LX + 508} y1={LY - 36} y2={LY - 22} stroke="var(--ink)" strokeDasharray="3 3" />
        <text x={LX + 514} y={LY - 25}>exact</text>
      </g>
      {[0, 4, 8].map((k) => (
        <text key={k} x={xOf(k)} y={LY + 26} fontSize="12" textAnchor="middle">
          {k === 0 ? '1' : `1 + ${k / ULP_UNITS} ULP`}
        </text>
      ))}
      {candidate != null ? (
        <g data-part="candidate" data-units={candidate}>
          <circle cx={mark(candidate)} cy={LY} r="6" fill="none" stroke="var(--ink-2)" strokeDasharray="2 2" />
          <text x={mark(candidate)} y={LY - 14} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>sum + y</text>
        </g>
      ) : null}
      {exact != null ? (
        <g data-part="exact" data-units={exact}>
          <line x1={mark(exact)} x2={mark(exact)} y1={LY - 22} y2={LY + 12} stroke="var(--ink)" strokeDasharray="3 3" />
          <text x={mark(exact)} y={LY + 46} fontSize="11.5" textAnchor="middle">exact</text>
        </g>
      ) : null}
      {plain != null ? (
        <g data-part="plain" data-units={plain}>
          <rect x={mark(plain) - 6} y={LY - 6} width="12" height="12" fill="var(--ink-3)" />
        </g>
      ) : null}
      {kahan != null ? (
        <g data-part="kahan" data-units={kahan}>
          <circle cx={mark(kahan)} cy={LY} r="6.5" fill="var(--ink)" />
        </g>
      ) : null}
    </g>
  );
}

// ---------------------------------------------------------------- the registers
interface Regs {
  x?: string;
  sum?: string;
  c?: string;
  y?: string;
  t?: string;
  hot?: keyof Regs;
}
function Registers({ r, cOpacity = 1 }: { r: Regs; cOpacity?: number }) {
  const box = (name: keyof Regs, label: string, x: number, opacity = 1) => {
    const v = r[name];
    if (v === undefined) return null;
    const hot = r.hot === name;
    return (
      <g key={name} data-part={'reg-' + name} data-value={v} opacity={opacity}>
        <text x={x} y={176} fontSize="12" fontWeight="600" style={{ fill: 'var(--reg)' }}>{label}</text>
        <rect x={x} y={182} width={124} height={34} rx="3" fill={hot ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" />
        <text x={x + 62} y={204} fontSize="12.5" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
      </g>
    );
  };
  return (
    <g>
      {box('x', 'x (next value)', 24)}
      {box('sum', 'sum', 160)}
      {box('c', 'c (lost part, negated)', 296, cOpacity)}
      {box('y', 'y = x − c', 432)}
      {box('t', 't = sum + y', 568)}
    </g>
  );
}

function Headline({ children }: { children: ReactNode }) {
  return (
    <text data-part="headline" x={24} y={260} fontSize="13.5" fontWeight="600">
      {children}
    </text>
  );
}
function Sub({ children, y = 282 }: { children: ReactNode; y?: number }) {
  return (
    <text x={24} y={y} fontSize="12.5" style={{ fill: 'var(--ink-2)' }}>
      {children}
    </text>
  );
}

function Stage({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 720 320" role="img" aria-label="Animation of Kahan's compensated summation" style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      {children}
    </svg>
  );
}

// ---------------------------------------------------------------- the plot of beat 5
const PX = 70;
const PY = 22;
const PW = 600;
const PH = 180;
const N = SCENE.n;
const TOP = units25(LAST.exact - 1) / ULP_UNITS; // the exact total, in ULPs above 1
const SAMPLE = Array.from({ length: 129 }, (_, j) => Math.round((j / 128) * N));
function Plot({ upto }: { upto: number }) {
  const px = (i: number) => PX + (i / N) * PW;
  const py = (ulps: number) => PY + PH - (ulps / TOP) * PH;
  const pts = (pick: (s: KahanStep) => number) =>
    SAMPLE.filter((i) => i <= upto).map((i) => `${px(i)},${py(units25(pick(TRACE[i]) - 1) / ULP_UNITS)}`).join(' ');
  const at = TRACE[Math.max(0, Math.min(N, upto))];
  return (
    <g data-part="plot" data-upto={upto} data-kahan={units25(at.sum - 1)} data-plain={units25(at.plain - 1)} data-exact={units25(at.exact - 1)}>
      <line x1={PX} x2={PX} y1={PY} y2={PY + PH} stroke="var(--rule)" />
      <line x1={PX} x2={PX + PW} y1={PY + PH} y2={PY + PH} stroke="var(--rule)" />
      <text x={PX - 8} y={PY + 4} fontSize="11" textAnchor="end">{TOP}</text>
      <text x={PX - 8} y={PY + PH} fontSize="11" textAnchor="end">0</text>
      <text x={PX - 8} y={PY + PH / 2} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>ULPs</text>
      <text x={PX - 8} y={PY + PH / 2 + 13} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>above 1</text>
      <text x={PX} y={PY + PH + 16} fontSize="11">0</text>
      <text x={PX + PW} y={PY + PH + 16} fontSize="11" textAnchor="end">{N} values</text>
      <polyline points={pts((s) => s.exact)} fill="none" stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="5 4" />
      <polyline points={pts((s) => s.sum)} fill="none" stroke="var(--ink)" strokeWidth="2.5" />
      <polyline points={pts((s) => s.plain)} fill="none" stroke="var(--ink-3)" strokeWidth="3" />
      <g fontSize="11.5">
        <line x1={PX + 12} x2={PX + 36} y1={PY + 8} y2={PY + 8} stroke="var(--ink)" strokeDasharray="5 4" />
        <text x={PX + 42} y={PY + 12}>exact</text>
        <line x1={PX + 92} x2={PX + 116} y1={PY + 8} y2={PY + 8} stroke="var(--ink)" strokeWidth="2.5" />
        <text x={PX + 122} y={PY + 12}>Kahan</text>
        <line x1={PX + 176} x2={PX + 200} y1={PY + 8} y2={PY + 8} stroke="var(--ink-3)" strokeWidth="3" />
        <text x={PX + 206} y={PY + 12}>plain</text>
      </g>
    </g>
  );
}

// ---------------------------------------------------------------- beats
const s1 = step(1);
const s2 = step(2);
const s3 = step(3);

const beats: Beat[] = [
  {
    clip: 'kahan-1',
    label: 'The plain sum stalls',
    draw: (p) => {
      const k = Math.min(8, Math.floor(p * 9));
      const at = TRACE[k];
      return (
        <Stage>
          <Line exact={units25(at.exact - 1)} plain={units25(at.plain - 1)} />
          <Registers r={{ x: k > 0 ? u25(SCENE.x, false) : '—', sum: u25(at.plain, true) }} />
          <Headline>
            {k === 0 ? 'sum = 1; values of 2⁻²⁵ are about to arrive' : `after ${k} value${k > 1 ? 's' : ''}: plain sum ${u25(at.plain, true)}, exact ${u25(at.exact, true)}`}
          </Headline>
          <Sub>each value is a quarter of the gap above 1, so 1 + 2⁻²⁵ rounds back to 1 every time</Sub>
        </Stage>
      );
    },
  },
  {
    clip: 'kahan-2',
    label: 'A second register',
    draw: (p) => {
      const showY = p >= 0.35;
      const showT = p >= 0.65;
      return (
        <Stage>
          <Line exact={1} kahan={units25(showT ? s1.t - 1 : 0)} candidate={showT ? units25(s1.sum - 1 + s1.y) : undefined} />
          <Registers
            r={{ x: u25(s1.x, false), sum: '1', c: '0', y: showY ? u25(s1.y, false) : undefined, t: showT ? u25(s1.t, true) : undefined, hot: showT ? 't' : showY ? 'y' : 'c' }}
            cOpacity={ramp(p, 0, 0.2)}
          />
          <Headline>{showT ? `t = sum + y = 1 + 2⁻²⁵, which rounds to ${u25(s1.t, true)}` : showY ? `y = x − c = ${u25(s1.y, false)}` : 'c starts at 0'}</Headline>
          <Sub>value 1 of {N}: nothing gained yet</Sub>
        </Stage>
      );
    },
  },
  {
    clip: 'kahan-3',
    label: 'Catch what was lost',
    draw: (p) => {
      const half = p >= 0.4;
      const done = p >= 0.7;
      const move = ramp(p, 0.7, 0.95);
      // the dropped part travels from the exact point on the line into c
      const fromX = xOf(1);
      const fromY = LY;
      const toX = 296 + 62;
      const toY = 199;
      return (
        <Stage>
          <Line exact={1} kahan={0} />
          <Registers r={{ x: u25(s1.x, false), sum: '1', c: done ? u25(s1.c, false) : '0', y: u25(s1.y, false), t: u25(s1.t, true), hot: done ? 'c' : undefined }} />
          {move > 0 && move < 1 ? (
            <circle data-part="dropped" cx={fromX + (toX - fromX) * move} cy={fromY + (toY - fromY) * move} r="5" fill="var(--reg)" />
          ) : null}
          <Headline>
            {done ? `c = (t − sum) − y = ${u25(s1.c, false)}: the lost part, negated` : half ? 't − sum = 0: what the addition really added' : 'c = (t − sum) − y'}
          </Headline>
          <Sub>{done ? 'nothing is lost from the record: it waits in c for the next addition' : 'in exact arithmetic this would be 0'}</Sub>
        </Stage>
      );
    },
  },
  {
    clip: 'kahan-4',
    label: 'Put it back',
    draw: (p) => {
      const st = p < 0.5 ? s2 : s3;
      const q = p < 0.5 ? p / 0.5 : (p - 0.5) / 0.5;
      const prev = p < 0.5 ? s1 : s2;
      const showT = q >= 0.35;
      const showC = q >= 0.7;
      return (
        <Stage>
          <Line exact={units25(st.exact - 1)} plain={0} kahan={units25((showT ? st.t : prev.sum) - 1)} candidate={showT ? units25(prev.sum - 1 + st.y) : undefined} />
          <Registers
            r={{
              x: u25(st.x, false),
              sum: u25(showT ? st.sum : prev.sum, true),
              c: u25(showC ? st.c : prev.c, false),
              y: u25(st.y, false),
              t: showT ? u25(st.t, true) : undefined,
              hot: showC ? 'c' : showT ? 't' : 'y',
            }}
          />
          <Headline>
            {`value ${st.i - 1}: y = x − c = ${u25(st.y, false)}` +
              (showT ? (st.t === prev.sum ? ', t ties and rounds back to 1' : `, t moves up one ULP to ${u25(st.t, true)}`) : '') +
              (showC ? `; c = ${u25(st.c, false)}` : '')}
          </Headline>
          <Sub>{showC && st.c > 0 ? 'c > 0: the step overshot by 2⁻²⁵, and the next value will be reduced by it' : 'the lost part rides along in c until it is big enough to count'}</Sub>
        </Stage>
      );
    },
  },
  {
    clip: 'kahan-5',
    label: 'All 4,096 values',
    draw: (p) => {
      const upto = Math.round(ramp(p, 0.05, 0.8) * N);
      const end = p >= 0.82;
      const plainUlps = units25(LAST.exact - LAST.plain) / ULP_UNITS;
      const kahanUlps = units25(Math.abs(LAST.exact - LAST.sum)) / ULP_UNITS;
      return (
        <Stage>
          <Plot upto={upto} />
          <Headline>
            {end ? `exact 1 + 2⁻¹³; plain sum ${plainUlps} ULPs short; Kahan ${kahanUlps} ULPs off` : `after ${upto} values`}
          </Headline>
          <Sub>{end ? 'the compensated sum ends on the exact total; the plain sum never left 1' : 'Kahan climbs about one ULP every four values; the plain sum stays at 1'}</Sub>
        </Stage>
      );
    },
  },
  {
    clip: 'kahan-6',
    label: 'Keep the compiler out',
    draw: (p) => {
      const strike = ramp(p, 0.1, 0.3);
      const rows = ramp(p, 0.4, 0.9);
      const lines = ['y = x − c;', 't = sum + y;', 'c = (t − sum) − y;', 'sum = t;'];
      return (
        <Stage>
          <g data-part="code" fontSize="14" style={{ fontVariationSettings: "'MONO' 1" }}>
            {lines.map((l, i) => (
              <text key={l} x={40} y={40 + i * 24} style={{ fill: i === 2 && strike > 0.5 ? 'var(--fail)' : 'var(--ink)' }}>{l}</text>
            ))}
            <line x1={36} x2={36 + 170 * strike} y1={84} y2={84} stroke="var(--fail)" strokeWidth="2" />
            {strike >= 1 ? <text x={230} y={88} fontSize="13" style={{ fill: 'var(--fail)' }}>= 0 if + were associative: the plain sum again</text> : null}
          </g>
          <g data-part="backends" opacity={rows} fontSize="12.5">
            <text x={40} y={150} fontWeight="600">How each backend keeps c</text>
            <text x={40} y={174}>CUDA: __fadd_rn / __fsub_rn, which are never fused; whether nvcc regroups is not documented</text>
            <text x={40} y={196}>HIP: Clang's default is no reassociation; -ffast-math turns it on</text>
            <text x={40} y={218}>Metal: the default modes allow it; #pragma METAL fp math_mode(safe) forbids it</text>
          </g>
          <Headline>{p >= 0.9 ? 'lab A.1 measures whether your device kept the compensation' : 'in exact arithmetic c is always 0'}</Headline>
        </Stage>
      );
    },
  },
];

export function KahanScene() {
  return <Scene chapter="appA" title="Kahan's correction" beats={beats} />;
}
