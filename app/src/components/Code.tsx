import { useState, type ReactNode } from 'react';
import type { Tier } from './notes';

// A small C-family highlighter. Its one unusual job: identifiers are coloured by
// the memory they live in (register, shared, global), the same colours the
// diagrams use, so a kernel can be read as a map of where its data is.
const KEYWORDS = new Set(
  (
    'void int unsigned float double char bool const if else for while return struct using namespace ' +
    'sizeof static extern uint uint3 size_t dim3 kernel device constant threadgroup __global__ __device__ ' +
    '__host__ __shared__ __constant__ define ifndef endif include id uint32_t NSUInteger'
  ).split(' '),
);
const SYNC = new Set(['__syncthreads', 'threadgroup_barrier']);

export const MATMUL_TIERS: Record<string, Tier> = {
  M: 'global', N: 'global', P: 'global', A: 'global', B: 'global', C: 'global',
  Mds: 'shared', Nds: 'shared', As: 'shared', Bs: 'shared',
  Pvalue: 'reg', Cvalue: 'reg', tx: 'reg', ty: 'reg', bx: 'reg', by: 'reg', Row: 'reg', Col: 'reg',
  row: 'reg', col: 'reg', ph: 'reg', k: 'reg', i: 'reg', aCol: 'reg', bRow: 'reg',
};

function highlightLine(line: string, tiers: Record<string, Tier>, inBlock: { v: boolean }): ReactNode[] {
  const out: ReactNode[] = [];
  let i = 0;
  let key = 0;
  const push = (cls: string | null, text: string) =>
    out.push(cls ? <span key={key++} className={cls}>{text}</span> : text);
  while (i < line.length) {
    if (inBlock.v) {
      const end = line.indexOf('*/', i);
      const stop = end < 0 ? line.length : end + 2;
      push('c', line.slice(i, stop));
      inBlock.v = end < 0;
      i = stop;
      continue;
    }
    const rest = line.slice(i);
    if (rest.startsWith('//')) {
      push('c', rest);
      break;
    }
    if (rest.startsWith('/*')) {
      inBlock.v = true;
      continue;
    }
    let m = /^"(?:[^"\\]|\\.)*"/.exec(rest);
    if (m) {
      push('n', m[0]);
      i += m[0].length;
      continue;
    }
    m = /^\d[\d.]*f?/.exec(rest);
    if (m) {
      push('n', m[0]);
      i += m[0].length;
      continue;
    }
    m = /^[A-Za-z_]\w*/.exec(rest);
    if (m) {
      const w = m[0];
      if (SYNC.has(w)) push('v-sync', w);
      else if (tiers[w]) push('v-' + tiers[w], w);
      else if (KEYWORDS.has(w)) push('k', w);
      else push(null, w);
      i += w.length;
      continue;
    }
    push(null, line[i]);
    i += 1;
  }
  return out;
}

export interface CodeTab {
  label: string;
  code: string;
  /** Verification status, in the project's fixed vocabulary. */
  status?: string;
  tiers?: Record<string, Tier>;
  /** 1-based lines to mark. */
  highlight?: number[];
}

export function Code({
  tabs,
  caption,
  highlight,
  firstLine = 1,
}: {
  tabs: CodeTab[];
  caption?: ReactNode;
  highlight?: number[];
  firstLine?: number;
}) {
  const [sel, setSel] = useState(0);
  const tab = tabs[Math.min(sel, tabs.length - 1)];
  const hl = new Set(tab.highlight ?? highlight ?? []);
  const inBlock = { v: false };
  const lines = tab.code.replace(/\n$/, '').split('\n');
  return (
    <figure className="code wide" style={{ maxWidth: '58rem' }}>
      {tabs.length > 1 || tab.status ? (
        <div className="tabs" role="tablist">
          {tabs.map((t, i) => (
            <button key={t.label} role="tab" aria-selected={i === sel} onClick={() => setSel(i)}>
              {t.label}
            </button>
          ))}
          {tab.status ? <span className="status">{tab.status}</span> : null}
        </div>
      ) : null}
      <pre>
        <code>
          {lines.map((ln, i) => (
            <span key={i} className={'ln' + (hl.has(i + firstLine) ? ' hl' : '')} data-n={i + firstLine}>
              {highlightLine(ln, tab.tiers ?? {}, inBlock)}
              {'\n'}
            </span>
          ))}
        </code>
      </pre>
      {caption ? <figcaption className="caption">{caption}</figcaption> : null}
    </figure>
  );
}

/** Loads a lab source file from the repository so lessons always show the real, tested file. */
export function useLabFile(path: string): string | null {
  const [text, setText] = useState<string | null>(null);
  const [asked, setAsked] = useState<string | null>(null);
  if (asked !== path) {
    setAsked(path);
    setText(null);
    fetch('/api/lab-file?path=' + encodeURIComponent(path))
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then(setText)
      .catch(() => setText('// could not load labs/' + path));
  }
  return text;
}
