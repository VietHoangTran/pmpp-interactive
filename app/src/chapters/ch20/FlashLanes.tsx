import { useEffect, useState } from 'react';
import { Bench } from '../../components/notes';
import { laneElements, ownerOf, wave64Broadcast } from '../../sim/flashplan';

// "Who holds Q element dd?": Fig. 20.10's interleaved layout and Fig. 20.12's broadcast,
// then the same code on a 64-wide AMD wavefront (facts B2). Model: src/sim/flashplan.ts,
// which takes HIP's shuffle rule from src/sim/shuffle.ts.

type Mode = 'nvidia' | 'wave64' | 'wave64w32';
const MODE_LABEL: Record<Mode, string> = {
  nvidia: '32-lane warp (the book)',
  wave64: 'AMD wave64, book’s code, default width',
  wave64w32: 'AMD wave64, width 32',
};
const D = 128;

export function FlashLanes() {
  const [dd, setDd] = useState(77);
  const [mode, setMode] = useState<Mode>('nvidia');
  const [pick, setPick] = useState(40);
  const [playing, setPlaying] = useState(false);
  const own = ownerOf(dd);

  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => setDd((x) => (x + 1) % D), 260);
    return () => window.clearTimeout(id);
  }, [playing, dd]);

  const wave = mode === 'nvidia' ? null : wave64Broadcast(dd, mode === 'wave64' ? 64 : 32);
  const picked = wave?.[pick];
  const wrong = wave ? wave.filter((x) => !x.ok).length : 0;

  return (
    <Bench
      title="Who holds Q element dd?"
      task="Before dragging: which lane and which register slot hold element 77?"
      foot={
        mode === 'nvidia'
          ? 'One warp, one row of Q with d = 128. Each lane holds four elements in four registers, slot 0 to slot 3.'
          : 'One 64-wide wavefront running the book’s code with WARP_SIZE 32: hardware lanes 0–31 are the first logical warp (one row of Q), 32–63 the second (another row). The width rule is HIP’s own source (ch20 facts B2); nothing was run on AMD hardware.'
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field" style={{ minWidth: '14rem' }}>
          Element dd = {dd}
          <input type="range" min={0} max={D - 1} value={dd} aria-label="Element dd" onChange={(e) => setDd(+e.target.value)} />
        </label>
        <button onClick={() => setDd((x) => (x + D - 1) % D)}>dd − 1</button>
        <button className="primary" onClick={() => setDd((x) => (x + 1) % D)}>dd + 1</button>
        <button onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play the inner loop'}</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Machine">
          {(Object.keys(MODE_LABEL) as Mode[]).map((m) => (
            <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>{MODE_LABEL[m]}</button>
          ))}
        </span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }} data-owner>
        Q element {dd} lives in <b className="t-reg">lane {own.lane}</b>, register slot <b className="t-reg">{own.slot}</b> ({dd} = {own.slot} · 32 + {own.lane}).
        {mode === 'nvidia' ? (
          <> At step dd of the inner product, every lane calls <code>__shfl_sync(mask, Q_i[ii][{own.slot}], {own.lane})</code> and
            receives it: a broadcast from lane {own.lane} to all 32.</>
        ) : (
          <> Every lane asks for source lane {own.lane}. {mode === 'wave64'
            ? <>With HIP's default <code>width = warpSize</code> = 64, all 64 hardware lanes read hardware lane {own.lane}: <b className="t-fail">the {wrong} lanes of the second logical warp get the other row's value.</b></>
            : <>With <code>width</code> 32, each half reads its own lane {own.lane}: hardware lanes 32–63 read lane {32 + own.lane}. All correct.</>}
          </>
        )}
      </div>

      {mode === 'nvidia' ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(3.3rem, 1fr))', gap: 4 }} data-lanes="32">
          {Array.from({ length: 32 }, (_, L) => {
            const els = laneElements(L, D);
            const isOwner = L === own.lane;
            return (
              <div key={L} data-lane={L} data-owner={isOwner ? 'yes' : 'no'}
                style={{ border: `1px solid ${isOwner ? 'var(--reg)' : 'var(--rule)'}`, background: isOwner ? 'var(--reg-wash)' : 'var(--page)', borderRadius: 3, padding: '2px 3px', fontSize: '0.7rem' }}>
                <div style={{ fontWeight: 600 }}>lane {L}</div>
                {els.map((e, s) => (
                  <div key={s} style={{ fontVariantNumeric: 'tabular-nums', background: e === dd ? 'var(--reg)' : undefined, color: e === dd ? 'var(--page)' : 'var(--ink-2)', borderRadius: 2, paddingLeft: 2 }}>
                    s{s}: {e}
                  </div>
                ))}
                <div style={{ color: 'var(--ink-3)' }}>← {own.lane}</div>
              </div>
            );
          })}
        </div>
      ) : (
        <>
          {[0, 1].map((half) => (
            <div key={half} style={{ marginBottom: '0.6rem' }}>
              <div className="mx-label">logical warp {half}: hardware lanes {half * 32}–{half * 32 + 31} ({half ? 'row B' : 'row A'} of Q)</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(2.6rem, 1fr))', gap: 3 }} data-half={half}>
                {wave!.slice(half * 32, half * 32 + 32).map((x) => (
                  <button key={x.hw} onClick={() => setPick(x.hw)} aria-pressed={pick === x.hw} data-hw={x.hw} data-ok={x.ok ? 'yes' : 'no'}
                    style={{ padding: '2px 0', fontSize: '0.68rem', fontWeight: 450, borderColor: x.ok ? 'var(--rule)' : 'var(--fail)', background: pick === x.hw ? 'var(--ink)' : x.ok ? 'var(--page)' : 'var(--fail-wash)', color: pick === x.hw ? 'var(--page)' : x.ok ? 'var(--ink)' : 'var(--fail)' }}
                    title={`hardware lane ${x.hw} wants ${x.wants}, gets ${x.gets}`}>
                    {x.hw}<br />←{x.gets}
                  </button>
                ))}
              </div>
            </div>
          ))}
          {picked ? (
            <p data-picked style={{ margin: '0.4rem 0 0' }}>
              Hardware lane {picked.hw} (lane {picked.hw % 32} of logical warp {picked.logicalWarp}) asks for source lane {own.lane} and
              receives hardware lane <b className={picked.ok ? 't-ok' : 't-fail'}>{picked.gets}</b>
              {picked.ok ? ', the right one.' : `, which belongs to the other logical warp; it wanted ${picked.wants}.`}
            </p>
          ) : null}
        </>
      )}
    </Bench>
  );
}
