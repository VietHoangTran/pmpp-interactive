import { useEffect, useRef, useState, type ReactNode } from 'react';

// A narrated animation: a sequence of beats, each a narration clip plus a drawing
// that is a pure function of how far through the beat we are. Picture and sound
// stay in sync because a beat lasts exactly as long as its clip.
//
// Scripts, audio and the manifest live in content/<chapter>/narration/ (see
// tools/narration.mjs). Captions are the scripts themselves. With no audio for a
// beat the animation still runs, on a duration estimated from the script length.
export interface Beat {
  /** Clip id: content/<chapter>/narration/<clip>.txt and .mp3 */
  clip: string;
  /** Short name shown on the beat's button. */
  label: string;
  /** Draws the beat; p runs from 0 to 1. */
  draw: (p: number) => ReactNode;
}

interface Manifest {
  model: string;
  voice: string;
  clips: Record<string, { file: string; sha256: string; duration: number | null }>;
}

const stripTags = (s: string) => s.replace(/\[[^\]]*\]\s*/g, '').trim();
const estimate = (script: string) => Math.max(4, stripTags(script).split(/\s+/).length / 2.6 + 1);

export function Scene({ chapter, title, beats }: { chapter: string; title: string; beats: Beat[] }) {
  const base = `/content/${chapter}/narration/`;
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scripts, setScripts] = useState<Record<string, string>>({});
  const [i, setI] = useState(0);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [sound, setSound] = useState(true);
  const audio = useRef<HTMLAudioElement | null>(null);
  const last = useRef<number | null>(null);

  useEffect(() => {
    fetch(base + 'manifest.json')
      .then((r) => (r.ok ? r.json() : null))
      .then(setManifest)
      .catch(() => setManifest(null));
    for (const b of beats)
      fetch(base + b.clip + '.txt')
        .then((r) => (r.ok ? r.text() : ''))
        .then((text) => setScripts((s) => ({ ...s, [b.clip]: text })))
        .catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chapter]);

  const beat = beats[i];
  const entry = manifest?.clips[beat.clip];
  const script = scripts[beat.clip] ?? '';
  const dur = entry?.duration ?? estimate(script);
  const hasAudio = !!entry;

  // Start or stop this beat's audio when playback state or the beat changes.
  useEffect(() => {
    const a = audio.current;
    if (!a) return;
    if (playing && hasAudio && sound) {
      if (!a.src.endsWith(base + entry!.file)) a.src = base + entry!.file;
      a.currentTime = Math.min(t, dur);
      a.play().catch(() => undefined);
    } else a.pause();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, i, hasAudio, sound]);

  // The clock. Audio, when present, is the master; otherwise wall time.
  useEffect(() => {
    if (!playing) {
      last.current = null;
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      const a = audio.current;
      const dt = last.current == null ? 0 : (now - last.current) / 1000;
      last.current = now;
      setT((prev) => {
        const next = hasAudio && sound && a && !a.paused && a.currentTime > 0 ? a.currentTime : prev + dt;
        return Math.min(next, dur);
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, hasAudio, sound, dur]);

  // Advance at the end of a beat.
  useEffect(() => {
    if (!playing || t < dur) return;
    const a = audio.current;
    if (hasAudio && sound && a && !a.ended && !a.paused) return; // let the clip finish
    if (i + 1 < beats.length) {
      setI(i + 1);
      setT(0);
    } else setPlaying(false);
  }, [t, dur, playing, i, beats.length, hasAudio, sound]);

  const go = (k: number) => {
    setI(Math.max(0, Math.min(beats.length - 1, k)));
    setT(0);
  };
  const p = dur > 0 ? Math.min(1, t / dur) : 0;
  const finished = !playing && i === beats.length - 1 && t >= dur;

  return (
    <section className="bench wide scene" aria-label={title}>
      <header>
        <h4>{title}</h4>
        <span className="task">
          {beats.length} parts, about {Math.round(beats.reduce((s, b) => s + (manifest?.clips[b.clip]?.duration ?? estimate(scripts[b.clip] ?? '')), 0))} seconds
        </span>
      </header>
      <div className="body">
        <div className="stage">{beat.draw(p)}</div>
        <p className="caption" aria-live="polite">{stripTags(script) || '…'}</p>
        <div className="row">
          <button className="primary" onClick={() => { if (finished) go(0); setPlaying(!playing); }}>
            {playing ? 'Pause' : finished ? 'Play again' : t > 0 || i > 0 ? 'Resume' : 'Play'}
          </button>
          <span className="seg" role="group" aria-label="Parts">
            {beats.map((b, k) => (
              <button key={b.clip} aria-pressed={k === i} onClick={() => go(k)} title={b.label}>
                {k + 1}. {b.label}
              </button>
            ))}
          </span>
          <label className="check">
            <input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} /> sound
          </label>
        </div>
        <input
          type="range" min="0" max="1000" value={Math.round(p * 1000)} aria-label="Position in this part"
          style={{ width: '100%', marginTop: '0.6rem' }}
          onChange={(e) => {
            const nt = (Number(e.target.value) / 1000) * dur;
            setT(nt);
            if (audio.current && hasAudio) audio.current.currentTime = nt;
          }}
        />
        <audio ref={audio} preload="auto" />
      </div>
      <footer>
        {manifest === null
          ? 'No narration manifest found; running silently on estimated timings.'
          : hasAudio
            ? `Narration generated with ${manifest.model}. Captions are the script.`
            : 'This part has no audio yet; it runs on an estimated timing. Captions are the script.'}
      </footer>
    </section>
  );
}
