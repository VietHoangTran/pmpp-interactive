// Speculative decoding for chapter 20, section 20.6 (book pp. 507–508, Fig. 20.18).
//
// The book's account: a small draft model proposes k tokens (the speculation depth); the
// large target model checks them in one pass, "executed in parallel" as a batch whose
// entries share one prefix, so the cached K and V of that prefix are loaded once for all
// k query rows; the tokens up to the first mismatch are accepted and the next round starts
// after them. Fig. 20.18 is replayed exactly by replayFigure().
//
// The rule the figure draws: in round 1 the target's own third token ("Parallel") is not
// kept; the accept band holds only the two accepted draft tokens and the next draft round
// proposes "Parallel" itself. figureRule follows that rule. keepTarget is ours: the verify
// pass has already computed the target's token at the first mismatch (the figure shows it),
// so keeping it costs nothing and every target pass yields at least one token. Neither
// models the draft's cost, and acceptance is drawn independently per token with
// probability alpha: a stand-in, not how real draft and target models agree.
//
// Per round with depth k and acceptance alpha, the expected tokens per target pass:
//   figure rule:  sum_{i=1..k} alpha^i       = alpha (1 - alpha^k) / (1 - alpha)
//   keep target:  sum_{i=0..k-1} alpha^i     = (1 - alpha^k) / (1 - alpha)
// (both k when alpha = 1). Our derivation, from the independence assumption.
//
// Erasable TypeScript only and no imports: the browser checks import this file directly.

// ------------------------------------------------------------------ the book's figure

export const FIG_PROMPT = ['The', 'title', 'of', 'this', 'book', 'is'];

export interface FigRound {
  /** Tokens before this round's drafts. */
  prefix: string[];
  /** The draft model's proposals, in order. */
  draft: string[];
  /** What the target model produces at each verify row (row i sees prefix + draft[0..i-1]). */
  target: string[];
  /** Whether each draft token matches the target's. */
  ok: boolean[];
  /** The tokens the round keeps (the figure's rule: leading matches only). */
  accepted: string[];
}

/** The two rounds of Fig. 20.18, as printed. */
export function replayFigure(): FigRound[] {
  const round = (prefix: string[], draft: string[], target: string[]): FigRound => {
    const ok = draft.map((t, i) => t === target[i]);
    const first = ok.indexOf(false);
    return { prefix, draft, target, ok, accepted: draft.slice(0, first < 0 ? draft.length : first) };
  };
  const r1 = round(FIG_PROMPT, ['Programming', 'Massively', 'Efficient'], ['Programming', 'Massively', 'Parallel']);
  const r2 = round([...FIG_PROMPT, ...r1.accepted], ['Parallel', 'Processors', 'EOS'], ['Parallel', 'Processors', 'EOS']);
  return [r1, r2];
}

export type FigPhase = 'start' | 'draft' | 'verify' | 'accept';
/** The figure as steps: start, then draft, verify, accept for each round. */
export const FIG_STEPS: { round: number; phase: FigPhase }[] = [
  { round: -1, phase: 'start' },
  { round: 0, phase: 'draft' },
  { round: 0, phase: 'verify' },
  { round: 0, phase: 'accept' },
  { round: 1, phase: 'draft' },
  { round: 1, phase: 'verify' },
  { round: 1, phase: 'accept' },
];

/** Counters after a step of the replay: target passes, tokens kept, and what plain decoding would need. */
export function figCounters(step: number): { targetPasses: number; draftTokens: number; tokens: string[] } {
  const rounds = replayFigure();
  let targetPasses = 0;
  let draftTokens = 0;
  const tokens: string[] = [];
  for (let s = 1; s <= Math.min(step, FIG_STEPS.length - 1); s++) {
    const { round, phase } = FIG_STEPS[s];
    if (phase === 'draft') draftTokens += rounds[round].draft.length;
    if (phase === 'verify') targetPasses++;
    if (phase === 'accept') tokens.push(...rounds[round].accepted);
  }
  return { targetPasses, draftTokens, tokens };
}

// ------------------------------------------------------------------ the simulator

export interface SpecConfig {
  /** Speculation depth: draft tokens per round. */
  k: number;
  /** Probability that one draft token matches the target's, independent per token. */
  alpha: number;
  seed: number;
  /** Tokens to generate. */
  tokens: number;
  /** Ours: also keep the target's own token at the first mismatch. */
  keepTarget: boolean;
}

export interface SpecRound {
  /** Leading draft tokens that matched. */
  matched: number;
  /** Tokens the round adds to the output. */
  kept: number;
}

export interface SpecResult {
  rounds: SpecRound[];
  /** One per round: each verifies k query rows against the cached prefix in one pass. */
  targetPasses: number;
  draftTokens: number;
  tokens: number;
  tokensPerPass: number;
  /** Times the whole cache is read per token kept. Plain decoding: 1. */
  cacheReadsPerToken: number;
  /** Query rows that share each read of the cache: k. */
  rowsPerRead: number;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Rounds stop once `tokens` are kept; a run never makes more than maxPasses target passes. */
export function simulate(cfg: SpecConfig, maxPasses = 200_000): SpecResult {
  const rand = mulberry32(cfg.seed * 2654435761 + cfg.k * 97 + Math.round(cfg.alpha * 1000));
  const rounds: SpecRound[] = [];
  let tokens = 0;
  while (tokens < cfg.tokens && rounds.length < maxPasses) {
    let matched = 0;
    while (matched < cfg.k && rand() < cfg.alpha) matched++;
    const kept = cfg.keepTarget ? Math.min(cfg.k, matched + 1) : matched;
    rounds.push({ matched, kept });
    tokens += kept;
  }
  const targetPasses = rounds.length;
  return {
    rounds,
    targetPasses,
    draftTokens: targetPasses * cfg.k,
    tokens,
    tokensPerPass: targetPasses ? tokens / targetPasses : 0,
    cacheReadsPerToken: tokens ? targetPasses / tokens : Infinity,
    rowsPerRead: cfg.k,
  };
}

/** Expected tokens per target pass (closed form, see the header). */
export function expectedPerPass(k: number, alpha: number, keepTarget: boolean): number {
  if (alpha >= 1) return k;
  const geo = (1 - alpha ** k) / (1 - alpha); // sum_{i=0..k-1} alpha^i
  return keepTarget ? geo : alpha * geo;
}

/** Mean tokens per target pass over many seeds (each run generating `tokens`). */
export function meanPerPass(k: number, alpha: number, keepTarget: boolean, seeds: number, tokens = 256): number {
  let t = 0;
  let passes = 0;
  for (let s = 1; s <= seeds; s++) {
    const r = simulate({ k, alpha, seed: s, tokens, keepTarget });
    t += r.tokens;
    passes += r.targetPasses;
  }
  return t / passes;
}
