import { describe, expect, it } from 'vitest';
import {
  FORMS,
  FORM_IDS,
  SCENARIOS,
  UNVISITED,
  canMove,
  changedLines,
  classify,
  enumerate,
  initial,
  region,
  replay,
  runOrder,
  step,
  type FormId,
  type OutcomeKind,
} from './bfsvisit';

const U = UNVISITED;
const kinds = (id: FormId, sc: keyof typeof SCENARIOS, spurious = 0) => enumerate(FORMS[id], SCENARIOS[sc], { spurious }).outcomes.map((o) => o.kind);
const final = (id: FormId, sc: keyof typeof SCENARIOS, order: 'forward' | 'reverse') => runOrder(FORMS[id], SCENARIOS[sc], order);

describe('the scenarios', () => {
  it("the book's graph at level 3, root 0: threads on 3, 4 and 6, and 8 is their only unvisited neighbour", () => {
    const s = SCENARIOS.book;
    expect(s.level0).toEqual([0, 1, 1, 2, 2, 2, 2, 2, U]);
    expect(s.threads).toEqual([{ vertex: 3, nbs: [4, 8] }, { vertex: 4, nbs: [5, 8] }, { vertex: 6, nbs: [8] }]);
  });
  it('the trap: 3 is shared, 4 is reached only by the thread on 2, after 3', () => {
    expect(SCENARIOS.trap.threads).toEqual([{ vertex: 1, nbs: [3] }, { vertex: 2, nbs: [3, 4] }]);
    expect(SCENARIOS.trap.expectNew).toEqual([3, 4]);
  });
});

describe('every interleaving', () => {
  for (const sc of ['book', 'trap'] as const) {
    it(`${sc}: compare-and-swap (Fig. 18.14) is exactly-once in every order`, () => {
      expect(kinds('cas', sc)).toEqual(['correct']);
    });
    it(`${sc}: plain check-then-label never misses, but some order writes a slot twice`, () => {
      const k = kinds('plain', sc);
      expect(k).toContain('duplicate-label');
      expect(k).toContain('correct');
      expect(k).not.toContain('missed');
    });
    it(`${sc}: plain with a frontier append puts a vertex in the frontier twice in some order`, () => {
      expect(kinds('plain-append', sc)).toContain('duplicate-entry');
    });
    it(`${sc}: while (!cas(e)) e = UNVISITED never ends once another thread has won`, () => {
      const e = enumerate(FORMS['weak-reset'], SCENARIOS[sc]);
      expect(e.cyclic).toBe(true);
      expect(e.outcomes.map((o) => o.kind)).toContain('never-terminates');
      expect(final('weak-reset', sc, 'forward').outcome.kind).toBe('never-terminates');
    });
    it(`${sc}: while (!cas(e) && e == UNVISITED) is right in every order with 0 to 4 injected failures`, () => {
      for (let k = 0; k <= 4; k++) expect(kinds('weak-correct', sc, k), `spurious ${k}`).toEqual(['correct']);
    });
  }
  it('trap: e hoisted above the loop misses vertex 4 in the forward order, and is right in reverse', () => {
    const f = final('hoisted', 'trap', 'forward');
    expect(f.outcome.kind).toBe('missed');
    expect(f.states.at(-1)!.level).toEqual([0, 1, 1, 2, U]);
    expect(f.states.at(-1)!.threads[1].e).toBe(U); // the failed swap on 4 loaded UNVISITED back
    expect(final('hoisted', 'trap', 'reverse').outcome.kind).toBe('correct');
    expect(kinds('hoisted', 'trap')).toEqual(['missed', 'correct']);
  });
  it('trap: the weak compare-and-swap with no loop is right with no spurious failure, and misses with one', () => {
    expect(kinds('weak', 'trap', 0)).toEqual(['correct']);
    expect(kinds('weak', 'trap', 1)).toContain('missed');
  });
  it('trap: while (!cas(e)) {} appends 3 twice: the retry succeeds against the winner\'s value', () => {
    const f = final('weak-loop', 'trap', 'forward');
    expect(f.outcome.kind).toBe('duplicate-entry');
    expect(f.states.at(-1)!.frontier).toEqual([3, 3, 4]);
    expect(kinds('weak-loop', 'trap')).toEqual(['duplicate-entry']);
  });
  it("book: while (!cas(e)) {} also overwrites an older level: 4 and 5 were at 2 and end at 3", () => {
    const f = final('weak-loop', 'book', 'forward');
    expect(f.outcome.kind).toBe('wrong-level');
    expect(f.states.at(-1)!.level).toEqual([0, 1, 1, 2, 3, 3, 2, 2, 3]);
    expect(f.states.at(-1)!.frontier).toEqual([4, 8, 5, 8, 8]);
  });
});

describe('stepping by hand', () => {
  it('a spurious failure can be injected only into a weak compare-and-swap that would succeed', () => {
    const sc = SCENARIOS.trap;
    let st = initial(sc);
    st = step(FORMS.weak, sc, st, { thread: 0 }).state; // takes neighbour 3
    expect(canMove(FORMS.weak, st, { thread: 0, spurious: true })).toBe(true);
    expect(canMove(FORMS.cas, st, { thread: 0, spurious: true })).toBe(false);
    const r = step(FORMS.weak, sc, st, { thread: 0, spurious: true });
    expect(r.state.level[3]).toBe(U);
    expect(r.state.threads[0].e).toBe(U);
    expect(r.event.text).toContain('spuriously');
  });
  it('the hoisted trap, step by step: the failed swap on 3 leaves e = 2, and the swap on 4 then fails', () => {
    const sc = SCENARIOS.trap;
    const moves = [0, 0, 0, 0, 1, 1].map((thread) => ({ thread }));
    const r = replay(FORMS.hoisted, sc, moves);
    expect(r.states.at(-1)!.threads[1].e).toBe(2);
    expect(r.events.at(-1)!.text).toContain('loads 2 into e');
    const r2 = replay(FORMS.hoisted, sc, [...moves, { thread: 1 }, { thread: 1 }]);
    expect(r2.events.at(-1)!.text).toContain('finds UNVISITED, not e');
  });
  it('each step names a line of the form\'s listing', () => {
    for (const id of FORM_IDS) {
      const f = FORMS[id];
      const r = runOrder(f, SCENARIOS.trap, 'forward', 6);
      for (const e of r.events) expect(f.code[e.line], `${id} line ${e.line}`).toBeDefined();
      // The compare-and-swap lines name the right helper.
      if (f.line.cas !== undefined) expect(f.code[f.line.cas]).toContain(f.weak ? 'casWeak' : 'casStrong');
      if (f.line.check !== undefined) expect(f.code[f.line.check]).toContain('level[neighbor] == UNVISITED');
      if (f.line.append !== undefined) expect(f.code[f.line.append]).toContain('append(');
    }
  });
  it('an example order for every outcome reproduces that outcome when replayed', () => {
    for (const sc of ['book', 'trap'] as const)
      for (const id of FORM_IDS) {
        const e = enumerate(FORMS[id], SCENARIOS[sc], { spurious: 1 });
        for (const o of e.outcomes) {
          if (o.kind === 'never-terminates') continue;
          const r = replay(FORMS[id], SCENARIOS[sc], o.example);
          expect(classify(SCENARIOS[sc], r.states.at(-1)!).kind, `${sc} ${id}`).toBe(o.kind as OutcomeKind);
        }
      }
  });
});

describe('listing helpers', () => {
  it('changedLines marks replaced and inserted lines of the second text', () => {
    expect(changedLines('a\nb\nc', 'a\nB\nc')).toEqual([2]);
    expect(changedLines('a\nb\nc', 'a\nx\nb\ny\nc')).toEqual([2, 4]);
    expect(changedLines('a\nb', 'a\nb')).toEqual([]);
  });
  it('region finds the text between the markers', () => {
    expect(region('x\n// region: r1\nline 1\nline 2\n// endregion\ny', 'r1')).toBe('line 1\nline 2');
    expect(region('nothing', 'r1')).toBeNull();
  });
});
