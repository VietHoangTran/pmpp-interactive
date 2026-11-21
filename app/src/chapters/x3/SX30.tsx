import { useEffect, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { ATTENTION_HOME } from '../../sim/attnroute';
import { ATTN_HOME_OPTIONS, OPS, PROJECTS, PROJECT_NAME, attentionHome, type AttnHome, type Project } from '../../sim/backendmap';
import { PINNED, READ_ON, permalink, shortCommit, type Excerpt as ExcerptData } from '../../sim/x3sources';
import { loadChapter } from '../registry';
import { Excerpt } from './Excerpt';

// Module X3's start page: what reading a real backend means here, the snapshot every claim
// rests on (rendered from sim/x3sources.ts PINNED), the method in one paragraph, the tour's
// ops and where the course taught each (from sim/backendmap.ts OPS), how excerpts are shown,
// milestone M10, a prediction X3.5 settles, and a prerequisite check that reuses earlier
// chapters' questions by id (loaded on demand through the registry, never imported).

const FACTS = 'notes/today/x3-facts.md';
const P_HOME = 'x3/start/p-attention-home';

export const EXCERPTS_SX30: Record<string, ExcerptData> = {
  addH: {
    id: 'x30-ufunc-add', project: 'pytorch', path: 'aten/src/ATen/native/ufunc/add.h', lines: null, factsItem: 'A3', lang: 'cpp',
    code: `return self + alpha * other;`,
  },
};

export const questionsSX30: Record<string, Question> = {
  'x3/start/q-hip-same': {
    kind: 'choice',
    id: 'x3/start/q-hip-same',
    prompt: (
      <>
        llama.cpp has a directory <code>ggml/src/ggml-hip</code>. Where is the HIP softmax kernel?
      </>
    ),
    options: [
      { label: 'In ggml-cuda/softmax.cu: ggml-hip holds build files that compile ../ggml-cuda/*.cu as HIP.', correct: true },
      { label: 'In a hipified copy of softmax.cu that the build writes into ggml-hip.', why: 'That is PyTorch’s approach. llama.cpp renames CUDA identifiers with #defines in vendors/hip.h and compiles the CUDA file itself.' },
      { label: 'In ggml-hip/softmax.hip, a port kept beside the CUDA one.', why: 'There is no such file: ggml-hip contains only CMakeLists.txt.' },
      { label: 'Nowhere: llama.cpp runs softmax on the CPU when the GPU is AMD.', why: 'The HIP backend accepts softmax; it is the CUDA kernel compiled for AMD.' },
    ],
    explain: (
      <p>
        <code>ggml-hip/CMakeLists.txt</code> globs <code>../ggml-cuda/*.cu</code> and sets <code>LANGUAGE HIP</code>. The one
        visible vendor branch in <code>softmax.cu</code> is an <code>#ifdef GGML_USE_HIP</code> that includes{' '}
        <code>&lt;hip/hip_cooperative_groups.h&gt;</code>.
      </p>
    ),
    source: `${FACTS} B9; notes/today/infra-facts.md D2; notes/today/ch05-facts.md B9`,
  },
  'x3/start/q-pinned': {
    kind: 'choice',
    id: 'x3/start/q-pinned',
    prompt: <>Why does every link in this module point at a commit hash rather than at <code>master</code> or <code>main</code>?</>,
    options: [
      { label: 'The excerpts and line numbers were read at that commit, and master moves: a master link can show different code tomorrow.', correct: true },
      { label: 'The projects’ licences require a link to a fixed version.', why: 'None of the licence texts read says so. The reason is accuracy, not licensing.' },
      { label: 'GitHub does not serve line anchors on branch links.', why: 'It does. They just point at whatever the branch holds today.' },
      { label: 'Master is private for these projects; only commits are public.', why: 'All three default branches are public.' },
    ],
    explain: (
      <p>
        The facts pass read each file at one commit. Several files in the tour changed on the day of reading, so a line number
        on <code>master</code> could already be off. A permalink keeps the excerpt, the line numbers and the page in agreement.
        Read <code>master</code> yourself when you want today’s code, and expect it to differ.
      </p>
    ),
    source: `${FACTS}, header (“Snapshot”)`,
  },
};

/** Earlier chapters' questions this page asks again, by id. */
const PRIOR: { id: string; label: string }[] = [
  { id: 'ch10/10.6/q-two-reductions', label: 'From chapter 10' },
  { id: 'ch10/10.2/q-argmax-rule', label: 'From chapter 10' },
  { id: 'ch04/4.5/q-amd32', label: 'From chapter 4' },
  { id: 'ch15/15.5/q-float4-alone', label: 'From chapter 15' },
  { id: 'ch20/20.1/q-rope', label: 'From chapter 20' },
  { id: 'ch20/20.5/q-init', label: 'From chapter 20' },
  { id: 'ch22/start/q-gemv-ceiling', label: 'From chapter 22' },
  { id: 'appA/A.6/q-fastmath', label: 'From appendix A' },
  { id: 'x1/X1.2/q-greedy-ties', label: 'From module X1' },
];

/** Ask an earlier chapter's question by its id: its chapter and section load on demand. */
function PriorAsk({ id, label }: { id: string; label: string }) {
  const [q, setQ] = useState<Question | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    const [ch, sec] = id.split('/');
    const p = loadChapter(ch);
    if (!p) {
      setQ(null);
      return;
    }
    p.then((c) => c.sections.find((s) => s.id === sec)?.load())
      .then((m) => live && setQ(m?.questions[id] ?? null))
      .catch(() => live && setQ(null));
    return () => {
      live = false;
    };
  }, [id]);
  if (q === undefined) return <p className="muted ui" data-prior={id}>Loading a question from {id.split('/')[0]}…</p>;
  if (q === null) return <p className="muted ui" data-prior={id} data-missing>The question {id} is not available.</p>;
  const [ch, sec] = id.split('/');
  return (
    <div data-prior={id}>
      <Ask q={q} label={label} />
      <p className="muted ui" style={{ margin: '-0.6rem 0 1.2rem', fontSize: '0.8rem' }}>
        Asked first in <a href={href(ch, sec)}>{ch} {sec === 'start' ? 'start page' : sec}</a>.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- the prediction X3.5 settles
type HomePick = Partial<Record<Project, AttnHome>>;
function AttentionHomePredict() {
  const { progress, answer } = useProgress();
  const prior = progress.answers[P_HOME]?.value as HomePick | undefined;
  const [pick, setPick] = useState<HomePick>({});
  const [show, setShow] = useState(false);
  const committed = prior && PROJECTS.every((p) => prior[p]) ? prior : null;
  const all = PROJECTS.every((p) => pick[p]);
  const cudaRows = ATTENTION_HOME.filter((r) => r.backend.includes('CUDA'));
  return (
    <div className="q" data-part="predict-home">
      <div className="kind">Predict first</div>
      <div className="prompt">
        <p style={{ margin: '0 0 0.5rem' }}>
          You will read three projects’ attention code on NVIDIA GPUs. For each, where do the CUDA attention kernels it runs
          come from? Commit one answer per project now; section <a href={href('x3', 'X3.5')}>X3.5</a> settles it.
        </p>
      </div>
      {committed == null ? (
        <>
          <div style={{ display: 'grid', gap: '0.5rem', marginBottom: '0.6rem' }}>
            {PROJECTS.map((p) => (
              <fieldset key={p} style={{ border: '1px solid var(--rule)', borderRadius: 4, padding: '0.4rem 0.6rem', margin: 0 }} data-project={p}>
                <legend className="ui" style={{ fontWeight: 600, padding: '0 0.3rem' }}>{PROJECT_NAME[p]}</legend>
                {ATTN_HOME_OPTIONS.map((o) => (
                  <label key={o.id} className="check" style={{ display: 'flex', marginBottom: '0.15rem' }}>
                    <input type="radio" name={`${P_HOME}-${p}`} checked={pick[p] === o.id} onChange={() => setPick({ ...pick, [p]: o.id })} data-home={o.id} />
                    <span>{o.label}</span>
                  </label>
                ))}
              </fieldset>
            ))}
          </div>
          <button className="primary" disabled={!all} onClick={() => answer(P_HOME, pick)}>Commit prediction</button>
        </>
      ) : (
        <div className="verdict info" data-readout="home-committed">
          <p>
            <b>You predicted:</b>{' '}
            {PROJECTS.map((p, i) => (
              <span key={p}>
                {i ? '; ' : ''}
                {PROJECT_NAME[p]}, {ATTN_HOME_OPTIONS.find((o) => o.id === committed[p])!.label}
              </span>
            ))}
            .
          </p>
          {show ? (
            <div data-readout="home-reveal">
              {PROJECTS.map((p) => {
                const right = attentionHome(p);
                const row = cudaRows.find((r) => PROJECT_NAME[p] === r.project);
                return (
                  <p key={p} data-project={p} data-right={committed[p] === right ? '1' : '0'}>
                    <b className={committed[p] === right ? 't-ok' : 't-fail'}>{PROJECT_NAME[p]}:</b>{' '}
                    {ATTN_HOME_OPTIONS.find((o) => o.id === right)!.label}. {row ? <span className="muted">({row.home}.)</span> : null}
                  </p>
                );
              })}
              <p className="muted">Section <a href={href('x3', 'X3.5')}>X3.5</a> follows each one to its files.</p>
            </div>
          ) : (
            <p>
              Section X3.5 reveals it as you read the code.{' '}
              <button className="quiet" onClick={() => setShow(true)}>Show me now</button>
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the section
export function SX30() {
  const q = questionsSX30;
  const tour = [...new Map(OPS.map((o) => [o.tour, o])).keys()].map((t) => {
    const rows = OPS.filter((o) => o.tour === t);
    const labs = [...new Map(rows.flatMap((o) => o.course).map((l) => [l.num, l])).values()];
    return { tour: t, section: rows[0].section, labs };
  });
  return (
    <>
      <p>
        The course has had you write every kernel of a Llama-style decode step yourself. This module reads the kernels that
        llama.cpp, MLX and PyTorch run for the same ops, on CUDA, HIP and Metal. The aim is the third goal you set: to read and
        contribute to those backends. By the end you should be able to name the file and function that runs for any op in
        the tour, on any of the three backends, and say what it does differently from your own kernel.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The book has no chapter on real backends, so there are no book pages and no notes file for this module. Every claim
          about a codebase was read in one facts pass on {READ_ON}, at the commits below, and is cited by its item in{' '}
          <code>{FACTS}</code>. Nothing was built or run: what a kernel “does” is a reading of its source.
        </p>
      </Beyond>

      <h2>The snapshot</h2>
      <p>
        Real projects move daily. Several of the files this module reads changed on the day they were read. So the module
        fixes one commit per project, and every link is a permalink to it: the excerpt you see, its line numbers and the file
        behind the link agree.
      </p>
      <Today checked={`${READ_ON} in the facts pass (${FACTS}, header and A4)`}>
        <p>
          Files were fetched at these commits; each file’s last-commit date is GitHub’s for that path at or before the commit.
          Not checked: anything by building or running the projects.
        </p>
      </Today>
      <table className="data wide" data-part="snapshot" style={{ maxWidth: '62rem' }}>
        <thead>
          <tr><th>Project</th><th>Commit</th><th>Committed</th><th>Licence</th><th>Copyright line</th></tr>
        </thead>
        <tbody>
          {PINNED.map((p) => (
            <tr key={p.id} data-project={p.id}>
              <td><a href={permalink(p.id, '', null, true)} target="_blank" rel="noreferrer">{p.name}</a> <span className="muted">{p.repo}</span></td>
              <td className="nowrap"><code data-commit={p.commit}>{shortCommit(p.id)}</code></td>
              <td className="nowrap">{p.date ?? '—'}</td>
              <td>{p.licence}{p.id === 'pytorch' ? ' (GitHub reports NOASSERTION)' : ''}</td>
              <td>{p.copyright ?? <span className="muted">not quoted in the facts pass</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Some cells of the tour rest on items settled in earlier chapters’ facts files (llama.cpp’s attention kernels in
        chapter 20, its GEMV switch in chapter 22, the samplers in module X1). Those were read at <code>master</code> on the
        same day. The module links them by their facts item and their own URL, and never gives them line numbers at the
        pinned commit.
      </p>
      <Ask q={q['x3/start/q-pinned']} />

      <h2>The method, in one paragraph</h2>
      <p>
        Each project has one place where an op becomes a backend call. In llama.cpp it is a switch over the graph’s op enum,
        one per backend. In MLX it is a method, <code>eval_gpu</code>, that each primitive implements once per backend
        directory. In PyTorch it is the dispatch table in <code>native_functions.yaml</code>. Start there and follow one
        name. On Metal the name is a string the host builds, so follow the string. Section{' '}
        <a href={href('x3', 'X3.1')}>X3.1</a> does this for every op, including the two places it breaks: fusion, which runs
        before the switch, and code that is generated at build time or at run time.
      </p>
      <Today checked={`${READ_ON} in the facts pass (${FACTS}, B9)`}>
        <p>
          llama.cpp and PyTorch compile their CUDA sources for AMD. MLX has no AMD backend: <code>mlx/backend</code> lists
          common, cpu, cuda, gpu, metal, no_cpu, no_gpu; pull request #2300, “[Experiment] ROCm backend”, is open.
        </p>
      </Today>
      <Ask q={q['x3/start/q-hip-same']} />

      <h2>The tour</h2>
      <p>
        Eight ops make a Llama decode step. Each has a section here and a place in the course where you met its technique.
        Three you have not written yet, RMSNorm, RoPE and SwiGLU, you build in this module’s first lab.
      </p>
      <Aside>
        <p>
          A bridge to keep in mind: <code>torch.nn.functional.rms_norm</code> in a Hugging Face Llama layer. By the end of
          X3.1 you will have found the kernel it reaches on CUDA and on MPS.
        </p>
      </Aside>
      <table className="data" data-part="tour" style={{ maxWidth: '44rem' }}>
        <thead>
          <tr><th>Op</th><th>Where you wrote it</th><th>Read in</th></tr>
        </thead>
        <tbody>
          {tour.map((t) => (
            <tr key={t.tour} data-tour={t.tour}>
              <td>{t.tour}</td>
              <td>
                {t.labs.map((l, i) => (
                  <span key={l.num}>
                    {i ? ', ' : ''}
                    <a href={href(l.chapter, 'labs')}>lab {l.num}</a>
                    {l.planned ? ' (new)' : ''}
                  </span>
                ))}
              </td>
              <td><a href={href('x3', t.section)}>{t.section}</a></td>
            </tr>
          ))}
        </tbody>
      </table>
      <AttentionHomePredict />

      <h2>How excerpts are shown</h2>
      <p>
        Code from the three projects appears in short excerpts, at most about a dozen lines, copied from the facts pass. Each
        names its file and lines, links to the pinned commit, and carries the project’s licence and copyright line. A gap
        marks lines left out. Spacing can differ from the file, because the facts file records some quotes on one line.
      </p>
      <Today checked={`${READ_ON} in the facts pass (${FACTS}, A4)`}>
        <p>
          llama.cpp and MLX are MIT (“…shall be included in all copies or substantial portions of the Software”); PyTorch is
          three-clause BSD-style. Whether a short excerpt is a “substantial portion” is a legal question the pass could not
          settle: unverified.
        </p>
      </Today>
      <Excerpt x={EXCERPTS_SX30.addH} caption="An example: the only op-specific source of PyTorch’s CUDA add kernel, its scalar body; torchgen generates the rest at build time, as X3.1 explains." />

      <h2>The milestone</h2>
      <p>
        Milestone M10 composes your milestones M0 to M8 into one Llama-style decode step, adds the pieces the course has not
        built (RMSNorm, RoPE, the SiLU-gated feed-forward), and runs it on a small model that the lab generates in the
        repository from a fixed seed, checked against a CPU reference. The build downloads nothing. Pointing the same pipeline
        at a real checkpoint you download yourself is described in the lab’s README and in section{' '}
        <a href={href('x3', 'X3.8')}>X3.8</a>.
      </p>
      <p>
        M10 runs float32 weights. Module X2's quantised layer (M9, lab X2.4) now exists, but its launch function takes a
        quantised matrix descriptor rather than the float weights M10’s linear-layer slot expects, so M10 does not switch it
        in yet. The <a href={href('x3', 'labs')}>labs page</a> says what would have to meet.
      </p>
      <Beyond title="Project note">
        <p>
          The facts file’s project notes list what M0 to M8 provide and what is missing for a Llama step: RMSNorm, RoPE,
          SwiGLU, and a head size other than 128 (none of the small real checkpoints has 128).
        </p>
      </Beyond>
      <ul>
        <li><b>Lab X3.1</b>, the Llama pieces the course has not built, and a port review.</li>
        <li><b>Lab X3.2</b>, milestone M10: a Llama-style model end to end on the course’s kernels.</li>
      </ul>
      <p>Both are on the <a href={href('x3', 'labs')}>labs page</a>, each with a prediction to make before you measure.</p>

      <h2>Before you start</h2>
      <p>
        Nine questions from earlier chapters, on the things this module assumes: two-stage reductions and their barriers, tie
        rules, warp width, vector stores, RoPE in a KV cache, the online softmax, the GEMV ceiling, Metal’s math mode, and
        greedy ties across backends. Each links to where it was first asked.
      </p>
      <Bench title="Prerequisite check" task="Answer each; wrong answers come back in review.">
        {PRIOR.map((p) => <PriorAsk key={p.id} id={p.id} label={p.label} />)}
      </Bench>
    </>
  );
}
