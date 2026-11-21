#!/usr/bin/env python3
"""Build, test and benchmark the labs on whatever backends this machine has.

  python3 labs/pmpp.py doctor                 which backends can be built here, and why not
  python3 labs/pmpp.py run 5.2                test your work for lab 5.2 on every backend
  python3 labs/pmpp.py run 5.2 --bench        ...and benchmark it
  python3 labs/pmpp.py run 5.2 --bench --reps 5   ...five times, each repetition recorded
  python3 labs/pmpp.py run 5.2 --solution     run the reference solution instead
  python3 labs/pmpp.py run 5.2 --backend emu  one backend only (emu, race, metal, cuda, hip)
  python3 labs/pmpp.py run 5.2 --full         print sanitizer reports in full (default: condensed)
  python3 labs/pmpp.py selftest               the emulator's own tests (labs/selftest), no report written
  python3 labs/pmpp.py verify                 everything, and write reports/verify-*.md
  python3 labs/pmpp.py reset 5.2 --yes        overwrite your work for a lab with the starter
  python3 labs/pmpp.py starters [--record]    check (or append to) labs/starter-history.json
  python3 labs/pmpp.py frontend               clang's CUDA front-end check of every .cu, recorded for the app

Standard library only. Needs cmake >= 3.21 and a C++17 compiler.
Environment: PMPP_BUILD_DIR (build directory), PMPP_ONLY (configure one chapter's labs),
PMPP_CMAKE_ARGS (extra arguments for the cmake configure step, split like a shell would,
e.g. PMPP_CMAKE_ARGS="-DPMPP_CH15_VENDOR=OFF -DPMPP_DISABLE_RACE=ON").
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

LABS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(LABS)
BUILD = os.environ.get("PMPP_BUILD_DIR", os.path.join(LABS, "build"))
REPORTS = os.path.join(REPO, "reports")
BACKENDS = ["emu", "race", "metal", "cuda", "hip"]
# CPU emulators: tests only. Their timings mean nothing, so they are never benchmarked.
EMULATED = ("emu", "race")
STUDENT_FILES = ["kernel.cu", "kernel.metal", "launch.mm"]


def scrub(text):
    """Remove machine-specific path prefixes so a report can be shared."""
    text = text.replace(REPO, "<repo>")
    home = os.path.expanduser("~")
    return text.replace(home, "~") if home and home != "/" else text


def sh(cmd, timeout=None, cwd=None):
    """Run a command; return (exit code, combined output). 124 = timed out."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b"").decode("utf-8", "replace") + "\n[timed out after %ss]" % timeout
    except FileNotFoundError:
        return 127, "command not found: %s" % cmd[0]


def tail(text, n):
    lines = text.rstrip().splitlines()
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------- configure
def configure():
    if shutil.which("cmake") is None:
        sys.exit("cmake not found. Install cmake >= 3.21 (macOS: brew install cmake; Linux: your package manager).")
    try:
        extra = shlex.split(os.environ.get("PMPP_CMAKE_ARGS", ""))
    except ValueError as e:
        sys.exit("PMPP_CMAKE_ARGS could not be split into arguments: %s" % e)
    rc, out = sh(["cmake", "-S", LABS, "-B", BUILD] + extra)
    backends, targets, selftests, skipped, incomplete = [], [], [], [], []
    try:
        for line in open(os.path.join(BUILD, "pmpp_backends.txt")):
            name, state, why = line.rstrip("\n").split("|", 2)
            backends.append({"name": name, "enabled": state == "enabled", "why": why})
        for line in open(os.path.join(BUILD, "pmpp_targets.txt")):
            lab, variant, backend, tile, target, exe = line.rstrip("\n").split("|")
            targets.append({"lab": lab, "variant": variant, "backend": backend, "tile": int(tile),
                            "target": target, "exe": exe})
        for line in open(os.path.join(BUILD, "pmpp_selftests.txt")):
            name, backend, expect, target, exe = line.rstrip("\n").split("|")
            selftests.append({"lab": "selftest/" + name, "variant": "selftest", "backend": backend, "tile": 0,
                              "expect": expect, "target": target, "exe": exe})
    except FileNotFoundError:
        pass
    # Builds pmpp_add_lab did not make, with the reason (no kernel.metal, no RACE, a missing
    # library), and labs it skipped as incomplete. Older build directories may lack them.
    for fname, keys, into in (("pmpp_skipped.txt", ("lab", "variant", "backend", "why"), skipped),
                              ("pmpp_incomplete.txt", ("lab", "missing"), incomplete)):
        try:
            for line in open(os.path.join(BUILD, fname)):
                parts = line.rstrip("\n").split("|", len(keys) - 1)
                if len(parts) == len(keys):
                    into.append(dict(zip(keys, parts)))
        except FileNotFoundError:
            pass
    return {"rc": rc, "log": out, "backends": backends, "targets": targets, "selftests": selftests,
            "skipped": skipped, "incomplete": incomplete, "cmake_args": extra}


def build(target):
    rc, out = sh(["cmake", "--build", BUILD, "--target", target, "-j"], timeout=1800)
    return rc == 0, out


# A sanitizer report has begun when one of these appears in a lab's output.
SANITIZER_MARKS = ("ERROR: AddressSanitizer", "ThreadSanitizer: SEGV", "ThreadSanitizer: BUS",
                   "ERROR: ThreadSanitizer", "ThreadSanitizer: CHECK failed")
SANITIZER_STALL = 60  # seconds of silence after a report starts before the run is stopped


def sh_watched(cmd, timeout, env=None):
    """Like sh(), for lab executables. The run gets its own process group, so a timeout
    stops the sanitizer's symbolizer too (macOS's atos can hang while a report prints and
    keeps the output pipe open). Once a sanitizer report has started, a minute without new
    output ends the run, keeping the report printed so far."""
    import threading, time
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    except FileNotFoundError:
        return 127, "command not found: %s" % cmd[0]
    chunks, last = [], [time.monotonic()]

    def reader():
        for line in iter(p.stdout.readline, b""):
            chunks.append(line)
            last[0] = time.monotonic()
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    start, note = time.monotonic(), ""
    while p.poll() is None:
        time.sleep(0.2)
        now = time.monotonic()
        if timeout and now - start > timeout:
            note = "\n[timed out after %ss]" % timeout
        elif now - last[0] > SANITIZER_STALL and any(m in b"".join(chunks[-400:]).decode("utf-8", "replace")
                                                      for m in SANITIZER_MARKS):
            note = "\n[stopped: the sanitizer report stalled for %ss (its symbolizer hung); the report so far is above]" % SANITIZER_STALL
        if note:
            try:
                os.killpg(p.pid, 9)
            except OSError:
                pass
            break
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(p.pid, 9)  # whatever the run left behind (a symbolizer still holding the pipe)
    except OSError:
        pass
    t.join(timeout=5)
    out = b"".join(chunks).decode("utf-8", "replace") + note
    if note.startswith("\n[timed out"):
        return 124, out
    return (p.returncode if p.returncode is not None else -9), out


# On macOS the sanitizers symbolize through atos attached to the live process, and that
# sometimes hangs before the report prints (seen on labs 18.1, X2.3 and X3.1). So emulator
# runs start with symbolize=0 and pmpp.py looks the addresses up afterwards with atos on the
# binary file, which does not attach and is quick; the report then reads as usual.
OFFLINE_SYMBOLIZE = platform.system() == "Darwin" and shutil.which("atos") is not None
_FRAME = re.compile(r"^(\s*#\d+ )(0x[0-9a-f]+\s+)?(?:in )?(?:<null> <null> |<null> )?\(([^()\s]+):(arm64e?|x86_64)\+(0x[0-9a-f]+)\)")
_SUMMARY_LOC = re.compile(r"\(([^()\s]+):(arm64e?|x86_64)\+(0x[0-9a-f]+)\)\s*$")


def emulator_env(exe):
    env = dict(os.environ)
    if OFFLINE_SYMBOLIZE and exe.endswith((".emu", ".race")):
        for var in ("ASAN_OPTIONS", "TSAN_OPTIONS"):
            env[var] = (env[var] + ":" if env.get(var) else "") + "symbolize=0"
    return env


def symbolize_offline(out, exe):
    """Replace unsymbolized frames of `exe` in a sanitizer report with 'function file:line'."""
    if not OFFLINE_SYMBOLIZE or not exe.endswith((".emu", ".race")):
        return out
    name = os.path.basename(exe)
    mine = lambda mod: mod == exe or os.path.basename(mod) == name
    lines, want = out.splitlines(), {}
    for ln in lines:
        m = _FRAME.match(ln)
        if m and mine(m.group(3)):
            want.setdefault(m.group(4), set()).add(m.group(5))
        if ln.lstrip().startswith("SUMMARY:"):
            s = _SUMMARY_LOC.search(ln)
            if s and mine(s.group(1)):
                want.setdefault(s.group(2), set()).add(s.group(3))
    if not want:
        return out
    names = {}
    for arch, addrs in want.items():
        addrs = sorted(addrs)
        for i in range(0, len(addrs), 400):
            chunk = addrs[i:i + 400]
            rc, res = sh(["atos", "-o", exe, "-arch", arch] + chunk, timeout=60)
            got = res.splitlines() if rc == 0 else []
            if len(got) != len(chunk):
                continue  # leave these frames as addresses
            for a, g in zip(chunk, got):
                m = re.match(r"^(.*?) \(in [^)]*\)(?: \(([^()]+:\d+)\))?", g)
                if m:
                    names[(arch, a)] = (m.group(1), m.group(2))
    res_lines = []
    for ln in lines:
        m = _FRAME.match(ln)
        if m and mine(m.group(3)) and (m.group(4), m.group(5)) in names:
            fn, loc = names[(m.group(4), m.group(5))]
            # ASan frames read "#0 0xADDR in fn file:line", TSan frames "#0 fn file:line".
            lead = m.group(1) + (m.group(2).strip() + " in " if m.group(2) else "")
            ln = "%s%s%s (%s:%s+%s)%s" % (lead, fn, " " + loc if loc else "", name, m.group(4), m.group(5), ln[m.end():])
        elif ln.lstrip().startswith("SUMMARY:"):
            s = _SUMMARY_LOC.search(ln)
            if s and mine(s.group(1)) and (s.group(2), s.group(3)) in names:
                fn, loc = names[(s.group(2), s.group(3))]
                ln = ln[:s.start()] + ("%s in %s" % (loc, fn) if loc else "in " + fn)
        res_lines.append(ln)
    return "\n".join(res_lines) + ("\n" if out.endswith("\n") else "")


def run_exe(exe, flags, timeout):
    rc, out = sh_watched([exe] + flags, timeout=timeout, env=emulator_env(exe))
    out = symbolize_offline(out, exe)
    rows, text = [], []
    for line in out.splitlines():
        if line.startswith("@pmpp "):
            try:
                rows.append(json.loads(line[6:]))
            except ValueError:
                text.append(line)
        else:
            text.append(line)
    return rc, rows, "\n".join(text)


def split_tsan(text):
    """Separate ThreadSanitizer reports (backend `race`) from a program's output.
    Returns (the output without the reports, one list of key lines per report):
    what kind of race, the two conflicting accesses with their innermost source
    line, the memory involved, and the summary."""
    keep, reports, cur, want_frame = [], [], None, False
    for line in text.splitlines():
        if cur is None:
            if line.startswith("WARNING: ThreadSanitizer:"):
                if keep and set(keep[-1].strip()) == {"="}:
                    keep.pop()
                cur = [re.sub(r"\s*\(pid=\d+\)", "", line.strip())]
            elif line.startswith("ThreadSanitizer: reported") or (reports and set(line.strip()) == {"="}):
                continue
            else:
                keep.append(line)
            continue
        t = line.strip()
        if re.match(r"(Previous )?(atomic )?(read|write) of size", t, re.I):
            cur.append("  " + re.sub(r" at 0x[0-9a-f]+", "", t))
            want_frame = True
        elif want_frame and t.startswith("#0 "):
            cur.append("      " + re.sub(r"\s*\(\S+\+0x[0-9a-f]+\)$", "", t[3:]))
            want_frame = False
        elif t.startswith("Location is"):
            cur.append("  " + re.sub(r" at 0x[0-9a-f]+", "", t))
        elif t.startswith("SUMMARY: ThreadSanitizer:"):
            cur.append(t)
            reports.append(cur)
            cur, want_frame = None, False
    if cur:
        reports.append(cur)
    return "\n".join(keep), reports


def tsan_text(reports, limit=4):
    """Key lines of the first report for each distinct source location, at most `limit`."""
    out, seen = [], []
    for rep in reports:
        where = rep[-1]
        if where in seen:
            continue
        seen.append(where)
        if len(seen) <= limit:
            out.extend(rep)
    more = len(reports) - min(len(seen), limit)
    if more > 0:
        out.append("(%d more ThreadSanitizer report%s not shown: %d on the lines above, %d elsewhere)" % (
            more, "" if more == 1 else "s", len(reports) - len(seen), max(0, len(seen) - limit)))
    return "\n".join(out)


# ---------------------------------------------------------------- condensed sanitizer reports
# An AddressSanitizer report from the emulator runs to a hundred lines or more: every frame
# of the emulator's thread machinery and of libc++, then the shadow-memory map. What a
# reader needs is what went wrong, where in their code, and where the memory came from.
# By default a run shows each report condensed (`--full` shows it all):
#   - the first report's headline (the ERROR line, the access, where the address lies),
#     each of its stacks cut down to the frames in the lab's own sources (the innermost
#     frame of the faulting stack is kept even when it is not), and its SUMMARY line;
#   - for any later report, its SUMMARY line only;
#   - a note saying how much was left out.
# ThreadSanitizer data-race reports (backend `race`) are condensed separately, by
# split_tsan() and tsan_text(). Offline symbolisation (symbolize_offline) runs first, so the
# frames it names are the ones condensed here.
_SAN_START = re.compile(r"^(==\d+==)?(ERROR: (AddressSanitizer|ThreadSanitizer|LeakSanitizer|UndefinedBehaviorSanitizer)"
                        r"|WARNING: ThreadSanitizer:|ThreadSanitizer: CHECK failed)")
_SAN_FRAME = re.compile(r"^\s*#\d+ ")
_SAN_FILE = re.compile(r"(?:^|\s)(\S+?\.[A-Za-z]+):(\d+)(?::\d+)?(?=\s|$)")
_SAN_TRAILER = (re.compile(r"^Shadow byte"), re.compile(r"^=>0x"), re.compile(r"^\s+0x[0-9a-f]+:"),
                re.compile(r"^\s+[A-Za-z][A-Za-z ]+:\s+([0-9a-f]{2}\s*)+$"), re.compile(r"^==\d+==ABORTING"),
                re.compile(r"^=+$"))
_SAN_NOISE = re.compile(r"^(==\d+==)?(AddressSanitizer:DEADLYSIGNAL|ABORTING|Register values:|"
                        r"AddressSanitizer can not provide additional info|HINT: this may be a false positive)")


def _frame_text(line):
    """A frame without its address, and without the module suffix when it has a file:line."""
    line = re.sub(r"^(\s*#\d+) 0x[0-9a-f]+ in ", r"\1 ", line)
    line = re.sub(r"^(\s*#\d+) 0x[0-9a-f]+ ", r"\1 ", line)
    if _SAN_FILE.search(line):
        line = re.sub(r"\s+\([^()\s]+:(arm64e?|x86_64)\+0x[0-9a-f]+\)\s*$", "", line)
    return line


def _own_frame(line, own_names):
    m = _SAN_FILE.search(line)
    return bool(m) and os.path.basename(m.group(1)) in own_names


def _condense_one(lines, own_names):
    out, dropped, pending, section, kept_in_section, first_frame = [], 0, None, 0, 0, None
    for ln in lines:
        if _SAN_FRAME.match(ln):
            if _own_frame(ln, own_names):
                if pending is not None:
                    out.append(pending)
                    pending = None
                out.append(_frame_text(ln))
                kept_in_section += 1
            else:
                dropped += 1
                if section == 0 and first_frame is None:
                    first_frame = (len(out), _frame_text(ln))
            continue
        if not ln.strip() or _SAN_NOISE.match(ln) or any(p.match(ln) for p in _SAN_TRAILER) \
                or len(re.findall(r"\S+ = 0x[0-9a-f]+", ln)) >= 2:
            continue
        # A new stack starts with a header such as "allocated by thread T0 here:".
        if ln.rstrip().endswith(":") and ("here" in ln or "by thread" in ln or "created by" in ln):
            if section == 0 and kept_in_section == 0 and first_frame is not None:
                out.insert(first_frame[0], first_frame[1])
                dropped -= 1
            section += 1
            kept_in_section = 0
            pending = re.sub(r"^==\d+==", "", ln.rstrip())
            continue
        out.append(re.sub(r"^==\d+==", "", ln.rstrip()))
    if section == 0 and kept_in_section == 0 and first_frame is not None:
        out.insert(first_frame[0], first_frame[1])
        dropped -= 1
    return out, dropped


def condense_sanitizer(text, own_names):
    """The program's output with each sanitizer report condensed as described above."""
    lines = text.splitlines()
    keep, reports, cur, after_summary = [], [], None, False
    for ln in lines:
        if cur is not None:
            if after_summary:
                if any(p.match(ln) for p in _SAN_TRAILER) or not ln.strip():
                    cur.append(ln)
                    if re.match(r"^==\d+==ABORTING", ln) or re.match(r"^=+$", ln):
                        reports.append(cur)
                        cur, after_summary = None, False
                    continue
                reports.append(cur)
                cur, after_summary = None, False
            else:
                cur.append(ln)
                if ln.lstrip().startswith("SUMMARY:"):
                    after_summary = True
                continue
        if _SAN_START.match(ln):
            while keep and (re.match(r"^=+$", keep[-1]) or _SAN_NOISE.match(keep[-1])):
                keep.pop()
            keep.append("\0report%d" % len(reports))
            cur, after_summary = [ln], ln.lstrip().startswith("SUMMARY:")
            continue
        if _SAN_NOISE.match(ln) and "DEADLYSIGNAL" in ln:
            continue
        keep.append(ln)
    if cur is not None:
        reports.append(cur)
    if not reports:
        return text
    out = []
    for ln in keep:
        if not ln.startswith("\0report"):
            out.append(ln)
            continue
        i = int(ln[len("\0report"):])
        rep = reports[i]
        if i == 0:
            body, dropped = _condense_one(rep, own_names)
            out.extend(body)
            notes = []
            if dropped > 0:
                notes.append("%d stack frame%s outside this lab's sources" % (dropped, "" if dropped == 1 else "s"))
            if any(x.startswith("Shadow byte") for x in rep):
                notes.append("the shadow-memory map")
            if any("Register values" in x for x in rep):
                notes.append("the register dump")
            if notes:
                out.append("[sanitizer report condensed: %s left out; --full shows the whole report]" % ", and ".join(notes))
        else:
            summ = [x for x in rep if x.lstrip().startswith("SUMMARY:")]
            out.append(summ[0].strip() if summ else re.sub(r"^==\d+==", "", rep[0]).strip())
            if i == len(reports) - 1:
                out.append("[%d more sanitizer report%s, condensed to the SUMMARY line above; --full shows them]" % (
                    len(reports) - 1, "" if len(reports) == 2 else "s"))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def lab_dir(lab):
    return os.path.join(LABS, lab)


STARTER_HISTORY = os.path.join(LABS, "starter-history.json")


def file_hashes(d):
    """{student file: sha256 of its bytes, or None when absent} for directory d."""
    out = {}
    for f in STUDENT_FILES:
        p = os.path.join(d, f)
        out[f] = None
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                out[f] = hashlib.sha256(fh.read()).hexdigest()
    return out


def load_starter_history(path=None):
    """{lab: [{"recorded": ..., "files": {file: sha256 or None}}, ...]} from starter-history.json."""
    try:
        with open(path or STARTER_HISTORY, encoding="utf-8") as fh:
            return json.load(fh).get("labs", {})
    except (OSError, ValueError):
        return {}


def work_detail(lab, labs_root=None, history=None):
    """How work/ relates to the starters: 'current' (byte-identical to starter/), 'earlier'
    (identical to a version of starter/ recorded in starter-history.json, so a corrected
    starter has not reached it: `reset` brings it), or 'edited'."""
    base = os.path.join(labs_root, lab) if labs_root else lab_dir(lab)
    mine = file_hashes(os.path.join(base, "work"))
    if mine == file_hashes(os.path.join(base, "starter")):
        return "current"
    versions = (load_starter_history() if history is None else history).get(lab, [])
    if any({f: v["files"].get(f) for f in STUDENT_FILES} == mine for v in versions):
        return "earlier"
    return "edited"


def work_state(lab, labs_root=None, history=None):
    """'untouched' if work/ is byte-identical to starter/ or to an earlier recorded starter
    (see work_detail), else 'edited'. These two values are what reports, runs/*.json and the
    app read; work_detail() tells the two kinds of untouched apart."""
    return "edited" if work_detail(lab, labs_root, history) == "edited" else "untouched"


def earlier_starter_note(lab):
    return ("labs/%s/work is an earlier version of the starter, unchanged by you; the starter has since been "
            "corrected. `python3 labs/pmpp.py reset %s --yes` brings the corrected starter." % (lab, lab))


def all_labs(labs_root=None):
    """Every lab directory with a starter/, as 'chNN/labM_name'."""
    root, out = labs_root or LABS, []
    for ch in sorted(os.listdir(root)):
        d = os.path.join(root, ch)
        if not os.path.isdir(d) or ch.startswith(("build", ".", "_")):
            continue
        for name in sorted(os.listdir(d)):
            if os.path.isdir(os.path.join(d, name, "starter")):
                out.append(ch + "/" + name)
    return out


def starters_missing(history=None, labs_root=None):
    """Labs whose current starter/ is not in the history (it must be recorded before the
    next edit, or untouched copies of it would read as 'edited' afterwards)."""
    history = load_starter_history() if history is None else history
    root = labs_root or LABS
    return [lab for lab in all_labs(root)
            if not any(v["files"] == file_hashes(os.path.join(root, lab, "starter")) for v in history.get(lab, []))]


def cmd_starters(a):
    """Check, or append to, labs/starter-history.json."""
    missing = starters_missing()
    if a.record:
        try:
            with open(STARTER_HISTORY, encoding="utf-8") as fh:
                doc = json.load(fh)
        except OSError:
            doc = {"labs": {}}
        today = datetime.date.today().isoformat()
        for lab in missing:
            doc["labs"].setdefault(lab, []).append(
                {"recorded": today, "files": file_hashes(os.path.join(lab_dir(lab), "starter"))})
            print("recorded the current starter of %s" % lab)
        doc["labs"] = {k: doc["labs"][k] for k in sorted(doc["labs"])}
        with open(STARTER_HISTORY, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
            fh.write("\n")
        print("%d starter version%s recorded; %d labs in labs/starter-history.json" % (
            len(missing), "" if len(missing) == 1 else "s", len(doc["labs"])))
        return 0
    for lab in missing:
        print("not recorded: the current starter of %s (run `python3 labs/pmpp.py starters --record`)" % lab)
    earlier = [lab for lab in all_labs() if work_detail(lab) == "earlier"]
    for lab in earlier:
        print(earlier_starter_note(lab))
    print("starter history: %s" % ("every current starter is recorded" if not missing
                                   else "%d current starter%s not recorded" % (len(missing), "" if len(missing) == 1 else "s")))
    return 1 if missing else 0


# What a lab's build reads. Files with these endings (and CMakeLists.txt) count as sources;
# READMEs, notes and data do not.
SOURCE_EXT = (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp", ".hh", ".inc", ".mm", ".m",
              ".metal", ".cmake", ".py")
VARIANTS = ("work", "solution", "starter")


def _is_source(name):
    return name == "CMakeLists.txt" or name.endswith(SOURCE_EXT)


def _under(path, root):
    root = os.path.abspath(root)
    return path == root or path.startswith(root + os.sep)


def _dir_sources(d):
    """The source files directly in directory d (not its subdirectories)."""
    try:
        return [os.path.join(d, f) for f in os.listdir(d)
                if _is_source(f) and os.path.isfile(os.path.join(d, f))]
    except OSError:
        return []


def _depfile_paths(path):
    """Prerequisites listed in a Makefile-style dependency file (`obj: a.h b.h ...`)."""
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    text = text.replace("\\\n", " ").replace("\\ ", "\0")
    out = []
    for i, tok in enumerate(text.split()):
        tok = tok.replace("\0", " ")
        if tok.endswith(":") and i == 0:
            continue
        if tok != ":":
            out.append(tok)
    return out


def read_inputs(target):
    """<build>/pmpp_inputs/<target>.txt (see cmake/Lab.cmake), or None for an older build."""
    try:
        lines = open(os.path.join(BUILD, "pmpp_inputs", target + ".txt")).read().splitlines()
    except OSError:
        return None
    rec = {}
    for ln in lines:
        k, _, v = ln.partition("=")
        rec[k] = [x for x in v.split("|") if x] if k != "binary_dir" else v
    return rec


_DEF_PATH = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*="?(/[^"]*)"?$')


def target_inputs(t):
    """(every file the target's build reads that lives in the repository or the build
    directory, the subset that are the lab's own sources). The second set excludes the
    shared runtime (labs/common) and files derived by the build (emulator translations).

    Covered: the lab directory (the variant's files, main.cpp, headers, provided/ and its
    CMake, but not the other variants); every source CMake compiles into the target,
    including provided copies made at configure time; the source files in the directory of
    each of those (so a provided kernel's .metal, compiled at run time, is covered); every
    absolute path the target's compile definitions name (PMPP_METAL_SHARED_DIR, the earlier
    labs' kernel.metal a milestone lab loads) and the source files beside it; and every
    header the compiler's dependency files list (Makefile generators keep them; with Ninja
    this part is missing). Not covered: the toolchain and system headers."""
    lab = t["lab"]
    if lab.startswith("selftest/"):
        labdir = os.path.join(LABS, "selftest")
    else:
        labdir = lab_dir(lab)
    files, own = set(), set()
    for root, dirs, names in os.walk(labdir):
        if os.path.abspath(root) == os.path.abspath(labdir):
            dirs[:] = [d for d in dirs if d not in VARIANTS or d == t["variant"]]
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"
                   and not d.startswith("build")]
        for n in names:
            if _is_source(n):
                files.add(os.path.join(root, n))
    own |= files
    rec = read_inputs(t["target"])
    if rec is None:
        return files, own
    common = os.path.join(LABS, "common")
    scanned = set()

    def add(path, is_own):
        path = os.path.abspath(path)
        if not (_under(path, REPO) or _under(path, BUILD)) or not os.path.exists(path):
            return
        if os.path.isdir(path):
            # A definition naming the whole labs tree (PMPP_LABS_DIR) says nothing about
            # which of its files the lab reads; the dependency files do.
            if path in (os.path.abspath(REPO), os.path.abspath(LABS)):
                return
            group = _dir_sources(path)
        else:
            # Only sources: a test model or a report a definition names is data. (Lab X3.2's
            # generated models are covered by their checksums, which are definitions.)
            group = [path] if _is_source(os.path.basename(path)) else []
            d = os.path.dirname(path)
            if not _under(d, BUILD) and d not in scanned:
                scanned.add(d)
                group += _dir_sources(d)
        for f in group:
            files.add(f)
            if is_own and not _under(f, common):
                own.add(f)

    for src in rec.get("sources", []):
        # The build's own derivatives (<target>.kernel.cpp from cu2emu, <target>.kernel.hip,
        # <target>.main.mm) are hashed but are not the lab's sources; copies of provided
        # pieces made at configure time are both.
        src = os.path.abspath(src)
        derived = _under(src, BUILD) and os.path.basename(src).startswith(t["target"] + ".")
        add(src, not derived)
        if derived and src.endswith(".cpp"):
            # cu2emu's translation starts with `#line 1 "<the .cu it came from>"`.
            try:
                with open(src, encoding="utf-8", errors="replace") as fh:
                    m = re.match(r'#line 1 "([^"]+)"', fh.readline())
            except OSError:
                m = None
            if m:
                add(m.group(1), True)
    for d in rec.get("definitions", []):
        m = _DEF_PATH.match(d)
        if m:
            add(m.group(1), True)
    bindir = rec.get("binary_dir")
    if bindir:
        tdir = os.path.join(bindir, "CMakeFiles", t["target"] + ".dir")
        for root, _, names in os.walk(tdir):
            for n in names:
                if n.endswith(".d"):
                    for dep in _depfile_paths(os.path.join(root, n)):
                        dep = os.path.abspath(dep if os.path.isabs(dep) else os.path.join(BUILD, dep))
                        if (_under(dep, REPO) or _under(dep, BUILD)) and os.path.isfile(dep):
                            files.add(dep)
                            if not _under(dep, common) and not _under(dep, BUILD):
                                own.add(dep)
    return files, own


def _portable(path):
    """A path that is the same on every checkout and build directory."""
    if _under(path, BUILD):
        return "<build>/" + os.path.relpath(path, BUILD)
    return os.path.relpath(path, REPO)


def source_hash(t, inputs=None):
    """Hash of what a build of this target read: see target_inputs(). Each file is hashed
    with its path; the compile definitions are hashed too, so a configure option that
    changes a define (PMPP_CMAKE_ARGS) changes the hash. Lets a benchmark-only run keep the
    last test result only when nothing it was built from has changed."""
    files = (inputs or target_inputs(t))[0]
    h = hashlib.sha256()
    h.update(b"v2\0")
    for f in sorted(files, key=_portable):
        try:
            data = open(f, "rb").read()
        except OSError:
            continue
        h.update(_portable(f).encode() + b"\0" + data + b"\0")
    rec = read_inputs(t["target"]) or {}
    for d in rec.get("definitions", []):
        m = _DEF_PATH.match(d)
        h.update((d if not m else d.replace(m.group(1), _portable(os.path.abspath(m.group(1))))).encode() + b"\0")
    return h.hexdigest()[:16]


def resolve_lab(cfg, token):
    labs = sorted({t["lab"] for t in cfg["targets"] if t["lab"] != "devinfo"})
    m = re.fullmatch(r"(\d+)\.(\d+)", token)
    a = re.fullmatch(r"([A-Za-z])\.(\d+)", token)      # appendix labs: A.1 -> appA/lab1_
    x = re.fullmatch(r"[Xx](\d+)\.(\d+)", token)       # our modules: X1.3 -> x1/lab3_
    want = None
    if m:
        want = "ch%02d/lab%s_" % (int(m.group(1)), m.group(2))
        hits = [l for l in labs if l.startswith(want)]
    elif a:
        want = "app%s/lab%s_" % (a.group(1).upper(), a.group(2))
        hits = [l for l in labs if l.startswith(want)]
    elif x:
        want = "x%s/lab%s_" % (x.group(1), x.group(2))
        hits = [l for l in labs if l.startswith(want)]
    else:
        hits = [l for l in labs if token in l]
    if len(hits) != 1:
        if not hits:
            # A lab pmpp_add_lab skipped because a file it needs is missing says so.
            inc = [i for i in cfg.get("incomplete", []) if (i["lab"].startswith(want) if want else token in i["lab"])]
            if len(inc) == 1:
                sys.exit("lab %s is not built: it is incomplete (missing %s)" % (inc[0]["lab"], scrub(inc[0]["missing"])))
            if os.environ.get("PMPP_ONLY"):
                sys.exit("lab '%s' matches nothing among the labs of PMPP_ONLY=%s: %s" % (
                    token, os.environ["PMPP_ONLY"], ", ".join(labs)))
        sys.exit("lab '%s' matches %s. Known labs: %s" % (token, hits or "nothing", ", ".join(labs)))
    return hits[0]


# ---------------------------------------------------------------- device info
def device_info(cfg):
    info = {}
    for b in cfg["backends"]:
        if not b["enabled"]:
            continue
        t = next((t for t in cfg["targets"] if t["lab"] == "devinfo" and t["backend"] == b["name"]), None)
        if not t:
            continue
        ok, out = build(t["target"])
        if not ok:
            info[b["name"]] = {"available": False, "why": "devinfo failed to build", "log": tail(out, 40)}
            continue
        rc, rows, text = run_exe(t["exe"], [], 120)
        dev = next((r for r in rows if r.get("kind") == "device"), None)
        if dev is None:
            info[b["name"]] = {"available": False, "why": "devinfo crashed (exit %d)" % rc, "log": tail(text, 40)}
        else:
            info[b["name"]] = dev
    return info


# ---------------------------------------------------------------- CUDA front-end check
def cuda_frontend_check():
    """Parse every .cu file with clang's CUDA front end against stub declarations.
    Catches syntax errors, bad launches and host/device call violations. It does
    not prove nvcc accepts the file, and nothing is run."""
    clang = os.environ.get("PMPP_CLANG") or shutil.which("clang++")
    res = {"available": False, "why": "", "files": []}
    if not clang:
        res["why"] = "no clang++ on PATH"
        return res
    inc = os.path.join(LABS, "common", "include")
    # The second include directory holds stand-ins for <cuda/atomic>, <cooperative_groups.h>
    # and <cuda_runtime.h> that switch on the matching sections of the stub header.
    base = [clang, "-x", "cuda", "--cuda-host-only", "-nocudainc", "-nocudalib", "-fsyntax-only",
            "-std=c++17", "-I", inc, "-I", os.path.join(inc, "pmpp", "emu", "shim"),
            "-include", "pmpp/check/cuda_stub.h", "-DTILE_WIDTH=16"]
    with tempfile.TemporaryDirectory() as d:
        probe = os.path.join(d, "probe.cu")
        open(probe, "w").write("__global__ void k(float* a) { a[threadIdx.x] = 0; }\n"
                               "void f(float* a) { k<<<1, 4>>>(a); }\n")
        rc, out = sh(base + [probe], timeout=60)
    if rc != 0:
        res["why"] = "this clang++ cannot parse CUDA: " + tail(out, 3)
        return res
    res["available"] = True
    res["compiler"] = sh([clang, "--version"])[1].splitlines()[0]
    for root, dirs, files in os.walk(LABS):
        # Skip every build directory (labs/build, labs/build-*, $PMPP_BUILD_DIR): they hold
        # translated or stale copies of the sources, not the sources themselves.
        dirs[:] = [d for d in dirs if not (root == LABS and d.startswith("build"))
                   and not os.path.abspath(os.path.join(root, d)).startswith(os.path.abspath(BUILD))]
        for f in sorted(files):
            if f.endswith(".cu"):
                path = os.path.join(root, f)
                rc, out = sh(base + [path], timeout=120)
                res["files"].append({"file": os.path.relpath(path, REPO), "ok": rc == 0,
                                     "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                                     "log": "" if rc == 0 else scrub(tail(out, 25))})
    res["files"].sort(key=lambda x: x["file"])
    return res


FRONTEND_RECORD = os.path.join(REPORTS, "runs", "checks", "cuda-frontend.json")


def save_frontend_check(res):
    """Record a front-end check for the app (GET /api/frontend-check): per file, the verdict
    and the sha256 of the text that was checked, so a file edited since shows no status.
    Per machine, like the rest of reports/runs."""
    if not res.get("available"):
        return
    rec = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "compiler": res.get("compiler", ""),
           "files": {f["file"]: {"ok": f["ok"], "sha256": f["sha256"]} for f in res["files"]}}
    os.makedirs(os.path.dirname(FRONTEND_RECORD), exist_ok=True)
    tmp = FRONTEND_RECORD + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, FRONTEND_RECORD)


def cmd_frontend(_):
    res = cuda_frontend_check()
    if not res["available"]:
        print("CUDA front-end check not available: " + res["why"])
        return 1
    save_frontend_check(res)
    bad = [f for f in res["files"] if not f["ok"]]
    for f in bad:
        print("FAIL %s\n%s" % (f["file"], f["log"]))
    print("CUDA front-end check: %d of %d .cu files parse (%s); recorded in %s" % (
        len(res["files"]) - len(bad), len(res["files"]), res.get("compiler", "clang"), os.path.relpath(FRONTEND_RECORD, REPO)))
    return 1 if bad else 0


# ---------------------------------------------------------------- running labs
def exercise(t, do_test, do_bench, timeout=None, reps=1, full=False):
    """Build one target, then test and/or benchmark it. With reps > 1 the benchmark runs
    that many times and each repetition is kept (bench_reps); `bench` holds the first.
    Sanitizer reports are condensed unless full is set (see condense_sanitizer).
    The test timeout is 900 s on devices and 2400 s on the emulators, which run every
    thread in turn (lab X3.2's suite needs more than 900 s there on a busy machine);
    PMPP_TEST_TIMEOUT (seconds) overrides both."""
    if timeout is None:
        timeout = int(os.environ.get("PMPP_TEST_TIMEOUT") or 0) or (2400 if t["backend"] in EMULATED else 900)
    r = dict(t)
    r.update({"built": False, "tests": [], "bench": [], "test_rc": None, "log": "", "tsan": []})
    ok, out = build(t["target"])
    r["built"] = ok
    # Hashed after the build, when the compiler's dependency files list every header read.
    inputs = target_inputs(t) if t["variant"] in VARIANTS + ("selftest",) else None
    if inputs and t["variant"] in VARIANTS:
        r["src"] = source_hash(t, inputs)
    if not ok:
        r["log"] = scrub(tail(out, 60))
        return r
    own_names = {os.path.basename(f) for f in (inputs[1] if inputs else ())}

    def readable(text):
        return text if full else condense_sanitizer(text, own_names)
    if do_test:
        rc, rows, raw = run_exe(t["exe"], ["--test"], timeout)
        # A ThreadSanitizer report (backend `race`) is a failure in its own right,
        # whatever the tests printed; keep its key lines and drop the rest of it.
        text, tsan = split_tsan(raw)
        text = raw if full else readable(text)
        r["test_rc"] = rc
        r["tests"] = [x for x in rows if x.get("kind") == "test"]
        r["tsan"] = [[scrub(l) for l in rep] for rep in tsan]
        r["test_text"] = scrub(text)
        if rc != 0 or tsan:
            r["log"] = scrub(tail(text, 60))
            if tsan:
                r["log"] += "\n\nThreadSanitizer reported %d data race(s). Key lines:\n%s" % (len(tsan), tsan_text(r["tsan"]))
    if do_bench and (not do_test or r["test_rc"] == 0):
        runs, texts, bench_rc = [], [], 0
        for i in range(max(1, reps)):
            at = datetime.datetime.now().isoformat(timespec="seconds")
            rc, rows, text = run_exe(t["exe"], ["--bench"], timeout)
            runs.append({"rep": i + 1, "at": at, "rc": rc, "rows": [x for x in rows if x.get("kind") == "bench"]})
            head = "-- benchmark, repetition %d of %d (%s)\n" % (i + 1, reps, at) if reps > 1 else ""
            texts.append(head + scrub(readable(text)))
            if rc != 0:   # a repetition that crashed: keep what ran, stop repeating
                bench_rc = rc
                break
        r["bench_rc"] = bench_rc
        r["bench"] = runs[0]["rows"]
        r["bench_text"] = "\n".join(texts)
        if reps > 1:
            r["bench_reps"] = runs
    return r


def passed(r):
    return (r["built"] and r["test_rc"] == 0 and r["tests"] and all(x["pass"] for x in r["tests"])
            and not r.get("tsan"))


def selftest_verdict(r):
    """(ok, text). A self-test marked expect=race is a planted data race: it is
    right only if ThreadSanitizer reported it and every one of its cases failed."""
    if not r["built"]:
        return False, "build failed"
    if r.get("expect") == "race":
        ok = bool(r["tsan"]) and bool(r["tests"]) and not any(x["pass"] for x in r["tests"])
        return ok, "race reported, as it must be" if ok else "planted race NOT reported"
    return (True, "pass") if passed(r) else (False, "FAIL")


def save_runs(results, states, devs=None):
    """Merge results into reports/runs/<backend>.json, which the app reads.
    A result replaces the earlier one for the same lab, variant and tile width."""
    runs = os.path.join(REPORTS, "runs")
    os.makedirs(runs, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    # Several pmpp.py processes may run at once (parallel builds, or the app's reader).
    # Hold an exclusive lock across each read-merge-write, and replace the file atomically,
    # so no run ever reads a half-written file or overwrites another run's merge.
    with open(os.path.join(runs, ".lock"), "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        for b in sorted({r["backend"] for r in results}):
            _merge_runs(os.path.join(runs, "%s.json" % b), b, results, states, devs, now)


def _merge_runs(path, b, results, states, devs, now):
        try:
            data = json.load(open(path))
        except OSError:
            data = {"device": {}, "results": [], "work_state": {}}
        except ValueError:
            # Unreadable: keep it for inspection instead of silently starting from nothing.
            shutil.copyfile(path, path + ".corrupt")
            print("warning: %s was unreadable; saved as %s.corrupt and started afresh" % (path, path), file=sys.stderr)
            data = {"device": {}, "results": [], "work_state": {}}
        new = [r for r in results if r["backend"] == b]
        key = lambda r: (r["lab"], r["variant"], r["tile"])
        fresh = {key(r) for r in new}
        # A run without benchmarks (--quick, or plain `run`) keeps the last measurements
        # for a kernel that still passes, marked with when they were taken.
        old = {key(r): r for r in data.get("results", [])}
        for r in new:
            prev = old.get(key(r))
            if not r.get("bench") and prev and prev.get("bench") and passed(r):
                r["bench"] = prev["bench"]
                if prev.get("bench_reps"):
                    r["bench_reps"] = prev["bench_reps"]
                r["bench_from"] = prev.get("bench_from") or data.get("generated")
            # Likewise a benchmark-only run (--no-test) keeps the last test result, but only
            # when the sources it compiled are the ones that result was tested on.
            if (r["built"] and r["test_rc"] is None and prev and prev.get("test_rc") is not None
                    and r.get("src") and prev.get("src") == r.get("src")):
                for k in ("tests", "test_rc", "tsan", "test_text"):
                    if k in prev:
                        r[k] = prev[k]
                r["tests_from"] = prev.get("tests_from") or data.get("generated")
        data["results"] = [r for r in data.get("results", []) if key(r) not in fresh] + new
        data["results"].sort(key=lambda r: (r["lab"], r["variant"], r["tile"]))
        data["generated"] = now
        data.setdefault("work_state", {}).update(states)
        if devs and b in devs:
            data["device"] = devs[b]
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".%s." % b, suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)


def counts(r):
    if not r["built"]:
        return "build failed"
    if r["test_rc"] is None:
        return "not run"
    n = len(r["tests"])
    p = sum(1 for x in r["tests"] if x["pass"])
    if r.get("tsan"):
        extra = " (ThreadSanitizer: %d data race report%s)" % (len(r["tsan"]), "" if len(r["tsan"]) == 1 else "s")
    else:
        extra = "" if r["test_rc"] in (0, 1) else " (exit %d: crashed or aborted)" % r["test_rc"]
    return "%d/%d%s" % (p, n, extra)


# ---------------------------------------------------------------- commands
# What a few device properties mean, from the vendors' references (notes/today/infra-facts.md
# F1, ch23-facts A7 and B3, ch22-facts), printed beside the value by `doctor`. Keyed by
# property; a (backend, property) entry overrides it for that backend.
PROPERTY_NOTES = {
    "asyncEngineCount": "number of copy engines: 1 lets one copy overlap kernels, 2 lets a copy each way overlap them",
    ("hip", "asyncEngineCount"): "on ROCm a runtime constant, not a count of copy engines (ch23-facts B3)",
    ("emu", "asyncEngineCount"): "the emulator runs streams synchronously: nothing overlaps",
    "concurrentKernels": "1 if the device can possibly run several kernels at once",
    ("emu", "concurrentKernels"): "the emulator runs one kernel at a time",
}


def cmd_doctor(_):
    cfg = configure()
    print("build directory: %s" % BUILD)
    if cfg["cmake_args"]:
        print("extra cmake arguments (PMPP_CMAKE_ARGS): %s" % " ".join(cfg["cmake_args"]))
    for b in cfg["backends"]:
        print("  %-6s %-8s %s" % (b["name"], "enabled" if b["enabled"] else "skipped", b["why"]))
    if cfg["incomplete"]:
        print("labs skipped as incomplete: %s" % ", ".join(i["lab"] for i in cfg["incomplete"]))
    earlier = [lab for lab in all_labs() if work_detail(lab) == "earlier"]
    if earlier:
        print("\nyour work/ is still an earlier, since corrected, starter in: %s\n"
              "  `python3 labs/pmpp.py reset <lab> --yes` brings the corrected starter (you have not edited these)."
              % ", ".join(earlier))
    for name, d in device_info(cfg).items():
        print("\n%s device: %s" % (name, d.get("name", d.get("why", "?"))))
        for k, v in d.items():
            if k not in ("kind", "backend", "name", "available", "log"):
                note = PROPERTY_NOTES.get((name, k), PROPERTY_NOTES.get(k))
                print("  %-34s %s%s" % (k, v, "   (%s)" % note if note else ""))
    if cfg["rc"] != 0:
        print("\ncmake configure failed:\n" + tail(cfg["log"], 30))
        return 1
    return 0


def cmd_run(a):
    cfg = configure()
    if cfg["rc"] != 0:
        sys.exit("cmake configure failed:\n" + tail(cfg["log"], 30))
    if a.reps < 1:
        sys.exit("--reps must be at least 1")
    if a.reps > 1 and not a.bench:
        sys.exit("--reps repeats the benchmark: use it with --bench")
    lab = resolve_lab(cfg, a.lab)
    variant = "solution" if a.solution else "work"
    enabled = [b["name"] for b in cfg["backends"] if b["enabled"]]
    want = enabled if a.backend == "all" else [a.backend]
    for b in want:
        if b not in enabled:
            why = next((x["why"] for x in cfg["backends"] if x["name"] == b), "unknown backend")
            print("\n== %s: skipped (%s)" % (b, why))
    rc_all = 0
    ran = []
    # A backend this machine has but this lab has no build for says why (no kernel.metal,
    # no RACE, a library missing here) instead of running nothing in silence.
    for b in want:
        if b in enabled and not any(t["lab"] == lab and t["variant"] == variant and t["backend"] == b
                                    for t in cfg["targets"]):
            whys = [x["why"] for x in cfg["skipped"]
                    if x["lab"] == lab and x["variant"] == variant and x["backend"] == b]
            print("\n== %s: %s" % (b, whys[0] if whys else "no %s build of this lab's %s was configured" % (b, variant)))
    for t in cfg["targets"]:
        if t["lab"] != lab or t["variant"] != variant or t["backend"] not in want:
            continue
        is_extra_tile = t["target"].split(".")[-1].startswith("t") and t["target"].count(".") == 3
        if is_extra_tile and not a.bench:
            continue
        print("\n== %s, %s, %s%s" % (lab, variant, t["backend"], ", TILE_WIDTH=%d" % t["tile"] if is_extra_tile else ""))
        r = exercise(t, not a.no_test and not is_extra_tile, a.bench and t["backend"] not in EMULATED,
                     reps=a.reps, full=a.full)
        if not r["built"]:
            print("BUILD FAILED\n" + r["log"])
            rc_all = 1
            continue
        if r.get("test_text"):
            print(r["test_text"])
        if r.get("tsan"):
            print("\nThreadSanitizer reported %d data race(s): two emulated threads touched the same memory "
                  "without synchronisation.\n%s" % (len(r["tsan"]), tsan_text(r["tsan"])))
        if r.get("bench_text"):
            print(r["bench_text"])
        if r["test_rc"] not in (None, 0) or r.get("tsan"):
            rc_all = 1
        ran.append(r)
    if ran:   # so the app shows what you just ran
        have = {b: os.path.exists(os.path.join(REPORTS, "runs", "%s.json" % b)) for b in {r["backend"] for r in ran}}
        devs = device_info(cfg) if not all(have.values()) else None
        save_runs(ran, {lab: work_state(lab)}, devs)
        if variant == "work" and work_detail(lab) == "earlier":
            print("\nnote: " + earlier_starter_note(lab))
    elif rc_all == 0:
        print("\nnothing was run: %s has no build for %s on this machine" % (
            lab, "any backend found" if a.backend == "all" else "backend " + a.backend))
        return 1
    return rc_all


def run_selftests(cfg, backends, only=None, echo=True, full=False):
    """Build and run the emulator's self-tests (labs/selftest) on the given backends."""
    results = []
    for t in cfg["selftests"]:
        if t["backend"] not in backends or (only and only not in t["lab"]):
            continue
        if echo:
            print("[%s] %s ..." % (t["backend"], t["target"]), flush=True)
        r = exercise(t, True, False, full=full)
        ok, verdict = selftest_verdict(r)
        r["ok"], r["verdict"] = ok, verdict
        if echo:
            print("    tests: %s -> %s" % (counts(r), verdict), flush=True)
        results.append(r)
    return results


def cmd_selftest(a):
    cfg = configure()
    if cfg["rc"] != 0:
        sys.exit("cmake configure failed:\n" + tail(cfg["log"], 30))
    enabled = [b["name"] for b in cfg["backends"] if b["enabled"]]
    want = enabled if a.backend == "all" else [a.backend]
    for b in want:
        if b not in enabled:
            why = next((x["why"] for x in cfg["backends"] if x["name"] == b), "unknown backend")
            print("== %s: skipped (%s)" % (b, why))
    results = run_selftests(cfg, want, a.name, full=a.full)
    for r in results:
        if not r["ok"] or a.verbose:
            print("\n== %s: %s" % (r["target"], r["verdict"]))
            print(r["log"] if not r["built"] else r.get("test_text", ""))
        if r["tsan"] and (not r["ok"] or a.verbose or r.get("expect") == "race"):
            print("\n== %s: ThreadSanitizer key lines\n%s" % (r["target"], tsan_text(r["tsan"])))
    bad = [r for r in results if not r["ok"]]
    print("\nself-tests: %d ran, %d failed" % (len(results), len(bad)))
    return 1 if bad or not results else 0


def cmd_reset(a):
    cfg = configure()
    lab = resolve_lab(cfg, a.lab)
    if not a.yes:
        sys.exit("This overwrites your work in labs/%s/work with the starter. Re-run with --yes to confirm." % lab)
    for f in STUDENT_FILES:
        src = os.path.join(lab_dir(lab), "starter", f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(lab_dir(lab), "work", f))
            print("reset labs/%s/work/%s" % (lab, f))
    return 0


def cmd_verify(a):
    if a.reps < 1:
        sys.exit("--reps must be at least 1")
    started = datetime.datetime.now()
    cfg = configure()
    devs = device_info(cfg) if cfg["rc"] == 0 else {}
    usable = [b["name"] for b in cfg["backends"] if b["enabled"] and devs.get(b["name"], {}).get("available")]
    results = []
    labs = sorted({t["lab"] for t in cfg["targets"] if t["lab"] != "devinfo"})
    states = {lab: work_state(lab) for lab in labs}
    earlier = [lab for lab in labs if work_detail(lab) == "earlier"]
    for t in cfg["targets"]:
        if t["lab"] == "devinfo" or t["backend"] not in usable:
            continue
        extra_tile = t["target"].count(".") == 3
        if t["variant"] == "solution":
            do_test, do_bench = True, (t["backend"] not in EMULATED and not a.quick)
        else:  # your work
            edited = states[t["lab"]] == "edited"
            if extra_tile and not edited:
                continue
            do_test, do_bench = True, (edited and t["backend"] not in EMULATED and not a.quick)
        print("[%s] %s ..." % (t["backend"], t["target"]), flush=True)
        r = exercise(t, do_test, do_bench, reps=a.reps, full=a.full)
        print("    tests: %s%s" % (counts(r), ", %d benchmark rows" % len(r["bench"]) if r["bench"] else ""), flush=True)
        results.append(r)
    selftests = run_selftests(cfg, usable, full=a.full)
    check = cuda_frontend_check()
    save_frontend_check(check)

    report = {
        "generated": started.isoformat(timespec="seconds"),
        "system": {"os": platform.platform(), "machine": platform.machine(),
                   "python": platform.python_version(),
                   "cmake": sh(["cmake", "--version"])[1].splitlines()[0]},
        "configure_ok": cfg["rc"] == 0,
        "configure_log": scrub(tail(cfg["log"], 60)),
        "backends": cfg["backends"], "devices": devs, "usable_backends": usable,
        "work_state": states, "work_earlier_starter": earlier, "results": results, "selftests": selftests, "cuda_frontend_check": check,
        "skipped": [x for x in cfg.get("skipped", []) if x["backend"] in usable and not x["why"].startswith(
            "this lab is not built for the race")],
        "incomplete": cfg.get("incomplete", []), "cmake_args": cfg.get("cmake_args", []), "reps": a.reps,
    }
    os.makedirs(REPORTS, exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    tag = "%s-%s" % (platform.system().lower(), "+".join(usable) or "none")
    base = os.path.join(REPORTS, "verify-%s-%s" % (stamp, tag))
    json.dump(report, open(base + ".json", "w"), indent=1)
    open(base + ".md", "w").write(render(report))
    # Latest measurements per backend, for the app to display.
    save_runs(results, states, devs)
    bad = [r for r in results if r["variant"] == "solution" and not passed(r)]
    print("\nreport: %s" % os.path.relpath(base + ".md", REPO))
    print("reference solutions: %d ran, %d failed" % (sum(1 for r in results if r["variant"] == "solution"), len(bad)))
    bad_self = [r for r in selftests if not r["ok"]]
    print("emulator self-tests: %d ran, %d failed" % (len(selftests), len(bad_self)))
    return 1 if (bad or bad_self or cfg["rc"] != 0) else 0


# ---------------------------------------------------------------- report
def render(rep):
    L = []
    w = L.append
    res = rep["results"]
    sols = [r for r in res if r["variant"] == "solution"]
    work = [r for r in res if r["variant"] == "work"]
    bad = [r for r in sols if not passed(r)]
    w("# PMPP labs verification report")
    w("")
    w("- Generated: %s" % rep["generated"])
    w("- System: %s (%s), Python %s, %s" % (rep["system"]["os"], rep["system"]["machine"],
                                            rep["system"]["python"], rep["system"]["cmake"]))
    w("- Backends with a usable device: **%s**" % (", ".join(rep["usable_backends"]) or "none"))
    w("- Reference solutions: **%d of %d passed**" % (len(sols) - len(bad), len(sols)))
    if rep.get("cmake_args"):
        w("- Extra cmake arguments (PMPP_CMAKE_ARGS): `%s`" % " ".join(rep["cmake_args"]))
    if rep.get("reps", 1) > 1:
        w("- Each benchmark was run %d times; every repetition is listed" % rep["reps"])
    w("")
    w("## What this report does and does not establish")
    w("")
    w("- A reference solution that passes on `metal`, `cuda` or `hip` was compiled and run on that")
    w("  real device and matched a CPU reference. Nothing else counts as verified on that backend.")
    w("- `emu` is a CPU emulator of the CUDA subset. It checks kernel logic (indexing, shared memory,")
    w("  barriers, bounds) with deterministic thread schedules and AddressSanitizer. It says nothing")
    w("  about performance, and passing it does not prove a kernel runs on a GPU.")
    w("- `race` is the same emulator with its threads running concurrently under ThreadSanitizer. A data")
    w("  race it reports is real for the schedule that ran; a clean run does not prove there is none. It")
    w("  has no bounds checking, and it too says nothing about performance. Labs opt in to it.")
    w("- The CUDA front-end check parses `.cu` files with clang against hand-written stub headers.")
    w("  It is a syntax and host/device-rule check only.")
    w("- Benchmark numbers are from this machine at this moment. They are not expectations.")
    w("")
    w("## Backends")
    w("")
    w("| Backend | Built | Device | Reason / device name |")
    w("|---|---|---|---|")
    for b in rep["backends"]:
        d = rep["devices"].get(b["name"], {})
        dev = "yes" if d.get("available") else ("no" if b["enabled"] else "-")
        note = d.get("name") if d.get("available") else (d.get("why") or b["why"])
        w("| %s | %s | %s | %s |" % (b["name"], "yes" if b["enabled"] else "skipped", dev, note))
    w("")
    if not rep["configure_ok"]:
        w("**cmake configure failed.** Log tail:")
        w("")
        w("```\n%s\n```" % rep["configure_log"])
        w("")
    for name, d in rep["devices"].items():
        if not d.get("available"):
            continue
        w("### Device properties reported by `%s`" % name)
        w("")
        w("| Property | Value |")
        w("|---|---|")
        for k, v in d.items():
            if k not in ("kind", "backend", "available"):
                w("| %s | %s |" % (k, v))
        w("")
    if rep.get("skipped") or rep.get("incomplete"):
        w("## Builds not made")
        w("")
        w("| Lab | Variant | Backend | Why |")
        w("|---|---|---|---|")
        for x in rep.get("incomplete", []):
            w("| %s | all | all | incomplete: missing %s |" % (x["lab"], scrub(x["missing"])))
        for x in rep.get("skipped", []):
            w("| %s | %s | %s | %s |" % (x["lab"], x["variant"], x["backend"], x["why"]))
        w("")
    w("## Reference solutions")
    w("")
    w("| Lab | Backend | Tile | Build | Tests passed | Verdict |")
    w("|---|---|---|---|---|---|")
    for r in sols:
        w("| %s | %s | %d | %s | %s | %s |" % (r["lab"], r["backend"], r["tile"], "ok" if r["built"] else "FAILED",
                                              counts(r), "pass" if passed(r) else "**FAIL**"))
    w("")
    w("## Starters (negative control) and your work")
    w("")
    w("An untouched starter must build and must fail its tests; if one passes, the tests are too weak.")
    w("")
    w("| Lab | Backend | State of work/ | Build | Tests passed | Reading |")
    w("|---|---|---|---|---|---|")
    for r in work:
        state = rep["work_state"].get(r["lab"], "?")
        if state == "untouched":
            reading = "starter fails as it should" if (r["built"] and not passed(r)) else (
                "**starter does not build**" if not r["built"] else "**starter passes: tests too weak**")
        else:
            reading = "your solution passes" if passed(r) else "your solution does not pass yet"
        shown = "untouched (earlier starter)" if r["lab"] in rep.get("work_earlier_starter", []) else state
        w("| %s | %s | %s | %s | %s | %s |" % (r["lab"], r["backend"], shown, "ok" if r["built"] else "FAILED",
                                              counts(r), reading))
    w("")
    if rep.get("work_earlier_starter"):
        w("*Untouched (earlier starter)*: work/ is a version of the starter that has since been corrected; "
          "`python3 labs/pmpp.py reset <lab> --yes` brings the corrected one.")
        w("")
    st = rep.get("selftests", [])
    w("## Emulator self-tests")
    w("")
    w("These test the emulator itself (`labs/selftest`), not any lab. A row marked *must be reported* is a")
    w("planted data race: it is right only when ThreadSanitizer reports it.")
    w("")
    if not st:
        w("None were run.")
    else:
        w("| Self-test | Backend | Build | Cases passed | Expectation | Verdict |")
        w("|---|---|---|---|---|---|")
        for r in st:
            w("| %s | %s | %s | %s | %s | %s |" % (
                r["lab"].split("/", 1)[1], r["backend"], "ok" if r["built"] else "FAILED", counts(r),
                "data race *must be reported*" if r.get("expect") == "race" else "all cases pass, no race report",
                r["verdict"] if r["ok"] else "**%s**" % r["verdict"]))
        for r in st:
            if r.get("tsan") and r["ok"]:
                w("")
                w("ThreadSanitizer key lines for `%s`:" % r["target"])
                w("")
                w("```\n%s\n```" % tsan_text(r["tsan"]))
    w("")
    w("## Benchmarks (this machine, this run)")
    w("")
    any_bench = False
    for r in res:
        if not r["bench"]:
            continue
        any_bench = True
        w("### %s, %s, %s, TILE_WIDTH=%d" % (r["lab"], r["variant"], r["backend"], r["tile"]))
        w("")
        rows = r["bench"]
        keys = []
        if r.get("bench_reps"):
            rows = [dict(row, rep=x["rep"]) for x in r["bench_reps"] for row in x["rows"]]
            keys = ["rep"]
        for row in rows:
            for k in row:
                if k not in ("kind", "lab", "variant", "backend") and k not in keys:
                    keys.append(k)
        w("| " + " | ".join(keys) + " |")
        w("|" + "---|" * len(keys))
        for row in rows:
            cells = []
            for k in keys:
                v = row.get(k, "")
                cells.append(("%.4g" % v) if isinstance(v, float) else str(v))
            w("| " + " | ".join(cells) + " |")
        w("")
    if not any_bench:
        w("None were run (no real device, or `--quick`).")
        w("")
    c = rep["cuda_frontend_check"]
    w("## CUDA front-end check (clang, stub headers, nothing run)")
    w("")
    if not c["available"]:
        w("Not run: %s" % c["why"])
    else:
        w("Compiler: %s" % c.get("compiler", "?"))
        w("")
        w("| File | Parses as CUDA |")
        w("|---|---|")
        for f in c["files"]:
            w("| %s | %s |" % (f["file"], "yes" if f["ok"] else "**NO**"))
    w("")
    fails = [r for r in res if r.get("log") and (r["variant"] == "solution" or rep["work_state"].get(r["lab"]) == "edited")]
    cfails = [f for f in c.get("files", []) if not f["ok"]]
    fails += [r for r in st if not r["ok"]]
    if fails or cfails:
        w("## Failure details")
        w("")
        for r in fails:
            w("### %s (%s)" % (r["target"], "build" if not r["built"] else "tests, exit %s" % r["test_rc"]))
            w("")
            w("```\n%s\n```" % r["log"])
            w("")
        for f in cfails:
            w("### front-end check: %s" % f["file"])
            w("")
            w("```\n%s\n```" % f["log"])
            w("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    r = sub.add_parser("run")
    r.add_argument("lab", help="e.g. 5.2 or lab2_tiled")
    r.add_argument("--backend", default="all", choices=["all"] + BACKENDS)
    r.add_argument("--solution", action="store_true", help="run the reference solution instead of your work")
    r.add_argument("--bench", action="store_true")
    r.add_argument("--reps", type=int, default=1, metavar="N",
                   help="with --bench: run the benchmark N times and record each repetition")
    r.add_argument("--no-test", action="store_true")
    r.add_argument("--full", action="store_true", help="print sanitizer reports in full instead of condensed")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("selftest", help="run the emulator's own tests (labs/selftest); writes no report")
    t.add_argument("name", nargs="?", help="only self-tests whose name contains this")
    t.add_argument("--backend", default="all", choices=["all"] + BACKENDS)
    t.add_argument("--verbose", action="store_true", help="print every test's output, not only failures")
    t.add_argument("--full", action="store_true", help="print sanitizer reports in full instead of condensed")
    t.set_defaults(fn=cmd_selftest)
    v = sub.add_parser("verify")
    v.add_argument("--quick", action="store_true", help="skip benchmarks")
    v.add_argument("--reps", type=int, default=1, metavar="N", help="run each benchmark N times, recording each")
    v.add_argument("--full", action="store_true", help="keep sanitizer reports in full instead of condensed")
    v.set_defaults(fn=cmd_verify)
    f = sub.add_parser("frontend", help="parse every .cu file with clang's CUDA front end and record the result for the app")
    f.set_defaults(fn=cmd_frontend)
    h = sub.add_parser("starters", help="check that every current starter is in labs/starter-history.json")
    h.add_argument("--record", action="store_true", help="append every current starter not yet recorded")
    h.set_defaults(fn=cmd_starters)
    s = sub.add_parser("reset")
    s.add_argument("lab")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_reset)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
