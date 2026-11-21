#!/usr/bin/env python3
"""Build, test and benchmark the labs on whatever backends this machine has.

  python3 labs/pmpp.py doctor                 which backends can be built here, and why not
  python3 labs/pmpp.py run 5.2                test your work for lab 5.2 on every backend
  python3 labs/pmpp.py run 5.2 --bench        ...and benchmark it
  python3 labs/pmpp.py run 5.2 --solution     run the reference solution instead
  python3 labs/pmpp.py run 5.2 --backend emu  one backend only (emu, race, metal, cuda, hip)
  python3 labs/pmpp.py selftest               the emulator's own tests (labs/selftest), no report written
  python3 labs/pmpp.py verify                 everything, and write reports/verify-*.md
  python3 labs/pmpp.py reset 5.2 --yes        overwrite your work for a lab with the starter

Standard library only. Needs cmake >= 3.21 and a C++17 compiler.
"""
import argparse
import datetime
import fcntl
import filecmp
import hashlib
import json
import os
import platform
import re
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
    rc, out = sh(["cmake", "-S", LABS, "-B", BUILD])
    backends, targets, selftests = [], [], []
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
    return {"rc": rc, "log": out, "backends": backends, "targets": targets, "selftests": selftests}


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


def lab_dir(lab):
    return os.path.join(LABS, lab)


def work_state(lab):
    """'untouched' if work/ is byte-identical to starter/, else 'edited'."""
    w, s = os.path.join(lab_dir(lab), "work"), os.path.join(lab_dir(lab), "starter")
    for f in STUDENT_FILES:
        a, b = os.path.join(w, f), os.path.join(s, f)
        if os.path.exists(a) != os.path.exists(b):
            return "edited"
        if os.path.exists(a) and not filecmp.cmp(a, b, shallow=False):
            return "edited"
    return "untouched"


def source_hash(lab, variant):
    """Hash of what a run of this lab variant compiled: the variant's files and the driver.
    Lets a run without tests keep the last test result only when nothing has changed."""
    h = hashlib.sha256()
    d = lab_dir(lab)
    for f in [os.path.join(variant, x) for x in STUDENT_FILES] + ["main.cpp"]:
        p = os.path.join(d, f)
        if os.path.exists(p):
            h.update(f.encode() + b"\0" + open(p, "rb").read() + b"\0")
    return h.hexdigest()[:16]


def resolve_lab(cfg, token):
    labs = sorted({t["lab"] for t in cfg["targets"] if t["lab"] != "devinfo"})
    m = re.fullmatch(r"(\d+)\.(\d+)", token)
    a = re.fullmatch(r"([A-Za-z])\.(\d+)", token)      # appendix labs: A.1 -> appA/lab1_
    x = re.fullmatch(r"[Xx](\d+)\.(\d+)", token)       # our modules: X1.3 -> x1/lab3_
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
    for root, _, files in os.walk(LABS):
        if os.path.abspath(root).startswith(os.path.abspath(BUILD)):
            continue
        for f in sorted(files):
            if f.endswith(".cu"):
                path = os.path.join(root, f)
                rc, out = sh(base + [path], timeout=120)
                res["files"].append({"file": os.path.relpath(path, REPO), "ok": rc == 0,
                                     "log": "" if rc == 0 else scrub(tail(out, 25))})
    res["files"].sort(key=lambda x: x["file"])
    return res


# ---------------------------------------------------------------- running labs
def exercise(t, do_test, do_bench, timeout=900):
    """Build one target, then test and/or benchmark it."""
    r = dict(t)
    r.update({"built": False, "tests": [], "bench": [], "test_rc": None, "log": "", "tsan": []})
    if t["variant"] in ("work", "solution", "starter"):
        r["src"] = source_hash(t["lab"], t["variant"])
    ok, out = build(t["target"])
    r["built"] = ok
    if not ok:
        r["log"] = scrub(tail(out, 60))
        return r
    if do_test:
        rc, rows, text = run_exe(t["exe"], ["--test"], timeout)
        # A ThreadSanitizer report (backend `race`) is a failure in its own right,
        # whatever the tests printed; keep its key lines and drop the rest of it.
        text, tsan = split_tsan(text)
        r["test_rc"] = rc
        r["tests"] = [x for x in rows if x.get("kind") == "test"]
        r["tsan"] = [[scrub(l) for l in rep] for rep in tsan]
        r["test_text"] = scrub(text)
        if rc != 0 or tsan:
            r["log"] = scrub(tail(text, 60))
            if tsan:
                r["log"] += "\n\nThreadSanitizer reported %d data race(s). Key lines:\n%s" % (len(tsan), tsan_text(r["tsan"]))
    if do_bench and (not do_test or r["test_rc"] == 0):
        rc, rows, text = run_exe(t["exe"], ["--bench"], timeout)
        r["bench_rc"] = rc
        r["bench"] = [x for x in rows if x.get("kind") == "bench"]
        r["bench_text"] = scrub(text)
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
def cmd_doctor(_):
    cfg = configure()
    print("build directory: %s" % BUILD)
    for b in cfg["backends"]:
        print("  %-6s %-8s %s" % (b["name"], "enabled" if b["enabled"] else "skipped", b["why"]))
    for name, d in device_info(cfg).items():
        print("\n%s device: %s" % (name, d.get("name", d.get("why", "?"))))
        for k, v in d.items():
            if k not in ("kind", "backend", "name", "available", "log"):
                print("  %-34s %s" % (k, v))
    if cfg["rc"] != 0:
        print("\ncmake configure failed:\n" + tail(cfg["log"], 30))
        return 1
    return 0


def cmd_run(a):
    cfg = configure()
    if cfg["rc"] != 0:
        sys.exit("cmake configure failed:\n" + tail(cfg["log"], 30))
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
    if "race" in want and "race" in enabled and not any(
            t["lab"] == lab and t["backend"] == "race" for t in cfg["targets"]):
        print("\n== race: this lab is not built for the race-detection emulator "
              "(its CMakeLists.txt does not pass RACE to pmpp_add_lab)")
    for t in cfg["targets"]:
        if t["lab"] != lab or t["variant"] != variant or t["backend"] not in want:
            continue
        is_extra_tile = t["target"].split(".")[-1].startswith("t") and t["target"].count(".") == 3
        if is_extra_tile and not a.bench:
            continue
        print("\n== %s, %s, %s%s" % (lab, variant, t["backend"], ", TILE_WIDTH=%d" % t["tile"] if is_extra_tile else ""))
        r = exercise(t, not a.no_test and not is_extra_tile, a.bench and t["backend"] not in EMULATED)
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
    return rc_all


def run_selftests(cfg, backends, only=None, echo=True):
    """Build and run the emulator's self-tests (labs/selftest) on the given backends."""
    results = []
    for t in cfg["selftests"]:
        if t["backend"] not in backends or (only and only not in t["lab"]):
            continue
        if echo:
            print("[%s] %s ..." % (t["backend"], t["target"]), flush=True)
        r = exercise(t, True, False)
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
    results = run_selftests(cfg, want, a.name)
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
    started = datetime.datetime.now()
    cfg = configure()
    devs = device_info(cfg) if cfg["rc"] == 0 else {}
    usable = [b["name"] for b in cfg["backends"] if b["enabled"] and devs.get(b["name"], {}).get("available")]
    results = []
    labs = sorted({t["lab"] for t in cfg["targets"] if t["lab"] != "devinfo"})
    states = {lab: work_state(lab) for lab in labs}
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
        r = exercise(t, do_test, do_bench)
        print("    tests: %s%s" % (counts(r), ", %d benchmark rows" % len(r["bench"]) if r["bench"] else ""), flush=True)
        results.append(r)
    selftests = run_selftests(cfg, usable)
    check = cuda_frontend_check()

    report = {
        "generated": started.isoformat(timespec="seconds"),
        "system": {"os": platform.platform(), "machine": platform.machine(),
                   "python": platform.python_version(),
                   "cmake": sh(["cmake", "--version"])[1].splitlines()[0]},
        "configure_ok": cfg["rc"] == 0,
        "configure_log": scrub(tail(cfg["log"], 60)),
        "backends": cfg["backends"], "devices": devs, "usable_backends": usable,
        "work_state": states, "results": results, "selftests": selftests, "cuda_frontend_check": check,
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
        w("| %s | %s | %s | %s | %s | %s |" % (r["lab"], r["backend"], state, "ok" if r["built"] else "FAILED",
                                              counts(r), reading))
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
        keys = []
        for row in r["bench"]:
            for k in row:
                if k not in ("kind", "lab", "variant", "backend") and k not in keys:
                    keys.append(k)
        w("| " + " | ".join(keys) + " |")
        w("|" + "---|" * len(keys))
        for row in r["bench"]:
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
    r.add_argument("--no-test", action="store_true")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("selftest", help="run the emulator's own tests (labs/selftest); writes no report")
    t.add_argument("name", nargs="?", help="only self-tests whose name contains this")
    t.add_argument("--backend", default="all", choices=["all"] + BACKENDS)
    t.add_argument("--verbose", action="store_true", help="print every test's output, not only failures")
    t.set_defaults(fn=cmd_selftest)
    v = sub.add_parser("verify")
    v.add_argument("--quick", action="store_true", help="skip benchmarks")
    v.set_defaults(fn=cmd_verify)
    s = sub.add_parser("reset")
    s.add_argument("lab")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_reset)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
