#!/usr/bin/env python3
"""Build, test and benchmark the labs on whatever backends this machine has.

  python3 labs/pmpp.py doctor                 which backends can be built here, and why not
  python3 labs/pmpp.py run 5.2                test your work for lab 5.2 on every backend
  python3 labs/pmpp.py run 5.2 --bench        ...and benchmark it
  python3 labs/pmpp.py run 5.2 --solution     run the reference solution instead
  python3 labs/pmpp.py run 5.2 --backend emu  one backend only (emu, metal, cuda, hip)
  python3 labs/pmpp.py verify                 everything, and write reports/verify-*.md
  python3 labs/pmpp.py reset 5.2 --yes        overwrite your work for a lab with the starter

Standard library only. Needs cmake >= 3.21 and a C++17 compiler.
"""
import argparse
import datetime
import filecmp
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
BACKENDS = ["emu", "metal", "cuda", "hip"]
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
    backends, targets = [], []
    try:
        for line in open(os.path.join(BUILD, "pmpp_backends.txt")):
            name, state, why = line.rstrip("\n").split("|", 2)
            backends.append({"name": name, "enabled": state == "enabled", "why": why})
        for line in open(os.path.join(BUILD, "pmpp_targets.txt")):
            lab, variant, backend, tile, target, exe = line.rstrip("\n").split("|")
            targets.append({"lab": lab, "variant": variant, "backend": backend, "tile": int(tile),
                            "target": target, "exe": exe})
    except FileNotFoundError:
        pass
    return {"rc": rc, "log": out, "backends": backends, "targets": targets}


def build(target):
    rc, out = sh(["cmake", "--build", BUILD, "--target", target, "-j"], timeout=1800)
    return rc == 0, out


def run_exe(exe, flags, timeout):
    rc, out = sh([exe] + flags, timeout=timeout)
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


def resolve_lab(cfg, token):
    labs = sorted({t["lab"] for t in cfg["targets"] if t["lab"] != "devinfo"})
    m = re.fullmatch(r"(\d+)\.(\d+)", token)
    if m:
        want = "ch%02d/lab%s_" % (int(m.group(1)), m.group(2))
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
    base = [clang, "-x", "cuda", "--cuda-host-only", "-nocudainc", "-nocudalib", "-fsyntax-only",
            "-std=c++17", "-I", inc, "-include", "pmpp/check/cuda_stub.h", "-DTILE_WIDTH=16"]
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
    r.update({"built": False, "tests": [], "bench": [], "test_rc": None, "log": ""})
    ok, out = build(t["target"])
    r["built"] = ok
    if not ok:
        r["log"] = scrub(tail(out, 60))
        return r
    if do_test:
        rc, rows, text = run_exe(t["exe"], ["--test"], timeout)
        r["test_rc"] = rc
        r["tests"] = [x for x in rows if x.get("kind") == "test"]
        r["test_text"] = scrub(text)
        if rc != 0:
            r["log"] = scrub(tail(text, 60))
    if do_bench and (not do_test or r["test_rc"] == 0):
        rc, rows, text = run_exe(t["exe"], ["--bench"], timeout)
        r["bench_rc"] = rc
        r["bench"] = [x for x in rows if x.get("kind") == "bench"]
        r["bench_text"] = scrub(text)
    return r


def passed(r):
    return r["built"] and r["test_rc"] == 0 and r["tests"] and all(x["pass"] for x in r["tests"])


def counts(r):
    if not r["built"]:
        return "build failed"
    if r["test_rc"] is None:
        return "not run"
    n = len(r["tests"])
    p = sum(1 for x in r["tests"] if x["pass"])
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
    for t in cfg["targets"]:
        if t["lab"] != lab or t["variant"] != variant or t["backend"] not in want:
            continue
        is_extra_tile = t["target"].split(".")[-1].startswith("t") and t["target"].count(".") == 3
        if is_extra_tile and not a.bench:
            continue
        print("\n== %s, %s, %s%s" % (lab, variant, t["backend"], ", TILE_WIDTH=%d" % t["tile"] if is_extra_tile else ""))
        r = exercise(t, not a.no_test and not is_extra_tile, a.bench)
        if not r["built"]:
            print("BUILD FAILED\n" + r["log"])
            rc_all = 1
            continue
        if r.get("test_text"):
            print(r["test_text"])
        if r.get("bench_text"):
            print(r["bench_text"])
        if r["test_rc"] not in (None, 0):
            rc_all = 1
    return rc_all


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
            do_test, do_bench = True, (t["backend"] != "emu" and not a.quick)
        else:  # your work
            edited = states[t["lab"]] == "edited"
            if extra_tile and not edited:
                continue
            do_test, do_bench = True, (edited and t["backend"] != "emu" and not a.quick)
        print("[%s] %s ..." % (t["backend"], t["target"]), flush=True)
        r = exercise(t, do_test, do_bench)
        print("    tests: %s%s" % (counts(r), ", %d benchmark rows" % len(r["bench"]) if r["bench"] else ""), flush=True)
        results.append(r)
    check = cuda_frontend_check()

    report = {
        "generated": started.isoformat(timespec="seconds"),
        "system": {"os": platform.platform(), "machine": platform.machine(),
                   "python": platform.python_version(),
                   "cmake": sh(["cmake", "--version"])[1].splitlines()[0]},
        "configure_ok": cfg["rc"] == 0,
        "configure_log": scrub(tail(cfg["log"], 60)),
        "backends": cfg["backends"], "devices": devs, "usable_backends": usable,
        "work_state": states, "results": results, "cuda_frontend_check": check,
    }
    os.makedirs(os.path.join(REPORTS, "runs"), exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    tag = "%s-%s" % (platform.system().lower(), "+".join(usable) or "none")
    base = os.path.join(REPORTS, "verify-%s-%s" % (stamp, tag))
    json.dump(report, open(base + ".json", "w"), indent=1)
    open(base + ".md", "w").write(render(report))
    # Latest measurements per backend, for the app to display.
    for b in usable:
        json.dump({"generated": report["generated"], "device": devs[b],
                   "results": [r for r in results if r["backend"] == b]},
                  open(os.path.join(REPORTS, "runs", "%s.json" % b), "w"), indent=1)
    bad = [r for r in results if r["variant"] == "solution" and not passed(r)]
    print("\nreport: %s" % os.path.relpath(base + ".md", REPO))
    print("reference solutions: %d ran, %d failed" % (sum(1 for r in results if r["variant"] == "solution"), len(bad)))
    return 1 if (bad or cfg["rc"] != 0) else 0


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
