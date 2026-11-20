#!/usr/bin/env python3
"""Compiles and runs Appendix C's Metal snippets on this Mac, and checks what section C.1
says about them. Every Metal claim on that page that is not a lab comes from here, so
that anyone can repeat it:

  c1_helper_overloads.metal  compiles at every MSL version the compiler offers for macOS,
                             2.0 to 4.1, and its kernel row_max_twice, run on the GPU,
                             matches a CPU reference (the Metal tab's "verified on Metal")
  the five MSL cards         each card's helper and call, compiled at every version up to
                             4.0, lands where the specification puts it (appC-facts C1);
                             at 4.1 and at the compiler's default only the card with no
                             address space moves ("observed, not documented", C1)
  check/c1_generic_probe.metal
                             the unqualified helper: rejected with -std=metal4.0, accepted
                             at the default, and, run on the GPU, it reads device,
                             threadgroup and thread memory correctly

Needs macOS with Xcode's Metal toolchain (xcrun metal). Nothing here runs on CUDA or HIP.

Usage: python3 labs/appC/snippets/check.py [--build DIR]
The build directory defaults to $PMPP_BUILD_DIR/appC-snippets, or a temporary one.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLOADS = os.path.join(HERE, "c1_helper_overloads.metal")
PROBE = os.path.join(HERE, "check", "c1_generic_probe.metal")
RUNNER = os.path.join(HERE, "check", "c1_run.mm")

# Every -std value the compiler accepts for macOS, oldest first. "default" passes none.
VERSIONS = ["macos-metal2.0", "macos-metal2.1", "macos-metal2.2", "macos-metal2.3", "macos-metal2.4",
            "metal3.0", "metal3.1", "metal3.2", "metal4.0", "metal4.1", "default"]
UP_TO_40 = VERSIONS[:VERSIONS.index("metal4.0") + 1]

MISSING = "pointer type must have explicit address space qualifier"
NO_MATCH = "no matching function for call"

BODY = "{ float m = p[0]; for (uint i = 1; i < n; ++i) m = p[i] > m ? p[i] : m; return m; }"
HELPER = {
    "none": "float rowMax(const float* p, uint n) " + BODY,
    "device": "float rowMax(device const float* p, uint n) " + BODY,
    "threadgroup": "float rowMax(threadgroup const float* p, uint n) " + BODY,
}
# The MSL cards of section C.1 (app/src/sim/addrspace.ts, CARDS): helper overloads, the
# call, and the expected outcome under the specification and at 4.1 / the default.
# None means "compiles"; otherwise the compiler message that must appear.
CARDS = [
    ("msl-none", ["none"], "rowMax(tile, n)", MISSING, None),
    ("msl-device-only", ["device"], "rowMax(tile, n)", NO_MATCH, NO_MATCH),
    ("msl-pair-tile", ["device", "threadgroup"], "rowMax(tile, n)", None, None),
    ("msl-pair-row", ["device", "threadgroup"], "rowMax(x + row * n, n)", None, None),
    ("msl-threadgroup", ["threadgroup"], "rowMax(tile, n)", None, None),
]
KERNEL = """kernel void card(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                 constant uint& n [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                 uint t [[thread_position_in_threadgroup]]) {
    threadgroup float tile[256];
    tile[t] = x[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) out[row] = %s;
}
"""


def run(cmd, **kw):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    return p.returncode, p.stdout


def metal(src, std, out):
    cmd = ["xcrun", "-sdk", "macosx", "metal"]
    if std != "default":
        cmd.append("-std=" + std)
    if out.endswith(".air"):
        cmd.append("-c")
    return run(cmd + [src, "-o", out])


def expect_compile(src, std, want, build, label):
    """want None: must compile. want a message: must fail with it. Returns a problem or None."""
    rc, log = metal(src, std, os.path.join(build, "probe.air"))
    if want is None and rc == 0:
        return None
    if want is not None and rc != 0 and want in log:
        return None
    got = "compiled" if rc == 0 else "error: " + next((l for l in log.splitlines() if "error" in l), log.strip()[:200])
    return "%s at %s: expected %s, %s" % (label, std, "to compile" if want is None else '"%s"' % want, got)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", help="build directory (kept)")
    a = ap.parse_args()
    if sys.platform != "darwin" or not shutil.which("xcrun"):
        print("skipped: needs macOS with the Metal toolchain")
        return 2
    rc, ver = run(["xcrun", "-sdk", "macosx", "metal", "--version"])
    if rc:
        print("skipped: xcrun metal is not available:\n" + ver)
        return 2
    print("compiler:", ver.splitlines()[0])
    build = a.build or (os.path.join(os.environ["PMPP_BUILD_DIR"], "appC-snippets")
                        if os.environ.get("PMPP_BUILD_DIR") else None)
    tmp = None
    if not build:
        tmp = tempfile.mkdtemp(prefix="pmpp-appC-snippets-")
        build = tmp
    os.makedirs(build, exist_ok=True)
    failed = []

    def report(name, problems, ok_text):
        for p in problems:
            print("  FAIL: " + p)
        if problems:
            failed.append(name)
        else:
            print("  ok: " + ok_text)

    # 1. The overload pair compiles at every version.
    print("c1_helper_overloads.metal, compile")
    report("overloads-compile", [p for p in (expect_compile(OVERLOADS, s, None, build, "overloads") for s in VERSIONS) if p],
           "compiles at all %d settings (%s to %s, and the default)" % (len(VERSIONS), VERSIONS[0], VERSIONS[-2]))

    # 2. The cards: specification up to 4.0; at 4.1 and the default only msl-none moves.
    print("section C.1 cards, compile")
    for cid, overloads, call, spec, observed in CARDS:
        src = os.path.join(build, cid + ".metal")
        with open(src, "w") as f:
            f.write("#include <metal_stdlib>\nusing namespace metal;\n\n")
            f.write("\n".join(HELPER[o] for o in overloads) + "\n\n" + KERNEL % call)
        problems = [p for p in (expect_compile(src, s, spec, build, cid) for s in UP_TO_40) if p]
        problems += [p for p in (expect_compile(src, s, observed, build, cid) for s in ("metal4.1", "default")) if p]
        say = lambda w: "compiles" if w is None else 'fails with "%s"' % w
        report(cid, problems, "%s %s up to metal4.0; %s at metal4.1 and the default" % (cid, say(spec), say(observed)))

    # 3. The unqualified probe compiles the same way as the msl-none card.
    print("check/c1_generic_probe.metal, compile")
    problems = [p for p in (expect_compile(PROBE, s, MISSING, build, "probe") for s in UP_TO_40) if p]
    problems += [p for p in (expect_compile(PROBE, s, None, build, "probe") for s in ("metal4.1", "default")) if p]
    report("probe-compile", problems, "rejected up to metal4.0, accepted at metal4.1 and the default")

    # 4. Run both on the GPU.
    exe = os.path.join(build, "c1_run")
    rc, log = run(["xcrun", "-sdk", "macosx", "clang++", "-std=c++17", "-O1", "-fobjc-arc", RUNNER,
                   "-framework", "Metal", "-framework", "Foundation", "-o", exe])
    if rc:
        print("  FAIL: the runner did not build:\n" + log)
        failed += ["run-overloads", "run-generic"]
    else:
        for name, src in (("overloads", OVERLOADS), ("generic", PROBE)):
            print("run on the GPU: %s (compiled at the default)" % os.path.basename(src))
            lib = os.path.join(build, name + ".metallib")
            rc, log = metal(src, "default", lib)
            if rc:
                print("  FAIL: metallib did not build:\n" + log)
                failed.append("run-" + name)
                continue
            rc, out = run([exe, name, lib], timeout=300)
            sys.stdout.write("".join("  " + l + "\n" for l in out.splitlines()))
            if rc:
                failed.append("run-" + name)

    total = 1 + len(CARDS) + 1 + 2
    print("%d of %d checks passed" % (total - len(failed), total))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
