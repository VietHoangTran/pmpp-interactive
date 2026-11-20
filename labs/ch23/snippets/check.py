#!/usr/bin/env python3
"""Checks chapter 23's snippets and what the lessons say about them. Every status a
chapter 23 page gives a snippet (not a lab) comes from here, so anyone can repeat it:

  staged_loop.cu        section 23.3: the CUDA front end parses it (clang, the course's
                        stub headers); on the CPU emulator both issue orders give the
                        right result for equal, uneven and empty chunks, more streams
                        than chunks, one stream and n = 0, writing nothing outside the
                        output ("front-end-checked", "emulator-checked")
  staged_loop_metal.mm  section 23.3, the Metal tab: built with ../staged_loop.metal and
  staged_loop.metal     run on this Mac's GPU for K = 1, 2, 3 and more buffers than
                        chunks, with equal, uneven and empty chunks ("verified on Metal")

The CUDA driver is translated by labs/tools/cu2emu.py and compiled as the lab runner
compiles the emu backend (pmpp/emu/cuda_emu.h force-included, AddressSanitizer on).
The emulator runs streams synchronously: it checks results, never overlap. Nothing
here runs on CUDA or HIP hardware.

Usage: python3 labs/ch23/snippets/check.py [--build DIR]
The build directory defaults to $PMPP_BUILD_DIR/ch23-snippets, or a temporary one.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LABS = os.path.abspath(os.path.join(HERE, "..", ".."))
INC = os.path.join(LABS, "common", "include")
CU2EMU = os.path.join(LABS, "tools", "cu2emu.py")

# (name, driver in check/, snippets it runs)
EMU_CASES = [
    ("staged_loop", "staged_loop_test.cu", ["staged_loop.cu"]),
]
# (name, driver in check/, Objective-C++ snippets, the .metal file the driver compiles)
METAL_CASES = [
    ("staged_loop_metal", "staged_loop_metal_test.mm", ["staged_loop_metal.mm"], "staged_loop.metal"),
]
# .cu files the CUDA front end must accept, as ./verify.sh checks them.
FRONT_END = ["staged_loop.cu", os.path.join("check", "staged_loop_test.cu")]

EMU_FLAGS = ["-std=c++17", "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", "-pthread",
             "-DPMPP_BACKEND_EMU", "-I", INC, "-I", os.path.join(INC, "pmpp", "emu", "shim"),
             "-include", "pmpp/emu/cuda_emu.h"]
FRONT_FLAGS = ["-x", "cuda", "--cuda-host-only", "-nocudainc", "-nocudalib", "-fsyntax-only", "-std=c++17",
               "-I", INC, "-I", os.path.join(INC, "pmpp", "emu", "shim"), "-include", "pmpp/check/cuda_stub.h"]


def run(cmd, **kw):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", help="build directory (kept)")
    a = ap.parse_args()
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        print("no C++ compiler found (set CXX)")
        return 2
    build = a.build or (os.path.join(os.environ["PMPP_BUILD_DIR"], "ch23-snippets")
                        if os.environ.get("PMPP_BUILD_DIR") else None)
    tmp = None
    if not build:
        tmp = tempfile.mkdtemp(prefix="pmpp-ch23-snippets-")
        build = tmp
    os.makedirs(build, exist_ok=True)
    env = dict(os.environ, ASAN_OPTIONS=os.environ.get("ASAN_OPTIONS", "detect_leaks=0"))
    print("compiler:", run([cxx, "--version"])[1].splitlines()[0])
    failed, total = [], 0

    # 1. The CUDA front end.
    clang = os.environ.get("PMPP_CLANG") or shutil.which("clang++")
    for f in FRONT_END:
        total += 1
        name = "front end: " + f
        if not clang:
            print("  skipped: %s (no clang++)" % name)
            continue
        rc, out = run([clang] + FRONT_FLAGS + [os.path.join(HERE, f)], timeout=120)
        print("  %s: %s" % ("ok" if rc == 0 else "FAIL", name))
        if rc:
            print(out)
            failed.append(name)

    # 2. The emulator.
    for name, driver, snippets in EMU_CASES:
        total += 1
        objs, problems = [], []
        for src in [os.path.join(HERE, s) for s in snippets] + [os.path.join(HERE, "check", driver)]:
            stem = os.path.splitext(os.path.basename(src))[0]
            cpp, obj = os.path.join(build, stem + ".cpp"), os.path.join(build, stem + ".o")
            rc, out = run([sys.executable, CU2EMU, src, cpp])
            if rc:
                problems.append("cu2emu failed on %s:\n%s" % (src, out))
                break
            rc, out = run([cxx] + EMU_FLAGS + ["-Wall", "-Wextra", "-c", cpp, "-o", obj])
            if rc:
                problems.append("compile failed for %s:\n%s" % (src, out))
                break
            objs.append(obj)
        exe = os.path.join(build, name)
        if not problems:
            rc, out = run([cxx, "-fsanitize=address", "-pthread"] + objs + ["-o", exe])
            if rc:
                problems.append("link failed:\n" + out)
        if not problems:
            rc, out = run([exe], env=env, timeout=600)
            sys.stdout.write(out)
            if rc:
                problems.append("%s exited with %d" % (name, rc))
        for p in problems:
            print("  FAIL: " + p)
        if problems:
            failed.append(name)

    # 3. Metal, on this Mac.
    for name, driver, snippets, kernel in METAL_CASES:
        total += 1
        if sys.platform != "darwin":
            print("  skipped: %s (needs macOS)" % name)
            continue
        exe = os.path.join(build, name)
        srcs = [os.path.join(HERE, s) for s in snippets] + [os.path.join(HERE, "check", driver)]
        rc, out = run([cxx, "-std=c++17", "-O1", "-fobjc-arc", "-Wall", "-Wextra"] + srcs +
                      ["-framework", "Metal", "-framework", "Foundation", "-o", exe])
        if rc:
            print("  FAIL: %s did not build:\n%s" % (name, out))
            failed.append(name)
            continue
        rc, out = run([exe, os.path.join(HERE, kernel)], timeout=300)
        sys.stdout.write(out)
        if rc:
            print("  FAIL: %s exited with %d" % (name, rc))
            failed.append(name)

    print("%d of %d snippet checks passed" % (total - len(failed), total))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
