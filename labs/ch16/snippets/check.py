#!/usr/bin/env python3
"""Runs chapter 16's snippets on the course's CPU emulator and checks what the lessons say
about them. Every claim a chapter 16 page marks "emulator-checked" for a snippet (not a
lab) comes from here, so that anyone can repeat it:

  fw_bottomup.cu   section 16.3: the corrected Fig. 16.4 equals a sequential CPU
                   Floyd-Warshall step by step on Fig. 16.5's table as printed, changes the
                   four cells the lesson names at step c, and equals the CPU on random
                   graphs with non-negative weights for block sizes that leave threads out
                   of range, under the emulator's three thread orders

The drivers are in check/. Each snippet and its driver are translated by
labs/tools/cu2emu.py and compiled as the lab runner compiles the emu backend
(pmpp/emu/cuda_emu.h force-included, AddressSanitizer on). Nothing here runs on a GPU.
To add a snippet, add a line to CASES.

Usage: python3 labs/ch16/snippets/check.py [--build DIR]
The build directory defaults to $PMPP_BUILD_DIR/ch16-snippets, or a temporary one.
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
CASES = [
    ("fw_bottomup", "fw_bottomup_test.cu", ["fw_bottomup.cu"]),
]

FLAGS = ["-std=c++17", "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", "-pthread",
         "-DPMPP_BACKEND_EMU", "-I", INC, "-I", os.path.join(INC, "pmpp", "emu", "shim"),
         "-include", "pmpp/emu/cuda_emu.h"]


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
    build = a.build or (os.path.join(os.environ["PMPP_BUILD_DIR"], "ch16-snippets")
                        if os.environ.get("PMPP_BUILD_DIR") else None)
    tmp = None
    if not build:
        tmp = tempfile.mkdtemp(prefix="pmpp-ch16-snippets-")
        build = tmp
    os.makedirs(build, exist_ok=True)
    env = dict(os.environ, ASAN_OPTIONS=os.environ.get("ASAN_OPTIONS", "detect_leaks=0"))
    print("compiler:", run([cxx, "--version"])[1].splitlines()[0])
    failed = []
    for name, driver, snippets in CASES:
        objs, problems = [], []
        for src in [os.path.join(HERE, s) for s in snippets] + [os.path.join(HERE, "check", driver)]:
            stem = os.path.splitext(os.path.basename(src))[0]
            cpp, obj = os.path.join(build, stem + ".cpp"), os.path.join(build, stem + ".o")
            rc, out = run([sys.executable, CU2EMU, src, cpp])
            if rc:
                problems.append("cu2emu failed on %s:\n%s" % (src, out))
                break
            rc, out = run([cxx] + FLAGS + ["-Wall", "-Wextra", "-I", os.path.join(HERE, "check"),
                                           "-c", cpp, "-o", obj])
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
    print("%d of %d snippet checks passed" % (len(CASES) - len(failed), len(CASES)))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
