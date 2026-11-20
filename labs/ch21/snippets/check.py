#!/usr/bin/env python3
"""Runs chapter 21's snippets on the course's CPU emulator and checks what section 21.2
says about them. Every claim the page marks "emulator-checked" for a snippet (not a lab)
comes from here, so that anyone can repeat it:

  scatter.cu           Fig. 21.5 with an atom bound: matches a float64 sum, chunked atoms,
                       two slices, several launch shapes
  gather.cu            Fig. 21.6 with bound checks: the same
  coarsen_adjacent.cu  Fig. 21.8 corrected (thread index scaled by 4): the same
  coarsen_strided.cu   Fig. 21.10 with a bound check per point: the same
  coarsen_printed.cu   Fig. 21.8 as printed: wrong in the way the lesson says (at blockDim.x
                       = 32, points 0 to 34 of a block's 128 written, 3 to 31 four times,
                       35 to 127 never)

The driver is check/dcs_test.cu, compiled once per snippet with the matching -DKIND_...
Each snippet and the driver are translated by labs/tools/cu2emu.py and compiled as the lab
runner compiles the emu backend (pmpp/emu/cuda_emu.h force-included, AddressSanitizer on,
-Wall -Wextra). Every case runs under the emulator's three thread orders. The emulator runs
one thread at a time, so it cannot see the scatter kernel's need for atomics (a plain +=
would pass here too); section 21.2 says so. Nothing here runs on a GPU.

Usage: python3 labs/ch21/snippets/check.py [--build DIR]
The build directory defaults to $PMPP_BUILD_DIR/ch21-snippets, or a temporary one.
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

# (name, snippet, defines). A chunk of 64 atoms makes the 150-atom cases take three launches.
CASES = [
    ("scatter", "scatter.cu", ["-DKIND_SCATTER", "-DCHUNK_SIZE=64"]),
    ("gather", "gather.cu", ["-DKIND_GATHER", "-DCHUNK_SIZE=64"]),
    ("coarsen_adjacent", "coarsen_adjacent.cu", ["-DKIND_ADJACENT", "-DCHUNK_SIZE=64"]),
    ("coarsen_strided", "coarsen_strided.cu", ["-DKIND_STRIDED", "-DCHUNK_SIZE=64"]),
    ("coarsen_printed", "coarsen_printed.cu", ["-DKIND_PRINTED", "-DCHUNK_SIZE=64"]),
]
DRIVER = os.path.join(HERE, "check", "dcs_test.cu")

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
    build = a.build or (os.path.join(os.environ["PMPP_BUILD_DIR"], "ch21-snippets")
                        if os.environ.get("PMPP_BUILD_DIR") else None)
    tmp = None
    if not build:
        tmp = tempfile.mkdtemp(prefix="pmpp-ch21-snippets-")
        build = tmp
    os.makedirs(build, exist_ok=True)
    env = dict(os.environ, ASAN_OPTIONS=os.environ.get("ASAN_OPTIONS", "detect_leaks=0"))
    print("compiler:", run([cxx, "--version"])[1].splitlines()[0])
    failed = []
    for name, snippet, defines in CASES:
        objs, problems = [], []
        for src in [os.path.join(HERE, snippet), DRIVER]:
            stem = name + "-" + os.path.splitext(os.path.basename(src))[0]
            cpp, obj = os.path.join(build, stem + ".cpp"), os.path.join(build, stem + ".o")
            rc, out = run([sys.executable, CU2EMU, src, cpp])
            if rc:
                problems.append("cu2emu failed on %s:\n%s" % (src, out))
                break
            rc, out = run([cxx] + FLAGS + defines + ["-Wall", "-Wextra", "-I", os.path.join(HERE, "check"),
                                                     "-c", cpp, "-o", obj])
            if rc:
                problems.append("compile failed for %s:\n%s" % (src, out))
                break
            warned = [l for l in out.splitlines() if "warning" in l and os.path.basename(src) in l]
            if warned:
                problems.append("warnings in %s:\n%s" % (src, "\n".join(warned)))
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
