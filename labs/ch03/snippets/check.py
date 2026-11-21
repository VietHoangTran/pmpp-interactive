#!/usr/bin/env python3
"""Runs chapter 3's snippets on the course's CPU emulator and checks what the lessons say
about them. Every claim a chapter 3 page marks "emulator-checked" for a snippet (not a
lab) comes from here, so that anyone can repeat it:

  launch_checked.cu    section 3.1: a (32, 32, 2) block is rejected and leaves the buffer
                       alone; a (1, 1, 128) block, a 65,536-block grid along y and a zero
                       dimension are rejected too
  scale_rows_defect.cu summary page: column 0 always ends at 1.0; the counts of 1.0 under
                       the three thread orders are the ones the page quotes
  blur_kernel.cu       section 3.3: Fig. 3.8 matches a CPU reference
  blur_unsigned.cu     section 3.3: the all-unsigned variant gives the same output, and a
                       build with -Wall -Wextra warns about its signed/unsigned comparisons
  grayscale_kernel.cu  section 3.2: the corrected Fig. 3.4 matches a CPU reference
  matvec_options.cu    section 3.4, exercise 2: option 2 computes B c, option 3 Bt c

The drivers are in check/. Each snippet and each driver is translated by
labs/tools/cu2emu.py and compiled as the lab runner compiles the emu backend
(pmpp/emu/cuda_emu.h force-included, AddressSanitizer on). Nothing here runs on a GPU.

Usage: python3 labs/ch03/snippets/check.py [--build DIR]
The build directory defaults to $PMPP_BUILD_DIR/ch03-snippets, or a temporary one.
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

# (name, driver in check/, snippets it runs, snippets whose -Wall -Wextra build must warn
# about a signed/unsigned comparison, snippets whose build must not)
CASES = [
    ("launch_checked", "launch_checked_test.cu", ["launch_checked.cu"], [], []),
    ("scale_rows_defect", "scale_rows_test.cu", ["scale_rows_defect.cu"], [], []),
    ("blur", "blur_test.cu", ["blur_kernel.cu", "blur_unsigned.cu"], ["blur_unsigned.cu"], ["blur_kernel.cu"]),
    ("grayscale", "grayscale_test.cu", ["grayscale_kernel.cu"], [], []),
    ("matvec", "matvec_test.cu", ["matvec_options.cu"], [], []),
]

FLAGS = ["-std=c++17", "-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", "-pthread",
         "-DPMPP_BACKEND_EMU", "-I", INC, "-I", os.path.join(INC, "pmpp", "emu", "shim"),
         "-include", "pmpp/emu/cuda_emu.h"]


def run(cmd, **kw):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    return p.returncode, p.stdout


def sign_warnings(log, name):
    """Warning lines about a signed/unsigned comparison in the source `name` (cu2emu keeps
    the original file name and line numbers through #line)."""
    return [l for l in log.splitlines()
            if ("/" + name + ":") in l and "warning" in l and "sign" in l]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", help="build directory (kept)")
    a = ap.parse_args()
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        print("no C++ compiler found (set CXX)")
        return 2
    build = a.build or (os.path.join(os.environ["PMPP_BUILD_DIR"], "ch03-snippets")
                        if os.environ.get("PMPP_BUILD_DIR") else None)
    tmp = None
    if not build:
        tmp = tempfile.mkdtemp(prefix="pmpp-ch03-snippets-")
        build = tmp
    os.makedirs(build, exist_ok=True)
    env = dict(os.environ, ASAN_OPTIONS=os.environ.get("ASAN_OPTIONS", "detect_leaks=0"))
    print("compiler:", run([cxx, "--version"])[1].splitlines()[0])
    failed = []
    for name, driver, snippets, must_warn, must_not_warn in CASES:
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
            base = os.path.basename(src)
            warned = sign_warnings(out, base)
            if base in must_warn:
                if warned:
                    print("  ok: %s warns about %d signed/unsigned comparison(s) with -Wall -Wextra" % (base, len(warned)))
                else:
                    problems.append("%s: expected a signed/unsigned comparison warning, got none" % base)
            if base in must_not_warn and warned:
                problems.append("%s: unexpected signed/unsigned warnings:\n%s" % (base, "\n".join(warned)))
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
