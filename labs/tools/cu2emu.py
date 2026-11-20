#!/usr/bin/env python3
"""Rewrite the two pieces of CUDA syntax that are not C++ so a .cu file can be
compiled for the pmpp CPU emulator. Everything else is handled by macros in
pmpp/emu/cuda_emu.h.

  kernel<<<grid, block[, shmem[, stream]]>>>(args);
      -> ::pmpp_emu::Launch(grid, block[, shmem[, stream]]).run(kernel, args);
  extern __shared__ T name[];
      -> T* name = (T*)::pmpp_emu::dyn_shared();

Usage: cu2emu.py in.cu out.cpp
"""
import re
import sys


def blank_comments_and_strings(src: str) -> str:
    """Same length as src, with comment and string contents replaced by spaces,
    so structural scanning ignores them."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        two = src[i:i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif src[i] in "\"'":
            q, j = src[i], i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
        else:
            i += 1
            continue
        for k in range(i, j):
            if out[k] != "\n":
                out[k] = " "
        i = j
    return "".join(out)


def rewrite(src: str) -> str:
    src = re.sub(r"\bextern\s+__shared__\s+([A-Za-z_][\w\s\*]*?)\s+(\w+)\s*\[\s*\]\s*;",
                 r"\1* \2 = (\1*)::pmpp_emu::dyn_shared();", src)
    out, pos = [], 0
    while True:
        mask = blank_comments_and_strings(src)
        a = mask.find("<<<", pos)
        if a < 0:
            break
        b = mask.find(">>>", a)
        if b < 0:
            raise SystemExit("cu2emu: '<<<' without matching '>>>'")
        m = re.search(r"([A-Za-z_][\w:]*)\s*$", mask[:a])
        if not m:
            raise SystemExit("cu2emu: no kernel name before '<<<'")
        k = b + 3
        while k < len(mask) and mask[k].isspace():
            k += 1
        if k >= len(mask) or mask[k] != "(":
            raise SystemExit("cu2emu: expected '(' after '>>>'")
        depth, e = 0, k
        while e < len(mask):
            depth += mask[e] == "("
            depth -= mask[e] == ")"
            if depth == 0:
                break
            e += 1
        if depth != 0:
            raise SystemExit("cu2emu: unbalanced parentheses in kernel launch")
        cfg, args = src[a + 3:b], src[k + 1:e]
        call = "::pmpp_emu::Launch(%s).run(%s%s)" % (
            cfg.strip(), m.group(1), (", " + args) if args.strip() else "")
        src = src[:m.start(1)] + call + src[e + 1:]
        pos = m.start(1) + len(call)
    return src


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    text = open(sys.argv[1], encoding="utf-8").read()
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write('#line 1 "%s"\n' % sys.argv[1].replace("\\", "/"))
        f.write(rewrite(text))
