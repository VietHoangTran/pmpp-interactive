#!/usr/bin/env python3
"""Rewrite the pieces of CUDA syntax that are not C++ so a .cu file can be
compiled for the pmpp CPU emulator. Everything else is handled by
pmpp/emu/cuda_emu.h, which the build force-includes.

  kernel<<<grid, block[, shmem[, stream]]>>>(args);
      -> ::pmpp_emu::Launch(grid, block[, shmem[, stream]]).run(kernel, args);
      `kernel` may be qualified (ns::k) or a template-id (k<float>). Not
      understood: launching an overloaded kernel (the name alone must identify
      one function).

  __shared__ T a[N], b;                      (inside a function)
      -> struct pmpp_shared_K { T a[N], b; };
         auto& pmpp_shared_K_ref = ::pmpp_emu::shared_block<pmpp_shared_K>();
         auto& a = pmpp_shared_K_ref.a; auto& b = pmpp_shared_K_ref.b;
      so that every block gets its own copy. Not understood: a __shared__
      declaration at file scope, one produced by a macro, or one whose
      declarators are not plain names with optional array bounds.

  extern __shared__ T name[];                (inside a function)
      -> T* name = (T*)::pmpp_emu::dyn_shared();

  __global__ void kernel(params) { ... }     (not a template)
      -> the same, followed by
         static ::pmpp_emu::KernelReg pmpp_emu_kreg_K(static_cast<void (*)(params)>(&kernel));
      which is how cudaLaunchCooperativeKernel finds the kernel's type.

Line numbers are preserved: every rewrite stays on the lines it replaces.

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


def line_of(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


def die(src: str, pos: int, msg: str):
    raise SystemExit("cu2emu: line %d: %s" % (line_of(src, pos), msg))


def match_close(mask: str, open_pos: int) -> int:
    """Index of the bracket that closes the one at open_pos, or -1."""
    pairs = {"(": ")", "{": "}", "[": "]"}
    o, c = mask[open_pos], pairs[mask[open_pos]]
    depth = 0
    for i in range(open_pos, len(mask)):
        if mask[i] == o:
            depth += 1
        elif mask[i] == c:
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_level(text: str, sep: str = ","):
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def apply_edits(src: str, edits):
    """edits: (start, end, replacement); replacement keeps the newline count."""
    for a, b, new in sorted(edits, reverse=True):
        new += "\n" * (src.count("\n", a, b) - new.count("\n"))
        src = src[:a] + new + src[b:]
    return src


def rewrite_shared(src: str) -> str:
    mask = blank_comments_and_strings(src)
    edits, count = [], 0
    for m in re.finditer(r"\b__shared__\b", mask):
        line_start = mask.rfind("\n", 0, m.start()) + 1
        if mask[line_start:m.start()].lstrip().startswith("#"):
            die(src, m.start(), "__shared__ inside a preprocessor directive is not understood by the emulator")
        depth = mask.count("{", 0, m.start()) - mask.count("}", 0, m.start())
        semi = mask.find(";", m.end())
        if semi < 0:
            die(src, m.start(), "__shared__ declaration without ';'")
        ext = re.search(r"\bextern\s+$", mask[:m.start()])
        if ext:
            d = re.fullmatch(r"\s*([A-Za-z_][\w\s\*:]*?)\s*\b(\w+)\s*\[\s*\]\s*", src[m.end():semi])
            if not d:
                die(src, m.start(), "expected `extern __shared__ T name[];`")
            if depth == 0:
                die(src, m.start(), "the emulator needs `extern __shared__` declared inside the kernel "
                                    "(each block has its own dynamic shared memory)")
            edits.append((ext.start(), semi + 1, "%s* %s = (%s*)::pmpp_emu::dyn_shared();" % (d.group(1), d.group(2), d.group(1))))
            continue
        if depth == 0:
            die(src, m.start(), "__shared__ at file scope is not understood by the emulator; declare it inside the kernel")
        decl = " ".join(mask[m.end():semi].split())
        names = []
        for piece in split_top_level(decl):
            d = re.search(r"([A-Za-z_]\w*)\s*(\[.*)?$", piece.strip())
            if not d or "(" in piece or "=" in piece:
                die(src, m.start(), "this form of __shared__ declaration is not understood by the emulator: " + decl)
            names.append(d.group(1))
        count += 1
        tag = "pmpp_shared_%d" % count
        text = "struct %s { %s; }; auto& %s_ref = ::pmpp_emu::shared_block<%s>();" % (tag, decl, tag, tag)
        for nm in names:
            text += " auto& %s = %s_ref.%s;" % (nm, tag, nm)
        edits.append((m.start(), semi + 1, text))
    return apply_edits(src, edits)


def register_kernels(src: str) -> str:
    mask = blank_comments_and_strings(src)
    edits, count = [], 0
    for m in re.finditer(r"\b__global__\b", mask):
        after = re.match(r"\s*(?:(?:static|inline|__forceinline__|__noinline__|__inline_hint__)\s+)*void\s+([A-Za-z_]\w*)\s*\(", mask[m.end():])
        if not after:   # `void __global__ name(`
            after = re.match(r"\s*([A-Za-z_]\w*)\s*\(", mask[m.end():])
            if not after or not re.search(r"\bvoid\s+$", mask[:m.start()]):
                continue
        name = after.group(1)
        lparen = m.end() + after.end() - 1
        rparen = match_close(mask, lparen)
        if rparen < 0:
            die(src, m.start(), "unbalanced parentheses in the parameter list of kernel " + name)
        body = re.match(r"\s*\{", mask[rparen + 1:])
        if not body:
            continue                                   # a prototype
        end = match_close(mask, rparen + body.end())
        if end < 0:
            die(src, m.start(), "unbalanced braces in kernel " + name)
        head_start = max(mask.rfind(";", 0, m.start()), mask.rfind("}", 0, m.start())) + 1
        params = " ".join(mask[lparen + 1:rparen].split())
        if re.search(r"\btemplate\b", mask[head_start:m.start()]) or "=" in params:
            continue                                   # templates and default arguments: not registered
        count += 1
        edits.append((end + 1, end + 1, " static ::pmpp_emu::KernelReg pmpp_emu_kreg_%d(static_cast<void (*)(%s)>(&%s));"
                      % (count, params, name)))
    return apply_edits(src, edits)


def kernel_name_before(mask: str, a: int):
    """The kernel expression that ends just before position a: `name`, `ns::name`
    or `name<template args>`. Returns its start index or -1."""
    i = a
    while i > 0 and mask[i - 1].isspace():
        i -= 1
    if i > 0 and mask[i - 1] == ">":
        depth = 0
        while i > 0:
            i -= 1
            if mask[i] == ">":
                depth += 1
            elif mask[i] == "<":
                depth -= 1
                if depth == 0:
                    break
        else:
            return -1
        while i > 0 and mask[i - 1].isspace():
            i -= 1
    m = re.search(r"([A-Za-z_][\w:]*)$", mask[:i])
    return m.start(1) if m else -1


def rewrite_launches(src: str) -> str:
    pos = 0
    while True:
        mask = blank_comments_and_strings(src)
        a = mask.find("<<<", pos)
        if a < 0:
            break
        b = mask.find(">>>", a)
        if b < 0:
            die(src, a, "'<<<' without matching '>>>'")
        start = kernel_name_before(mask, a)
        if start < 0:
            die(src, a, "no kernel name before '<<<'")
        k = b + 3
        while k < len(mask) and mask[k].isspace():
            k += 1
        if k >= len(mask) or mask[k] != "(":
            die(src, a, "expected '(' after '>>>'")
        e = match_close(mask, k)
        if e < 0:
            die(src, a, "unbalanced parentheses in kernel launch")
        cfg, args, kernel = src[a + 3:b], src[k + 1:e], src[start:a].strip()
        call = "::pmpp_emu::Launch(%s).run(%s%s)" % (cfg.strip(), kernel, (", " + args) if args.strip() else "")
        src = src[:start] + call + src[e + 1:]
        pos = start + len(call)
    return src


def rewrite(src: str) -> str:
    return rewrite_launches(register_kernels(rewrite_shared(src)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    text = open(sys.argv[1], encoding="utf-8").read()
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        f.write('#line 1 "%s"\n' % sys.argv[1].replace("\\", "/"))
        f.write(rewrite(text))
