#!/usr/bin/env python3
"""Lab X3.2, provided: writes a synthetic Llama-style model in llama2.c's version 0 .bin
format, from a fixed seed, with the Python standard library only. Nothing is downloaded.

The format (notes/today/x3-facts.md D1): a 28-byte header of seven little-endian int32
(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len), then float32
weights in this order:

    token_embedding_table  vocab x dim
    rms_att_weight         layers x dim
    wq                     layers x (n_heads * head) x dim
    wk, wv                 layers x (n_kv_heads * head) x dim
    wo                     layers x dim x (n_heads * head)
    rms_ffn_weight         layers x dim
    w1                     layers x hidden x dim
    w2                     layers x dim x hidden
    w3                     layers x hidden x dim
    rms_final_weight       dim
    two skipped RoPE tables, seq_len x head / 2 floats each (written as zeros)

A positive vocab_size means the classifier is the embedding (shared), as in the small
tinyllamas checkpoints (D2). The file size is therefore (E1)
    28 + 4 * (vocab*dim + L*dim + L*dim*dq + 2*L*dim*dkv + L*dq*dim + L*dim
              + 3*L*dim*hidden + dim + seq_len*head)
with head = dim / n_heads, dq = n_heads * head = dim and dkv = n_kv_heads * head.

The numbers: splitmix64 (our own few lines, not the random module, so every Python on every
machine writes the same bytes), two 24-bit uniforms per 64-bit output, u = k / 2^23 - 1 in
[-1, 1), exact in float32; each tensor is u times its scale, rounded once to float32 by
struct.pack('<f'). The scales are chosen so that the reference's greedy steps have large
top-two margins (README.md, "What the tests do").

--permute-qk writes the same weights with llama.cpp's row permutation of wq and wk, the
one its converter applies to Hugging Face Llama checkpoints (B6):
    w.reshape(n_head, 2, rows // n_head // 2, cols).swapaxes(1, 2).reshape(rows, cols)
with n_head = n_heads for wq and n_kv_heads for wk. Row a * head/2 + i of a head moves to
row 2 * i + a, so the half-split pair (i, i + head/2) becomes the consecutive pair
(2i, 2i + 1): the permuted model under consecutive RoPE (mode 0) computes what this one
computes under half-split RoPE (mode 1).

--fill tile (the benchmark's model): every tensor repeats one block of 4096 random
values, so a large file is written in seconds. Its numbers mean nothing; only its shape
and its timing do. The tests never use it.

It prints the shape, the file size and a checksum: FNV-1a 64 over the bytes of the first
256 floats of the embedding and the last 256 floats of rms_final_weight.
"""
import argparse
import os
import struct
import sys

MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
HEAD = 128          # the head size of the course's decode kernels (llama.h, LLAMA_HEAD_DIM)
TILE = 4096


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK

    def uniforms(self, n):
        """n floats u = k / 2^23 - 1, k a 24-bit integer: two per 64-bit output."""
        out = [0.0] * n
        s = self.state
        inv = 1.0 / 8388608.0   # 2^-23
        i = 0
        while i < n:
            s = (s + GOLDEN) & MASK
            z = s
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
            z ^= z >> 31
            out[i] = (z >> 40) * inv - 1.0
            if i + 1 < n:
                out[i + 1] = ((z >> 8) & 0xFFFFFF) * inv - 1.0
            i += 2
        self.state = s
        return out


def fnv1a64(data, h=0xCBF29CE484222325):
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & MASK
    return h


def permute_rows(w, rows, cols, n_head):
    """llama.cpp's permute (x3-facts B6) on a row-major rows x cols list."""
    per = rows // n_head // 2
    out = [0.0] * (rows * cols)
    for h in range(n_head):
        for a in range(2):
            for i in range(per):
                src = (h * 2 * per + a * per + i) * cols
                dst = (h * 2 * per + i * 2 + a) * cols
                out[dst:dst + cols] = w[src:src + cols]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--permute-qk", action="store_true")
    ap.add_argument("--fill", choices=["random", "tile"], default="random")
    # Scales (ours). A projection's weights are u * scale / sqrt(fan_in); norms are 1 + 0.25 u.
    ap.add_argument("--qk-scale", type=float, default=4.0)
    ap.add_argument("--block-scale", type=float, default=6.0)
    a = ap.parse_args()

    d, hidden, L, hq, hkv, V, T = a.dim, a.hidden, a.layers, a.heads, a.kv_heads, a.vocab, a.seq_len
    if hq <= 0 or hkv <= 0 or hq % hkv or d != hq * HEAD or min(hidden, L, V, T) <= 0 or V < 256 or d < 256:
        sys.exit("make_model.py: need dim = heads * %d, heads a multiple of kv-heads, vocab >= 256, dim >= 256" % HEAD)
    dq, dkv = hq * HEAD, hkv * HEAD

    rng = SplitMix64(a.seed)
    tile = rng.uniforms(TILE) if a.fill == "tile" else None

    tmp = a.out + ".part"
    with open(tmp, "wb") as f:
        def tensor(n, scale, offset=0.0, rows=0, n_head=0):
            """Writes n values offset + scale * u; returns them (random fill) or one tile's worth."""
            if tile is not None:                             # the benchmark's model: one block, repeated
                block = struct.pack("<%df" % TILE, *[offset + scale * u for u in tile])
                f.write(block * (n // TILE))
                f.write(block[:4 * (n % TILE)])
                return [offset + scale * u for u in tile]
            vals = [offset + scale * u for u in rng.uniforms(n)]
            out = permute_rows(vals, rows, n // rows, n_head) if (a.permute_qk and n_head) else vals
            f.write(struct.pack("<%df" % n, *out))
            return vals

        f.write(struct.pack("<7i", d, hidden, L, hq, hkv, V, T))
        first = struct.pack("<256f", *tensor(V * d, 1.0)[:256])  # token_embedding_table
        for l in range(L):                                   # rms_att_weight
            tensor(d, 0.25, 1.0)
        for l in range(L):                                   # wq (permuted with --permute-qk)
            tensor(dq * d, a.qk_scale / d ** 0.5, rows=dq, n_head=hq)
        for l in range(L):                                   # wk (permuted with --permute-qk)
            tensor(dkv * d, a.qk_scale / d ** 0.5, rows=dkv, n_head=hkv)
        for l in range(L):                                   # wv
            tensor(dkv * d, a.block_scale / d ** 0.5)
        for l in range(L):                                   # wo
            tensor(d * dq, a.block_scale / dq ** 0.5)
        for l in range(L):                                   # rms_ffn_weight
            tensor(d, 0.25, 1.0)
        for l in range(L):                                   # w1
            tensor(hidden * d, a.block_scale / d ** 0.5)
        for l in range(L):                                   # w2
            tensor(d * hidden, a.block_scale / hidden ** 0.5)
        for l in range(L):                                   # w3
            tensor(hidden * d, a.block_scale / d ** 0.5)
        fin = tensor(d, 0.25, 1.0)                           # rms_final_weight
        last = struct.pack("<256f", *(fin[-256:] if tile is None else [fin[(d - 256 + i) % TILE] for i in range(256)]))
        zeros = struct.pack("<%df" % (T * HEAD // 2), *([0.0] * (T * HEAD // 2)))
        f.write(zeros)                                       # freq_cis_real (skipped by llama2.c)
        f.write(zeros)                                       # freq_cis_imag
    os.replace(tmp, a.out)

    expect = 28 + 4 * (V * d + L * d + L * d * dq + 2 * L * d * dkv + L * dq * d + L * d
                       + 3 * L * d * hidden + d + T * HEAD)
    size = os.path.getsize(a.out)
    if size != expect:
        sys.exit("make_model.py: wrote %d bytes, the E1 formula says %d" % (size, expect))
    print("model: dim %d, hidden %d, layers %d, heads %d, kv_heads %d, vocab %d, seq_len %d, seed %d%s%s"
          % (d, hidden, L, hq, hkv, V, T, a.seed, ", wq/wk permuted" if a.permute_qk else "",
             ", tiled fill" if tile else ""))
    print("bytes: %d" % size)
    print("checksum: %016x" % fnv1a64(last, fnv1a64(first)))


if __name__ == "__main__":
    main()
