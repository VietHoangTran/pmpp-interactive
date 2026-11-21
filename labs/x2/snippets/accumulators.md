# Accumulator fragments

The four fragments of section X2.6's bug spot, "Safe or unsafe?". Each shows one way to hold a
quantised dot product's sum; nothing else is meant to be complete. They are not compiled
(unverified). `q`, `x4`, `w2`, `x2`, `v` and `u` are the block's codes and activations, and
`dot4` is dp4a's arithmetic. `app/src/sim/qdot.ts` (`runFragment`, `FRAGMENT_MODEL`) models
their arithmetic, and `app/src/sim/qdot.snippets.test.ts` ties each fragment's accumulator
to that model. The section parses this file by its `##` headings: keep one fenced block per
fragment.

## metal-half-dot

Title: 1. Metal, dequantised to half
Language: MSL

```metal
float sumf = 0.0f;
for (int i = 0; i < 32; i += 4) {
    half4 w = half4(d) * (half4(q[i/4]) - 8.0h);
    sumf += dot(w, half4(x4[i/4]));
}
```

## cuda-hfma2

Title: 2. CUDA, __hfma2 over the row
Language: CUDA

```cuda
__half2 acc = __float2half2_rn(0.0f);
for (int i = 0; i < n / 2; ++i)
    acc = __hfma2(w2[i], x2[i], acc);
float y = __low2float(acc) + __high2float(acc);
```

## short-sum

Title: 3. A short block sum
Language: CUDA

```cuda
short sumi = 0;
for (int i = 0; i < 4; ++i) {
    sumi = dot4( v[i]       & 0x0F0F0F0F, u[i],     sumi);
    sumi = dot4((v[i] >> 4) & 0x0F0F0F0F, u[i + 4], sumi);
}
sumf += d4 * (sumi * d8 - 8.0f * s8);
```

## int-then-float

Title: 4. An int block sum, a float across blocks
Language: CUDA

```cuda
int sumi = 0;
for (int i = 0; i < 4; ++i) {
    sumi = dot4( v[i]       & 0x0F0F0F0F, u[i],     sumi);
    sumi = dot4((v[i] >> 4) & 0x0F0F0F0F, u[i + 4], sumi);
}
sumf += d4 * (sumi * d8 - 8.0f * s8);
```
