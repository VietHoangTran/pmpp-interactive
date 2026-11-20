// Check a llama2.c checkpoint ("version 0" .bin) before loading it: read the 28-byte
// header, predict the file size from it, and compare with the size on disk. Ours (module
// X3, section X3.8); the layout is llama2.c's run.c, memory_map_weights
// (notes/today/x3-facts.md D1, E1). Like run.c it reads the header with fread, so the
// host must store int32 in the file's byte order.
// Build: c++ -std=c++17 -O2 ckpt_check.cpp -o ckpt_check     Run: ./ckpt_check model.bin
#include <cstdint>
#include <cstdio>

struct Config {
    int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
};

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s model.bin\n", argv[0]); return 2; }
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror(argv[1]); return 2; }
    Config c;
    const bool whole = std::fread(&c, sizeof c, 1, f) == 1;
    std::fseek(f, 0, SEEK_END);
    const long long actual = std::ftell(f);
    std::fclose(f);
    if (!whole) { std::printf("%lld bytes: shorter than a header\n", actual); return 1; }
    if (c.dim <= 0 || c.n_heads <= 0 || c.n_kv_heads <= 0 || c.dim % c.n_heads || c.n_heads % c.n_kv_heads) {
        std::printf("not a usable header: dim %d, n_heads %d, n_kv_heads %d\n", c.dim, c.n_heads, c.n_kv_heads);
        return 1;
    }
    const bool shared = c.vocab_size > 0;           // negative: a separate classifier follows
    const long long V = shared ? c.vocab_size : -(long long)c.vocab_size;
    const long long d = c.dim, L = c.n_layers, hidden = c.hidden_dim;
    const long long head = d / c.n_heads;
    const long long kv_dim = d * c.n_kv_heads / c.n_heads;
    long long floats = V * d                        // token_embedding_table
        + L * d                                     // rms_att_weight
        + L * d * d + 2 * L * kv_dim * d + L * d * d  // wq; wk and wv; wo
        + L * d                                     // rms_ffn_weight
        + 3 * L * d * hidden                        // w1, w2, w3
        + d                                         // rms_final_weight
        + c.seq_len * head;                         // two old RoPE tables, which run.c skips
    if (!shared) floats += V * d;                   // wcls
    const long long expected = 28 + 4 * floats;
    std::printf("dim %d, hidden_dim %d, n_layers %d, n_heads %d, n_kv_heads %d, vocab_size %d, seq_len %d\n",
                c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size, c.seq_len);
    std::printf("head size %lld, %lld query heads per KV head, classifier %s\n",
                head, d / head / c.n_kv_heads, shared ? "shared" : "unshared");
    std::printf("expected %lld bytes, file has %lld: %s\n", expected, actual, expected == actual ? "ok" : "MISMATCH");
    return expected == actual ? 0 : 1;
}
