#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <arm_neon.h>
#include <CommonCrypto/CommonDigest.h>

#define HASH_LEN 32

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static const uint32_t H0_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// compute SHA-256 of exactly 4 input bytes (packed big-endian into input_word).
// output: 8 hash words in state_out (each word = big-endian read of 4 hash bytes).
static inline __attribute__((always_inline))
void sha256_4byte(uint32_t input_word, uint32_t state_out[8]) {
    uint32x4_t state0 = vld1q_u32(H0_INIT);
    uint32x4_t state1 = vld1q_u32(H0_INIT + 4);
    const uint32x4_t saved0 = state0;
    const uint32x4_t saved1 = state1;

    // message block for 4-byte input:
    //   W[0]  = input bytes (big-endian)
    //   W[1]  = 0x80000000  (1-bit pad followed by zeros)
    //   W[2..14] = 0
    //   W[15] = 32           (bit-length of input)
    uint32x4_t msg0 = {input_word, 0x80000000U, 0U, 0U};
    uint32x4_t msg1 = vdupq_n_u32(0);
    uint32x4_t msg2 = vdupq_n_u32(0);
    uint32x4_t msg3 = {0U, 0U, 0U, 32U};

    uint32x4_t tmp, prev;

#define R_SCHED(a, b, c, d, ki) do { \
    tmp = vaddq_u32((a), vld1q_u32(K + (ki))); \
    (a) = vsha256su0q_u32((a), (b)); \
    prev = state0; \
    state0 = vsha256hq_u32(state0, state1, tmp); \
    state1 = vsha256h2q_u32(state1, prev, tmp); \
    (a) = vsha256su1q_u32((a), (c), (d)); \
} while (0)

#define R_NOSCHED(a, ki) do { \
    tmp = vaddq_u32((a), vld1q_u32(K + (ki))); \
    prev = state0; \
    state0 = vsha256hq_u32(state0, state1, tmp); \
    state1 = vsha256h2q_u32(state1, prev, tmp); \
} while (0)

    // rounds 0-47 with schedule (12 iterations)
    R_SCHED(msg0, msg1, msg2, msg3,  0);
    R_SCHED(msg1, msg2, msg3, msg0,  4);
    R_SCHED(msg2, msg3, msg0, msg1,  8);
    R_SCHED(msg3, msg0, msg1, msg2, 12);
    R_SCHED(msg0, msg1, msg2, msg3, 16);
    R_SCHED(msg1, msg2, msg3, msg0, 20);
    R_SCHED(msg2, msg3, msg0, msg1, 24);
    R_SCHED(msg3, msg0, msg1, msg2, 28);
    R_SCHED(msg0, msg1, msg2, msg3, 32);
    R_SCHED(msg1, msg2, msg3, msg0, 36);
    R_SCHED(msg2, msg3, msg0, msg1, 40);
    R_SCHED(msg3, msg0, msg1, msg2, 44);

    // rounds 48-63 no schedule
    R_NOSCHED(msg0, 48);
    R_NOSCHED(msg1, 52);
    R_NOSCHED(msg2, 56);
    R_NOSCHED(msg3, 60);

#undef R_SCHED
#undef R_NOSCHED

    // add saved (initial) state
    state0 = vaddq_u32(state0, saved0);
    state1 = vaddq_u32(state1, saved1);

    vst1q_u32(state_out, state0);
    vst1q_u32(state_out + 4, state1);
}

// check NEON SHA-256 against Apples CommonCrypto on random inputs.
static int verify_sha256(void) {
    uint32_t test_inputs[] = {
        0x00000000, 0x61626364, 0x776f7264, 0xdeadbeef,
        0xffffffff, 0x12345678, 0xcafebabe, 0x80808080,
    };
    for (size_t t = 0; t < sizeof(test_inputs) / sizeof(test_inputs[0]); t++) {
        uint32_t w = test_inputs[t];
        uint8_t input_bytes[4] = {
            (uint8_t)(w >> 24), (uint8_t)(w >> 16),
            (uint8_t)(w >> 8),  (uint8_t)w
        };
        uint8_t cc_hash[32];
        CC_SHA256(input_bytes, 4, cc_hash);

        uint32_t neon_state[8];
        sha256_4byte(w, neon_state);
        uint8_t neon_hash[32];
        for (int i = 0; i < 8; i++) {
            neon_hash[i*4]     = (uint8_t)(neon_state[i] >> 24);
            neon_hash[i*4 + 1] = (uint8_t)(neon_state[i] >> 16);
            neon_hash[i*4 + 2] = (uint8_t)(neon_state[i] >> 8);
            neon_hash[i*4 + 3] = (uint8_t)neon_state[i];
        }
        if (memcmp(cc_hash, neon_hash, 32) != 0) {
            fprintf(stderr, "SHA mismatch for input 0x%08x\n", w);
            fprintf(stderr, "  CC:   ");
            for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", cc_hash[i]);
            fprintf(stderr, "\n  NEON: ");
            for (int i = 0; i < 32; i++) fprintf(stderr, "%02x", neon_hash[i]);
            fprintf(stderr, "\n");
            return -1;
        }
    }
    return 0;
}

typedef struct {
    uint64_t start;
    uint64_t end;
    uint32_t target_state[8];
    atomic_uint_fast64_t *progress;
    atomic_int *found_flag;
    atomic_uint_fast64_t *found_index;
} worker_args_t;

void *worker(void *arg) {
    worker_args_t *args = (worker_args_t *)arg;
    const uint32_t target_a = args->target_state[0];
    uint32_t state[8];
    uint64_t local_count = 0;

    for (uint64_t i = args->start; i < args->end; i++) {
        if ((local_count & 0xFFFFF) == 0xFFFFF) {
            atomic_fetch_add_explicit(args->progress, local_count + 1,
                                       memory_order_relaxed);
            local_count = 0;
            if (atomic_load_explicit(args->found_flag,
                                      memory_order_relaxed)) {
                return NULL;
            }
            continue;
        }

        sha256_4byte((uint32_t)i, state);

        // compare first 32-bit hash word ('a').
        // false positive rate is 1 in 2^32, so almost every non-match exits here.
        if (__builtin_expect(state[0] == target_a, 0)) {
            if (memcmp(state, args->target_state, 32) == 0) {
                atomic_store(args->found_flag, 1);
                atomic_store(args->found_index, i);
                atomic_fetch_add_explicit(args->progress, local_count + 1,
                                           memory_order_relaxed);
                return NULL;
            }
        }
        local_count++;
    }
    atomic_fetch_add_explicit(args->progress, local_count, memory_order_relaxed);
    return NULL;
}

int hex_to_bytes(const char *hex, uint8_t *bytes, int len) {
    for (int i = 0; i < len; i++) {
        unsigned int b;
        if (sscanf(hex + 2 * i, "%2x", &b) != 1) return -1;
        bytes[i] = (uint8_t)b;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr,
                "usage: %s <compressed_file> <output_file> <num_threads>\n",
                argv[0]);
        return 1;
    }

    if (verify_sha256() != 0) {
        fprintf(stderr, "NEON SHA-256 self-test failed!\n");
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("open compressed"); return 1; }

    char hash_hex[65];
    unsigned long long length_ull;
    if (fscanf(f, "%64s\n%llu\n", hash_hex, &length_ull) != 2) {
        fprintf(stderr, "bad format\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    if (length_ull != 4) {
        fprintf(stderr, "this NEON build only supports 4-byte inputs (got %llu)\n",
                length_ull);
        fprintf(stderr, "use ./brute for other sizes\n");
        return 1;
    }

    uint8_t target_hash[HASH_LEN];
    if (hex_to_bytes(hash_hex, target_hash, HASH_LEN) != 0) {
        fprintf(stderr, "bad hash\n");
        return 1;
    }

    // convert target hash bytes to 8 big-endian uint32 words for fast compare.
    uint32_t target_state[8];
    for (int i = 0; i < 8; i++) {
        target_state[i] = ((uint32_t)target_hash[i*4]     << 24) |
                          ((uint32_t)target_hash[i*4 + 1] << 16) |
                          ((uint32_t)target_hash[i*4 + 2] << 8)  |
                           (uint32_t)target_hash[i*4 + 3];
    }

    int num_threads = atoi(argv[3]);
    if (num_threads <= 0) num_threads = 8;

    uint64_t total = 0x100000000ULL;  // 2^32

    printf("target hash:  %s\n", hash_hex);
    printf("length:       4 bytes (NEON-specialized)\n");
    printf("search space: %llu candidates\n", (unsigned long long)total);
    printf("threads:      %d\n\n", num_threads);

    atomic_uint_fast64_t progress = 0;
    atomic_int found_flag = 0;
    atomic_uint_fast64_t found_index = 0;

    pthread_t threads[num_threads];
    worker_args_t args[num_threads];
    uint64_t chunk = total / num_threads;

    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++) {
        args[i].start = (uint64_t)i * chunk;
        args[i].end = (i == num_threads - 1) ? total : (uint64_t)(i + 1) * chunk;
        memcpy(args[i].target_state, target_state, sizeof(target_state));
        args[i].progress = &progress;
        args[i].found_flag = &found_flag;
        args[i].found_index = &found_index;
        if (pthread_create(&threads[i], NULL, worker, &args[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    uint64_t last_progress = 0;
    struct timespec last_time = start;
    int is_tty = isatty(STDOUT_FILENO);
    double last_print_pct = 0;

    while (1) {
        struct timespec sleep_time = {0, 100000000};
        nanosleep(&sleep_time, NULL);

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - start.tv_sec) +
                         (now.tv_nsec - start.tv_nsec) / 1e9;
        double dt = (now.tv_sec - last_time.tv_sec) +
                    (now.tv_nsec - last_time.tv_nsec) / 1e9;

        uint64_t cur = atomic_load(&progress);
        double rate = dt > 0 ? (cur - last_progress) / dt : 0;
        double pct = total > 0 ? 100.0 * cur / total : 0;
        double eta = rate > 0 ? (total - cur) / rate : 0;

        if (is_tty) {
            printf("\r\033[K[%5.2f%%] %llu/%llu | %.1fM/s | elapsed %.1fs | ETA %.0fs",
                   pct, (unsigned long long)cur, (unsigned long long)total,
                   rate / 1e6, elapsed, eta);
            fflush(stdout);
        } else if (pct - last_print_pct >= 5.0) {
            printf("  [%5.1f%%] %llu | %.1fM/s | %.1fs elapsed\n",
                   pct, (unsigned long long)cur, rate / 1e6, elapsed);
            last_print_pct = pct;
        }

        last_progress = cur;
        last_time = now;

        if (atomic_load(&found_flag)) break;
        if (cur >= total) break;
    }

    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_elapsed = (end.tv_sec - start.tv_sec) +
                           (end.tv_nsec - start.tv_nsec) / 1e9;

    if (is_tty) printf("\r\033[K");

    if (atomic_load(&found_flag)) {
        uint64_t idx = atomic_load(&found_index);
        uint8_t result[4] = {
            (uint8_t)(idx >> 24), (uint8_t)(idx >> 16),
            (uint8_t)(idx >> 8),  (uint8_t)idx
        };

        printf("\n>>> MATCH FOUND <<<\n");
        printf("index:   %llu\n", (unsigned long long)idx);
        printf("hex:     %02x%02x%02x%02x\n",
               result[0], result[1], result[2], result[3]);
        printf("ascii:   ");
        for (int i = 0; i < 4; i++) {
            printf("%c", (result[i] >= 32 && result[i] < 127) ? result[i] : '.');
        }
        printf("\n");
        printf("elapsed: %.3fs\n", total_elapsed);
        printf("rate:    %.1f M/s\n",
               (double)atomic_load(&progress) / total_elapsed / 1e6);

        FILE *out = fopen(argv[2], "wb");
        if (out) {
            fwrite(result, 1, 4, out);
            fclose(out);
        }
    } else {
        printf("not found\n");
        return 1;
    }

    return 0;
}
