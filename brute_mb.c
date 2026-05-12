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

#define ROTR_V(x, n) (vorrq_u32(vshrq_n_u32((x), (n)), vshlq_n_u32((x), 32 - (n))))

// process 4 SHA-256 hashes in parallel via SIMD lanes (one candidate per lane).
// specialized for 4-byte input.
static inline __attribute__((always_inline))
void sha256_4way_4byte(uint32x4_t inputs, uint32x4_t state_out[8]) {
    uint32x4_t a = vdupq_n_u32(H0_INIT[0]);
    uint32x4_t b = vdupq_n_u32(H0_INIT[1]);
    uint32x4_t c = vdupq_n_u32(H0_INIT[2]);
    uint32x4_t d = vdupq_n_u32(H0_INIT[3]);
    uint32x4_t e = vdupq_n_u32(H0_INIT[4]);
    uint32x4_t f = vdupq_n_u32(H0_INIT[5]);
    uint32x4_t g = vdupq_n_u32(H0_INIT[6]);
    uint32x4_t h = vdupq_n_u32(H0_INIT[7]);

    // rolling W window: 16 vec slots cover the full 64-round schedule.
    uint32x4_t W[16];
    W[0] = inputs;
    W[1] = vdupq_n_u32(0x80000000U);
    for (int i = 2; i < 15; i++) W[i] = vdupq_n_u32(0);
    W[15] = vdupq_n_u32(32U);

#define R(i, w) do { \
    uint32x4_t S1 = veorq_u32(veorq_u32(ROTR_V(e, 6), ROTR_V(e, 11)), ROTR_V(e, 25)); \
    uint32x4_t ch = vbslq_u32(e, f, g); \
    uint32x4_t t1 = vaddq_u32(vaddq_u32(vaddq_u32(vaddq_u32(h, S1), ch), vdupq_n_u32(K[i])), (w)); \
    uint32x4_t S0 = veorq_u32(veorq_u32(ROTR_V(a, 2), ROTR_V(a, 13)), ROTR_V(a, 22)); \
    uint32x4_t mj = vbslq_u32(veorq_u32(a, b), c, b); \
    uint32x4_t t2 = vaddq_u32(S0, mj); \
    h = g; g = f; f = e; \
    e = vaddq_u32(d, t1); \
    d = c; c = b; b = a; \
    a = vaddq_u32(t1, t2); \
} while (0)

    // Rounds 0-15 use W[0..15] directly.
    R(0, W[0]);  R(1, W[1]);  R(2, W[2]);  R(3, W[3]);
    R(4, W[4]);  R(5, W[5]);  R(6, W[6]);  R(7, W[7]);
    R(8, W[8]);  R(9, W[9]);  R(10, W[10]); R(11, W[11]);
    R(12, W[12]); R(13, W[13]); R(14, W[14]); R(15, W[15]);

    // rounds 16-63: rolling schedule + round, fully unrolled.
#define SCHED_AND_R(i) do { \
    uint32x4_t w16 = W[(i) & 15]; \
    uint32x4_t w15 = W[((i) + 1) & 15]; \
    uint32x4_t w7  = W[((i) + 9) & 15]; \
    uint32x4_t w2  = W[((i) + 14) & 15]; \
    uint32x4_t s0 = veorq_u32(veorq_u32(ROTR_V(w15, 7), ROTR_V(w15, 18)), vshrq_n_u32(w15, 3)); \
    uint32x4_t s1 = veorq_u32(veorq_u32(ROTR_V(w2, 17), ROTR_V(w2, 19)), vshrq_n_u32(w2, 10)); \
    uint32x4_t w_i = vaddq_u32(vaddq_u32(vaddq_u32(w16, s0), w7), s1); \
    W[(i) & 15] = w_i; \
    R((i), w_i); \
} while (0)

    SCHED_AND_R(16); SCHED_AND_R(17); SCHED_AND_R(18); SCHED_AND_R(19);
    SCHED_AND_R(20); SCHED_AND_R(21); SCHED_AND_R(22); SCHED_AND_R(23);
    SCHED_AND_R(24); SCHED_AND_R(25); SCHED_AND_R(26); SCHED_AND_R(27);
    SCHED_AND_R(28); SCHED_AND_R(29); SCHED_AND_R(30); SCHED_AND_R(31);
    SCHED_AND_R(32); SCHED_AND_R(33); SCHED_AND_R(34); SCHED_AND_R(35);
    SCHED_AND_R(36); SCHED_AND_R(37); SCHED_AND_R(38); SCHED_AND_R(39);
    SCHED_AND_R(40); SCHED_AND_R(41); SCHED_AND_R(42); SCHED_AND_R(43);
    SCHED_AND_R(44); SCHED_AND_R(45); SCHED_AND_R(46); SCHED_AND_R(47);
    SCHED_AND_R(48); SCHED_AND_R(49); SCHED_AND_R(50); SCHED_AND_R(51);
    SCHED_AND_R(52); SCHED_AND_R(53); SCHED_AND_R(54); SCHED_AND_R(55);
    SCHED_AND_R(56); SCHED_AND_R(57); SCHED_AND_R(58); SCHED_AND_R(59);
    SCHED_AND_R(60); SCHED_AND_R(61); SCHED_AND_R(62); SCHED_AND_R(63);

#undef R
#undef SCHED_AND_R

    state_out[0] = vaddq_u32(a, vdupq_n_u32(H0_INIT[0]));
    state_out[1] = vaddq_u32(b, vdupq_n_u32(H0_INIT[1]));
    state_out[2] = vaddq_u32(c, vdupq_n_u32(H0_INIT[2]));
    state_out[3] = vaddq_u32(d, vdupq_n_u32(H0_INIT[3]));
    state_out[4] = vaddq_u32(e, vdupq_n_u32(H0_INIT[4]));
    state_out[5] = vaddq_u32(f, vdupq_n_u32(H0_INIT[5]));
    state_out[6] = vaddq_u32(g, vdupq_n_u32(H0_INIT[6]));
    state_out[7] = vaddq_u32(h, vdupq_n_u32(H0_INIT[7]));
}

// self test: compare 4-way output against CC_SHA256 for known inputs.
static int verify(void) {
    uint32_t test[4] = {0x61626364, 0x776f7264, 0xdeadbeef, 0xcafebabe};
    uint32x4_t inputs = vld1q_u32(test);
    uint32x4_t state[8];
    sha256_4way_4byte(inputs, state);

    uint32_t out[8][4];
    for (int i = 0; i < 8; i++) vst1q_u32(out[i], state[i]);

    for (int j = 0; j < 4; j++) {
        uint32_t w = test[j];
        uint8_t bytes[4] = {
            (uint8_t)(w >> 24), (uint8_t)(w >> 16),
            (uint8_t)(w >> 8),  (uint8_t)w
        };
        uint8_t cc[32];
        CC_SHA256(bytes, 4, cc);

        for (int i = 0; i < 8; i++) {
            uint32_t expected = ((uint32_t)cc[i*4] << 24) |
                                ((uint32_t)cc[i*4+1] << 16) |
                                ((uint32_t)cc[i*4+2] << 8) |
                                cc[i*4+3];
            if (out[i][j] != expected) {
                fprintf(stderr,
                    "mismatch lane %d word %d: got %08x expected %08x (input %08x)\n",
                    j, i, out[i][j], expected, w);
                return -1;
            }
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
    const uint32x4_t target_a_vec = vdupq_n_u32(target_a);

    uint32x4_t state[8];
    uint64_t local_count = 0;

    for (uint64_t i = args->start; i + 4 <= args->end; i += 4) {
        if ((local_count & 0xFFFFF) < 4) {
            atomic_fetch_add_explicit(args->progress, local_count,
                                       memory_order_relaxed);
            if (atomic_load_explicit(args->found_flag,
                                      memory_order_relaxed)) {
                return NULL;
            }
            local_count = 0;
        }

        uint32_t cand[4] = {
            (uint32_t)i, (uint32_t)(i + 1),
            (uint32_t)(i + 2), (uint32_t)(i + 3)
        };
        uint32x4_t inputs = vld1q_u32(cand);

        sha256_4way_4byte(inputs, state);

        uint32x4_t cmp = vceqq_u32(state[0], target_a_vec);
        if (__builtin_expect(vmaxvq_u32(cmp) != 0, 0)) {
            uint32_t lanes[8][4];
            for (int s = 0; s < 8; s++) vst1q_u32(lanes[s], state[s]);

            for (int j = 0; j < 4; j++) {
                if (lanes[0][j] == target_a) {
                    uint32_t full[8] = {
                        lanes[0][j], lanes[1][j], lanes[2][j], lanes[3][j],
                        lanes[4][j], lanes[5][j], lanes[6][j], lanes[7][j]
                    };
                    if (memcmp(full, args->target_state, 32) == 0) {
                        atomic_store(args->found_flag, 1);
                        atomic_store(args->found_index, i + j);
                        atomic_fetch_add_explicit(args->progress,
                            local_count + 4, memory_order_relaxed);
                        return NULL;
                    }
                }
            }
        }

        local_count += 4;
    }
    atomic_fetch_add_explicit(args->progress, local_count,
                               memory_order_relaxed);
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
    if (verify() != 0) {
        fprintf(stderr, "multi-buffer self-test failed!\n");
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("open"); return 1; }
    char hash_hex[65];
    unsigned long long length;
    if (fscanf(f, "%64s\n%llu\n", hash_hex, &length) != 2) {
        fclose(f);
        return 1;
    }
    fclose(f);
    if (length != 4) {
        fprintf(stderr, "this build only supports 4-byte input\n");
        return 1;
    }

    uint8_t target_hash[HASH_LEN];
    if (hex_to_bytes(hash_hex, target_hash, HASH_LEN) != 0) return 1;
    uint32_t target_state[8];
    for (int i = 0; i < 8; i++) {
        target_state[i] = ((uint32_t)target_hash[i*4]     << 24) |
                          ((uint32_t)target_hash[i*4 + 1] << 16) |
                          ((uint32_t)target_hash[i*4 + 2] << 8)  |
                           (uint32_t)target_hash[i*4 + 3];
    }

    int num_threads = atoi(argv[3]);
    if (num_threads <= 0) num_threads = 8;

    uint64_t total = 0x100000000ULL;

    printf("target hash:  %s\n", hash_hex);
    printf("length:       4 bytes (4-way multi-buffer NEON)\n");
    printf("search space: %llu candidates\n", (unsigned long long)total);
    printf("threads:      %d (each processes 4 candidates per iter)\n\n",
           num_threads);

    atomic_uint_fast64_t progress = 0;
    atomic_int found_flag = 0;
    atomic_uint_fast64_t found_index = 0;

    pthread_t threads[num_threads];
    worker_args_t args[num_threads];
    uint64_t chunk = total / num_threads;
    chunk = (chunk / 4) * 4;  // align to 4

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
        if (out) { fwrite(result, 1, 4, out); fclose(out); }
    } else {
        printf("not found\n");
        return 1;
    }
    return 0;
}
