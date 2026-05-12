#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <CommonCrypto/CommonDigest.h>

#define HASH_LEN 32
#define MAX_FILE_LEN 8

typedef struct {
    uint64_t start;
    uint64_t end;
    uint64_t length;
    uint8_t target_hash[HASH_LEN];
    uint64_t target_first8;
    atomic_uint_fast64_t *progress;
    atomic_int *found_flag;
    atomic_uint_fast64_t *found_index;
} worker_args_t;

static inline void index_to_bytes(uint64_t index, uint8_t *buf, int length) {
    for (int i = length - 1; i >= 0; i--) {
        buf[i] = (uint8_t)(index & 0xFF);
        index >>= 8;
    }
}

void *worker(void *arg) {
    worker_args_t *args = (worker_args_t *)arg;
    uint8_t candidate[MAX_FILE_LEN];
    uint8_t hash[HASH_LEN];
    uint64_t local_count = 0;

    for (uint64_t i = args->start; i < args->end; i++) {
        // periodically check if another thread already won, and flush progress
        if ((local_count & 0xFFFF) == 0xFFFF) {
            atomic_fetch_add_explicit(args->progress, local_count + 1,
                                       memory_order_relaxed);
            local_count = 0;
            if (atomic_load_explicit(args->found_flag, memory_order_relaxed)) {
                return NULL;
            }
            continue;
        }

        index_to_bytes(i, candidate, (int)args->length);
        CC_SHA256(candidate, (CC_LONG)args->length, hash);

        // compare first 8 bytes only. False-positive rate is
        // 1 in 2^64, so almost every non-match exits here.
        uint64_t first8;
        memcpy(&first8, hash, 8);
        if (first8 == args->target_first8) {
            if (memcmp(hash, args->target_hash, HASH_LEN) == 0) {
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

    uint64_t length = (uint64_t)length_ull;
    if (length > MAX_FILE_LEN) {
        fprintf(stderr, "file too long for this demo (max %d bytes)\n",
                MAX_FILE_LEN);
        return 1;
    }

    uint8_t target_hash[HASH_LEN];
    if (hex_to_bytes(hash_hex, target_hash, HASH_LEN) != 0) {
        fprintf(stderr, "bad hash\n");
        return 1;
    }

    uint64_t target_first8;
    memcpy(&target_first8, target_hash, 8);

    int num_threads = atoi(argv[3]);
    if (num_threads <= 0) num_threads = 8;

    uint64_t total = 1;
    for (uint64_t i = 0; i < length; i++) total *= 256;

    printf("target hash:  %s\n", hash_hex);
    printf("length:       %llu bytes\n", (unsigned long long)length);
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
        args[i].length = length;
        memcpy(args[i].target_hash, target_hash, HASH_LEN);
        args[i].target_first8 = target_first8;
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
        struct timespec sleep_time = {0, 100000000}; // 100ms
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

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_elapsed = (end.tv_sec - start.tv_sec) +
                           (end.tv_nsec - start.tv_nsec) / 1e9;

    if (is_tty) printf("\r\033[K");

    if (atomic_load(&found_flag)) {
        uint64_t idx = atomic_load(&found_index);
        uint8_t result[MAX_FILE_LEN];
        index_to_bytes(idx, result, (int)length);

        printf("\n>>> MATCH FOUND <<<\n");
        printf("index:   %llu\n", (unsigned long long)idx);
        printf("hex:     ");
        for (uint64_t i = 0; i < length; i++) printf("%02x", result[i]);
        printf("\n");
        printf("ascii:   ");
        for (uint64_t i = 0; i < length; i++) {
            printf("%c", (result[i] >= 32 && result[i] < 127) ? result[i] : '.');
        }
        printf("\n");
        printf("elapsed: %.3fs\n", total_elapsed);
        printf("rate:    %.1f M/s\n",
               (double)atomic_load(&progress) / total_elapsed / 1e6);

        FILE *out = fopen(argv[2], "wb");
        if (out) {
            fwrite(result, 1, length, out);
            fclose(out);
        }
    } else {
        printf("not found in search space\n");
        return 1;
    }

    return 0;
}
