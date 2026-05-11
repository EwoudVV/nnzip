#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <arm_neon.h>

#define HASH_LEN 32

static const uint32_t K_CPU[64] = {
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

static const uint32_t H0_CPU[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// SHA-256 using ARMv8 hardware crypto instructions, 4-byte input.
static inline __attribute__((always_inline))
void sha256_4byte_hw(uint32_t input_word, uint32_t state_out[8]) {
    uint32x4_t state0 = vld1q_u32(H0_CPU);
    uint32x4_t state1 = vld1q_u32(H0_CPU + 4);
    const uint32x4_t saved0 = state0;
    const uint32x4_t saved1 = state1;

    uint32x4_t msg0 = {input_word, 0x80000000U, 0U, 0U};
    uint32x4_t msg1 = vdupq_n_u32(0);
    uint32x4_t msg2 = vdupq_n_u32(0);
    uint32x4_t msg3 = {0U, 0U, 0U, 32U};

    uint32x4_t tmp, prev;

#define R_SCHED(a, b, c, d, ki) do { \
    tmp = vaddq_u32((a), vld1q_u32(K_CPU + (ki))); \
    (a) = vsha256su0q_u32((a), (b)); \
    prev = state0; \
    state0 = vsha256hq_u32(state0, state1, tmp); \
    state1 = vsha256h2q_u32(state1, prev, tmp); \
    (a) = vsha256su1q_u32((a), (c), (d)); \
} while (0)

#define R_NOSCHED(a, ki) do { \
    tmp = vaddq_u32((a), vld1q_u32(K_CPU + (ki))); \
    prev = state0; \
    state0 = vsha256hq_u32(state0, state1, tmp); \
    state1 = vsha256h2q_u32(state1, prev, tmp); \
} while (0)

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
    R_NOSCHED(msg0, 48);
    R_NOSCHED(msg1, 52);
    R_NOSCHED(msg2, 56);
    R_NOSCHED(msg3, 60);

#undef R_SCHED
#undef R_NOSCHED

    state0 = vaddq_u32(state0, saved0);
    state1 = vaddq_u32(state1, saved1);
    vst1q_u32(state_out, state0);
    vst1q_u32(state_out + 4, state1);
}

// SHA-256 with HW SHA-2 instructions, 5-byte input.
// Same as 4-byte version except message block packs the 5 bytes into W[0..1].
static inline __attribute__((always_inline))
void sha256_5byte_hw(uint64_t input, uint32_t state_out[8]) {
    uint32x4_t state0 = vld1q_u32(H0_CPU);
    uint32x4_t state1 = vld1q_u32(H0_CPU + 4);
    const uint32x4_t saved0 = state0;
    const uint32x4_t saved1 = state1;

    // bytes 0..3 packed big-endian into W[0]; byte 4 + 0x80 padding into W[1]
    uint32_t w0 = (uint32_t)(input >> 8);
    uint32_t w1 = ((uint32_t)(input & 0xFFU) << 24) | 0x00800000U;

    uint32x4_t msg0 = {w0, w1, 0U, 0U};
    uint32x4_t msg1 = vdupq_n_u32(0);
    uint32x4_t msg2 = vdupq_n_u32(0);
    uint32x4_t msg3 = {0U, 0U, 0U, 40U};  // 5 bytes = 40 bits

    uint32x4_t tmp, prev;

#define R_SCHED(a, b, c, d, ki) do { \
    tmp = vaddq_u32((a), vld1q_u32(K_CPU + (ki))); \
    (a) = vsha256su0q_u32((a), (b)); \
    prev = state0; \
    state0 = vsha256hq_u32(state0, state1, tmp); \
    state1 = vsha256h2q_u32(state1, prev, tmp); \
    (a) = vsha256su1q_u32((a), (c), (d)); \
} while (0)

#define R_NOSCHED(a, ki) do { \
    tmp = vaddq_u32((a), vld1q_u32(K_CPU + (ki))); \
    prev = state0; \
    state0 = vsha256hq_u32(state0, state1, tmp); \
    state1 = vsha256h2q_u32(state1, prev, tmp); \
} while (0)

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
    R_NOSCHED(msg0, 48);
    R_NOSCHED(msg1, 52);
    R_NOSCHED(msg2, 56);
    R_NOSCHED(msg3, 60);

#undef R_SCHED
#undef R_NOSCHED

    state0 = vaddq_u32(state0, saved0);
    state1 = vaddq_u32(state1, saved1);
    vst1q_u32(state_out, state0);
    vst1q_u32(state_out + 4, state1);
}

typedef struct {
    uint64_t start;
    uint64_t end;
    uint32_t target_state[8];
    int length;
    atomic_uint_fast64_t *progress;
    atomic_int *found_flag;
    atomic_uint_fast64_t *found_index;
} cpu_worker_args_t;

void *cpu_worker(void *arg) {
    cpu_worker_args_t *args = (cpu_worker_args_t *)arg;
    const uint32_t target_a = args->target_state[0];
    const int length = args->length;
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

        // length never changes; branch predictor handles this perfectly
        if (length == 4) {
            sha256_4byte_hw((uint32_t)i, state);
        } else {
            sha256_5byte_hw(i, state);
        }
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
    atomic_fetch_add_explicit(args->progress, local_count,
                               memory_order_relaxed);
    return NULL;
}

static const char *kShaderSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint LENGTH [[function_constant(0)]];\n"
"constant uint K[64] = {\n"
"    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,\n"
"    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,\n"
"    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,\n"
"    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,\n"
"    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,\n"
"    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,\n"
"    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,\n"
"    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,\n"
"    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,\n"
"    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,\n"
"    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,\n"
"    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,\n"
"    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,\n"
"    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,\n"
"    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,\n"
"    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u\n"
"};\n"
"constant uint H0[8] = {\n"
"    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,\n"
"    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u\n"
"};\n"
"static inline uint rotr32(uint x, uint n) { return (x >> n) | (x << (32u - n)); }\n"
"\n"
"kernel void brute_force(\n"
"    constant ulong& base_index [[buffer(0)]],\n"
"    constant uint* target_state [[buffer(1)]],\n"
"    device atomic_uint* found_flag [[buffer(2)]],\n"
"    device atomic_uint* found_index [[buffer(3)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    ulong candidate = base_index + (ulong)tid;\n"
"    uint W[16];\n"
"    W[2]=W[3]=W[4]=W[5]=W[6]=W[7]=0u;\n"
"    W[8]=W[9]=W[10]=W[11]=W[12]=W[13]=W[14]=0u;\n"
"    if (LENGTH == 4u) { W[0] = (uint)candidate; W[1] = 0x80000000u; }\n"
"    else if (LENGTH == 5u) { W[0] = (uint)(candidate >> 8); W[1] = ((uint)(candidate & 0xFFul) << 24) | 0x00800000u; }\n"
"    else if (LENGTH == 6u) { W[0] = (uint)(candidate >> 16); W[1] = ((uint)(candidate & 0xFFFFul) << 16) | 0x00008000u; }\n"
"    else if (LENGTH == 7u) { W[0] = (uint)(candidate >> 24); W[1] = ((uint)(candidate & 0xFFFFFFul) << 8) | 0x80u; }\n"
"    W[15] = LENGTH * 8u;\n"
"    uint a=H0[0], b=H0[1], c=H0[2], d=H0[3], e=H0[4], f=H0[5], g=H0[6], h=H0[7];\n"
"    for (uint i = 0; i < 16; i++) {\n"
"        uint S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);\n"
"        uint ch = (e & f) ^ ((~e) & g);\n"
"        uint t1 = h + S1 + ch + K[i] + W[i];\n"
"        uint S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);\n"
"        uint mj = (a & b) ^ (a & c) ^ (b & c);\n"
"        uint t2 = S0 + mj;\n"
"        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;\n"
"    }\n"
"    for (uint i = 16; i < 64; i++) {\n"
"        uint w16=W[i&15], w15=W[(i+1)&15], w7=W[(i+9)&15], w2=W[(i+14)&15];\n"
"        uint s0 = rotr32(w15,7)^rotr32(w15,18)^(w15>>3);\n"
"        uint s1 = rotr32(w2,17)^rotr32(w2,19)^(w2>>10);\n"
"        uint w_i = w16 + s0 + w7 + s1;\n"
"        W[i&15] = w_i;\n"
"        uint S1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);\n"
"        uint ch = (e & f) ^ ((~e) & g);\n"
"        uint t1 = h + S1 + ch + K[i] + w_i;\n"
"        uint S0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);\n"
"        uint mj = (a & b) ^ (a & c) ^ (b & c);\n"
"        uint t2 = S0 + mj;\n"
"        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;\n"
"    }\n"
"    a+=H0[0]; b+=H0[1]; c+=H0[2]; d+=H0[3];\n"
"    e+=H0[4]; f+=H0[5]; g+=H0[6]; h+=H0[7];\n"
"    if (a==target_state[0] && b==target_state[1] && c==target_state[2] && d==target_state[3] &&\n"
"        e==target_state[4] && f==target_state[5] && g==target_state[6] && h==target_state[7]) {\n"
"        atomic_store_explicit(&found_index[0], (uint)candidate, memory_order_relaxed);\n"
"        atomic_store_explicit(&found_index[1], (uint)(candidate >> 32), memory_order_relaxed);\n"
"        atomic_store_explicit(found_flag, 1u, memory_order_relaxed);\n"
"    }\n"
"}\n";

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 4) {
            fprintf(stderr,
                "usage: %s <compressed_file> <output_file> <num_cpu_threads>\n",
                argv[0]);
            return 1;
        }

        FILE *f = fopen(argv[1], "r");
        if (!f) { perror("open"); return 1; }
        char hash_hex[65];
        unsigned long long length;
        if (fscanf(f, "%64s\n%llu\n", hash_hex, &length) != 2) {
            fclose(f); return 1;
        }
        fclose(f);
        if (length != 4 && length != 5) {
            fprintf(stderr,
                "combined runner supports 4 or 5 byte input (got %llu). "
                "Use ./brute_metal for other sizes.\n", length);
            return 1;
        }

        uint8_t target_hash[HASH_LEN];
        for (int i = 0; i < HASH_LEN; i++) {
            unsigned int b;
            sscanf(hash_hex + 2*i, "%2x", &b);
            target_hash[i] = (uint8_t)b;
        }
        uint32_t target_state[8];
        for (int i = 0; i < 8; i++) {
            target_state[i] = ((uint32_t)target_hash[i*4]     << 24) |
                              ((uint32_t)target_hash[i*4 + 1] << 16) |
                              ((uint32_t)target_hash[i*4 + 2] << 8)  |
                               (uint32_t)target_hash[i*4 + 3];
        }

        int num_cpu_threads = atoi(argv[3]);
        if (num_cpu_threads <= 0) num_cpu_threads = 8;

        // Setup Metal
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "no Metal device\n"); return 1; }
        printf("Metal device: %s\n", [[device name] UTF8String]);

        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:kShaderSource];
        id<MTLLibrary> library = [device newLibraryWithSource:src
                                                       options:nil
                                                         error:&error];
        if (!library) {
            fprintf(stderr, "shader compile: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }

        MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
        uint32_t length_val = (uint32_t)length;
        [constants setConstantValue:&length_val type:MTLDataTypeUInt atIndex:0];
        id<MTLFunction> kernel = [library newFunctionWithName:@"brute_force"
                                              constantValues:constants
                                                       error:&error];
        if (!kernel) {
            fprintf(stderr, "function: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:kernel error:&error];
        if (!pipeline) {
            fprintf(stderr, "pipeline error\n"); return 1;
        }
        NSUInteger threads_per_group = pipeline.maxTotalThreadsPerThreadgroup;

        id<MTLBuffer> base_buf = [device newBufferWithLength:sizeof(uint64_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> target_buf = [device newBufferWithBytes:target_state
                                                        length:sizeof(target_state)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> gpu_flag_buf = [device newBufferWithLength:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> gpu_idx_buf = [device newBufferWithLength:2*sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
        *(uint32_t *)gpu_flag_buf.contents = 0;
        ((uint32_t *)gpu_idx_buf.contents)[0] = 0;
        ((uint32_t *)gpu_idx_buf.contents)[1] = 0;

        // Search space and split
        uint64_t total = 1;
        for (unsigned long long i = 0; i < length; i++) total *= 256;
        // GPU is ~2.7x faster than CPU. Give GPU ~73% of work.
        uint64_t gpu_share = (total * 73) / 100;
        gpu_share = (gpu_share / threads_per_group) * threads_per_group;
        uint64_t cpu_start = gpu_share;
        uint64_t cpu_share = total - cpu_start;

        printf("target hash:  %s\n", hash_hex);
        printf("length:       %llu bytes (CPU NEON-HW + GPU Metal concurrent)\n", length);
        printf("search space: %llu candidates\n", (unsigned long long)total);
        printf("split:        GPU [0..%llu)  CPU [%llu..%llu)\n",
               (unsigned long long)gpu_share,
               (unsigned long long)cpu_start, (unsigned long long)total);
        printf("threads:      %d CPU + GPU (threads/group %lu)\n\n",
               num_cpu_threads, (unsigned long)threads_per_group);

        // Shared atomic state (visible to both CPU threads and main GPU loop)
        atomic_int found_flag = 0;
        atomic_uint_fast64_t found_index = 0;
        atomic_uint_fast64_t cpu_progress = 0;

        // Spawn CPU worker threads on [cpu_start, total)
        pthread_t cpu_threads[num_cpu_threads];
        cpu_worker_args_t cpu_args[num_cpu_threads];
        uint64_t cpu_chunk = cpu_share / num_cpu_threads;

        struct timespec start;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (int i = 0; i < num_cpu_threads; i++) {
            cpu_args[i].start = cpu_start + (uint64_t)i * cpu_chunk;
            cpu_args[i].end = (i == num_cpu_threads - 1)
                ? total : cpu_start + (uint64_t)(i + 1) * cpu_chunk;
            memcpy(cpu_args[i].target_state, target_state, sizeof(target_state));
            cpu_args[i].length = (int)length;
            cpu_args[i].progress = &cpu_progress;
            cpu_args[i].found_flag = &found_flag;
            cpu_args[i].found_index = &found_index;
            pthread_create(&cpu_threads[i], NULL, cpu_worker, &cpu_args[i]);
        }

        // GPU dispatch loop on [0, gpu_share)
        const uint64_t batch_size = 1ULL << 27; // 128M
        uint64_t gpu_processed = 0;
        int is_tty = isatty(STDOUT_FILENO);
        struct timespec last_print = start;

        for (uint64_t base = 0; base < gpu_share; base += batch_size) {
            // Stop if either side already found
            if (atomic_load(&found_flag)) break;

            *(uint64_t *)base_buf.contents = base;
            uint64_t this_batch =
                (base + batch_size > gpu_share) ? (gpu_share - base) : batch_size;

            id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:base_buf offset:0 atIndex:0];
            [encoder setBuffer:target_buf offset:0 atIndex:1];
            [encoder setBuffer:gpu_flag_buf offset:0 atIndex:2];
            [encoder setBuffer:gpu_idx_buf offset:0 atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(this_batch, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [encoder endEncoding];
            [cmdbuf commit];
            [cmdbuf waitUntilCompleted];
            gpu_processed = base + this_batch;

            // Check if GPU found something this batch
            if (*(uint32_t *)gpu_flag_buf.contents != 0) {
                uint32_t lo = ((uint32_t *)gpu_idx_buf.contents)[0];
                uint32_t hi = ((uint32_t *)gpu_idx_buf.contents)[1];
                uint64_t idx = ((uint64_t)hi << 32) | lo;
                // Set shared flag so CPU threads stop quickly
                atomic_store(&found_index, idx);
                atomic_store(&found_flag, 1);
                break;
            }

            // Progress
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - start.tv_sec) +
                             (now.tv_nsec - start.tv_nsec) / 1e9;
            uint64_t cpu_done = atomic_load(&cpu_progress);
            uint64_t total_done = gpu_processed + cpu_done;
            double rate = elapsed > 0 ? total_done / elapsed : 0;
            double pct = 100.0 * total_done / total;
            double eta = rate > 0 ? (total - total_done) / rate : 0;
            if (is_tty) {
                printf("\r\033[K[%5.2f%%] GPU %llu CPU %llu | combined %.0fM/s | %.1fs | ETA %.0fs",
                       pct,
                       (unsigned long long)gpu_processed,
                       (unsigned long long)cpu_done,
                       rate / 1e6, elapsed, eta);
                fflush(stdout);
            } else {
                double dt = (now.tv_sec - last_print.tv_sec) +
                            (now.tv_nsec - last_print.tv_nsec) / 1e9;
                if (dt >= 0.5) {
                    printf("  [%5.1f%%] GPU %llu CPU %llu | %.0fM/s | %.1fs\n",
                           pct,
                           (unsigned long long)gpu_processed,
                           (unsigned long long)cpu_done,
                           rate / 1e6, elapsed);
                    last_print = now;
                }
            }
        }

        // Wait for CPU threads (they exit when found_flag set or their range done)
        for (int i = 0; i < num_cpu_threads; i++) {
            pthread_join(cpu_threads[i], NULL);
        }

        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        double total_elapsed = (end.tv_sec - start.tv_sec) +
                               (end.tv_nsec - start.tv_nsec) / 1e9;

        if (is_tty) printf("\r\033[K");

        if (atomic_load(&found_flag)) {
            uint64_t idx = atomic_load(&found_index);
            uint8_t result[8] = {0};
            for (unsigned long long i = 0; i < length; i++) {
                result[i] = (uint8_t)(idx >> (8 * (length - 1 - i)));
            }

            uint64_t cpu_done = atomic_load(&cpu_progress);
            uint64_t total_done = gpu_processed + cpu_done;

            printf("\n>>> MATCH FOUND <<<\n");
            printf("index:    %llu\n", (unsigned long long)idx);
            printf("hex:      ");
            for (unsigned long long i = 0; i < length; i++) printf("%02x", result[i]);
            printf("\n");
            printf("ascii:    ");
            for (unsigned long long i = 0; i < length; i++) {
                printf("%c", (result[i] >= 32 && result[i] < 127) ? result[i] : '.');
            }
            printf("\n");
            printf("found by: %s\n",
                   (idx < gpu_share) ? "GPU" : "CPU");
            printf("elapsed:  %.3fs\n", total_elapsed);
            printf("total:    %llu candidates checked (GPU %llu, CPU %llu)\n",
                   (unsigned long long)total_done,
                   (unsigned long long)gpu_processed,
                   (unsigned long long)cpu_done);
            printf("rate:     %.1f M/s combined\n",
                   (double)total_done / total_elapsed / 1e6);

            FILE *out = fopen(argv[2], "wb");
            if (out) { fwrite(result, 1, length, out); fclose(out); }
        } else {
            printf("not found\n");
            return 1;
        }
    }
    return 0;
}