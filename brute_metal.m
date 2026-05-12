#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static const char *kShaderSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"constant uint LENGTH [[function_constant(0)]];\n"
"\n"
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
"\n"
"constant uint H0[8] = {\n"
"    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,\n"
"    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u\n"
"};\n"
"\n"
"static inline uint rotr32(uint x, uint n) {\n"
"    return (x >> n) | (x << (32u - n));\n"
"}\n"
"\n"
"kernel void brute_force(\n"
"    constant ulong& base_index [[buffer(0)]],\n"
"    constant uint* target_state [[buffer(1)]],\n"
"    device atomic_uint* found_flag [[buffer(2)]],\n"
"    device atomic_uint* found_index [[buffer(3)]],\n" // [lo, hi]
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    ulong candidate = base_index + (ulong)tid;\n"
"    \n"
"    // Rolling window: only 16 W values live at once (vs 64). Frees registers for other work.\n"
"    uint W[16];\n"
"    W[2]  = 0u; W[3]  = 0u; W[4]  = 0u; W[5]  = 0u;\n"
"    W[6]  = 0u; W[7]  = 0u; W[8]  = 0u; W[9]  = 0u;\n"
"    W[10] = 0u; W[11] = 0u; W[12] = 0u; W[13] = 0u;\n"
"    W[14] = 0u;\n"
"    \n"
"    // LENGTH is a function constant, so dead branches are eliminated at compile time\n"
"    if (LENGTH == 4u) {\n"
"        W[0] = (uint)candidate;\n"
"        W[1] = 0x80000000u;\n"
"    } else if (LENGTH == 5u) {\n"
"        W[0] = (uint)(candidate >> 8);\n"
"        W[1] = ((uint)(candidate & 0xFFul) << 24) | 0x00800000u;\n"
"    } else if (LENGTH == 6u) {\n"
"        W[0] = (uint)(candidate >> 16);\n"
"        W[1] = ((uint)(candidate & 0xFFFFul) << 16) | 0x00008000u;\n"
"    } else if (LENGTH == 7u) {\n"
"        W[0] = (uint)(candidate >> 24);\n"
"        W[1] = ((uint)(candidate & 0xFFFFFFul) << 8) | 0x80u;\n"
"    }\n"
"    W[15] = LENGTH * 8u;\n"
"    \n"
"    uint a = H0[0], b = H0[1], c = H0[2], d = H0[3];\n"
"    uint e = H0[4], f = H0[5], g = H0[6], h = H0[7];\n"
"    \n"
"    // Rounds 0..15 use W[0..15] directly\n"
"    for (uint i = 0; i < 16; i++) {\n"
"        uint S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);\n"
"        uint ch = (e & f) ^ ((~e) & g);\n"
"        uint t1 = h + S1 + ch + K[i] + W[i];\n"
"        uint S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);\n"
"        uint mj = (a & b) ^ (a & c) ^ (b & c);\n"
"        uint t2 = S0 + mj;\n"
"        h = g; g = f; f = e;\n"
"        e = d + t1;\n"
"        d = c; c = b; b = a;\n"
"        a = t1 + t2;\n"
"    }\n"
"    \n"
"    // Rounds 16..63: compute W[i] from rolling window, do round, store W[i] back\n"
"    for (uint i = 16; i < 64; i++) {\n"
"        uint w_im16 = W[i & 15];\n"
"        uint w_im15 = W[(i + 1) & 15];\n"
"        uint w_im7  = W[(i + 9) & 15];\n"
"        uint w_im2  = W[(i + 14) & 15];\n"
"        uint s0 = rotr32(w_im15, 7) ^ rotr32(w_im15, 18) ^ (w_im15 >> 3);\n"
"        uint s1 = rotr32(w_im2, 17) ^ rotr32(w_im2, 19) ^ (w_im2 >> 10);\n"
"        uint w_i = w_im16 + s0 + w_im7 + s1;\n"
"        W[i & 15] = w_i;\n"
"        \n"
"        uint S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);\n"
"        uint ch = (e & f) ^ ((~e) & g);\n"
"        uint t1 = h + S1 + ch + K[i] + w_i;\n"
"        uint S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);\n"
"        uint mj = (a & b) ^ (a & c) ^ (b & c);\n"
"        uint t2 = S0 + mj;\n"
"        h = g; g = f; f = e;\n"
"        e = d + t1;\n"
"        d = c; c = b; b = a;\n"
"        a = t1 + t2;\n"
"    }\n"
"    \n"
"    a += H0[0]; b += H0[1]; c += H0[2]; d += H0[3];\n"
"    e += H0[4]; f += H0[5]; g += H0[6]; h += H0[7];\n"
"    \n"
"    if (a == target_state[0] && b == target_state[1] &&\n"
"        c == target_state[2] && d == target_state[3] &&\n"
"        e == target_state[4] && f == target_state[5] &&\n"
"        g == target_state[6] && h == target_state[7]) {\n"
"        atomic_store_explicit(&found_index[0], (uint)candidate, memory_order_relaxed);\n"
"        atomic_store_explicit(&found_index[1], (uint)(candidate >> 32), memory_order_relaxed);\n"
"        atomic_store_explicit(found_flag, 1u, memory_order_relaxed);\n"
"    }\n"
"}\n";

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s <compressed_file> <output_file>\n", argv[0]);
            return 1;
        }

        FILE *f = fopen(argv[1], "r");
        if (!f) { perror("open"); return 1; }
        char hash_hex[65];
        unsigned long long length;
        if (fscanf(f, "%64s\n%llu\n", hash_hex, &length) != 2) {
            fclose(f);
            fprintf(stderr, "bad format\n");
            return 1;
        }
        fclose(f);

        if (length < 4 || length > 7) {
            fprintf(stderr,
                "this GPU build supports lengths 4-7 (got %llu)\n", length);
            return 1;
        }

        uint32_t target_state[8];
        for (int i = 0; i < 8; i++) {
            unsigned int b0, b1, b2, b3;
            sscanf(hash_hex + i * 8, "%2x%2x%2x%2x", &b0, &b1, &b2, &b3);
            target_state[i] = ((uint32_t)b0 << 24) | ((uint32_t)b1 << 16) |
                              ((uint32_t)b2 << 8)  | (uint32_t)b3;
        }

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

        // function constants: length gets baked in at pipeline-compile time
        // so dead branches in the shader are eliminated.
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
            fprintf(stderr, "pipeline: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }

        // Use the largest threadgroup the pipeline allows for max occupancy.
        NSUInteger threads_per_group = pipeline.maxTotalThreadsPerThreadgroup;
        printf("threads/group: %lu  exec width: %lu\n",
               (unsigned long)threads_per_group,
               (unsigned long)pipeline.threadExecutionWidth);

        id<MTLBuffer> base_buf = [device newBufferWithLength:sizeof(uint64_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> target_buf = [device newBufferWithBytes:target_state
                                                        length:sizeof(target_state)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> flag_buf = [device newBufferWithLength:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> idx_buf = [device newBufferWithLength:2 * sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];

        *(uint32_t *)flag_buf.contents = 0;
        ((uint32_t *)idx_buf.contents)[0] = 0;
        ((uint32_t *)idx_buf.contents)[1] = 0;

        // total search space = 256^length
        uint64_t total = 1;
        for (unsigned long long i = 0; i < length; i++) total *= 256;

        // bigger batches → less dispatch overhead.
        // cap at 2^28 so progress updates feel reasonable and dispatch fits in uint32 grid.
        uint64_t batch_size = (1ULL << 28);
        if (batch_size > total) batch_size = total;

        printf("target hash:  %s\n", hash_hex);
        printf("length:       %llu bytes (Metal GPU)\n", length);
        printf("search space: %llu candidates\n", (unsigned long long)total);
        printf("batch size:   %llu\n\n", (unsigned long long)batch_size);

        struct timespec start;
        clock_gettime(CLOCK_MONOTONIC, &start);

        for (uint64_t base = 0; base < total; base += batch_size) {
            *(uint64_t *)base_buf.contents = base;
            uint64_t this_batch =
                (base + batch_size > total) ? (total - base) : batch_size;

            id<MTLCommandBuffer> cmdbuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:base_buf offset:0 atIndex:0];
            [encoder setBuffer:target_buf offset:0 atIndex:1];
            [encoder setBuffer:flag_buf offset:0 atIndex:2];
            [encoder setBuffer:idx_buf offset:0 atIndex:3];
            [encoder dispatchThreads:MTLSizeMake(this_batch, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
            [encoder endEncoding];
            [cmdbuf commit];
            [cmdbuf waitUntilCompleted];

            if (*(uint32_t *)flag_buf.contents != 0) break;

            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            double elapsed = (now.tv_sec - start.tv_sec) +
                             (now.tv_nsec - start.tv_nsec) / 1e9;
            uint64_t processed = base + this_batch;
            double rate = elapsed > 0 ? processed / elapsed : 0;
            double pct = 100.0 * processed / total;
            double eta = rate > 0 ? (total - processed) / rate : 0;
            printf("\r\033[K[%5.2f%%] %llu/%llu | %.0fM/s | %.1fs elapsed | ETA %.0fs",
                   pct,
                   (unsigned long long)processed, (unsigned long long)total,
                   rate / 1e6, elapsed, eta);
            fflush(stdout);
        }

        struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        double total_elapsed = (end.tv_sec - start.tv_sec) +
                               (end.tv_nsec - start.tv_nsec) / 1e9;

        printf("\r\033[K");

        if (*(uint32_t *)flag_buf.contents != 0) {
            uint32_t lo = ((uint32_t *)idx_buf.contents)[0];
            uint32_t hi = ((uint32_t *)idx_buf.contents)[1];
            uint64_t idx = ((uint64_t)hi << 32) | lo;
            uint8_t result[8] = {0};
            for (unsigned long long i = 0; i < length; i++) {
                result[i] = (uint8_t)(idx >> (8 * (length - 1 - i)));
            }

            printf("\n>>> MATCH FOUND <<<\n");
            printf("index:   %llu\n", (unsigned long long)idx);
            printf("hex:     ");
            for (unsigned long long i = 0; i < length; i++) printf("%02x", result[i]);
            printf("\n");
            printf("ascii:   ");
            for (unsigned long long i = 0; i < length; i++) {
                printf("%c", (result[i] >= 32 && result[i] < 127) ? result[i] : '.');
            }
            printf("\n");
            printf("elapsed: %.3fs\n", total_elapsed);
            printf("rate:    %.1f M/s\n",
                   (double)idx / total_elapsed / 1e6);

            FILE *out = fopen(argv[2], "wb");
            if (out) { fwrite(result, 1, length, out); fclose(out); }
        } else {
            printf("not found\n");
            return 1;
        }
    }
    return 0;
}
