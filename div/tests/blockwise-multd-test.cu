#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-division.cu.h"
// #include "../sequential/helper.h"
// #include "../sequential/div.h"
#include "../../cuda/helper.h"

__global__ void CallMultD(
    uint32_t* u,
    uint32_t b,
    uint32_t* v,
    const uint32_t m) {
        // __shared__ uint64_t* buf = (uint64_t*)cudaMalloc(sizeof(uint64_t) * m);
        extern __shared__ char sh_mem[];
        volatile uint64_t* shmem_u64 = (uint64_t*)sh_mem;

        BlockwiseMultD<10>(u, b, v, shmem_u64, m);
    }

void printSlice(uint32_t* u, char name, int i, uint32_t m) {
    int min = i-3 < 0 ? 0 : i-3;
    int max = i+3 > m ? m : i+3;

    printf("%c[%u-%u]: [", name, min, max);
    for (int i = min; i < max; i++) {
        printf("%u, ", u[i]);
    }
    printf("]\n");
}

void multd(uint32_t* a, uint32_t b, uint32_t* r, uint32_t m)
{
    uint64_t buf[m];

    for (int i = 0; i < m; i++) {
        buf[i] = ((uint64_t)a[i]) * (uint64_t)b;
    }

    for (int i = 0; i < m - 1; i++) {
        buf[i + 1] += buf[i] >> 32;
    }

    for (int i = 0; i < m; i++) {
        r[i] = (uint32_t)buf[i];
    }
}



int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }

    const uint32_t m = 10;
    int size = m * sizeof(uint32_t);

    uint32_t* u = (uint32_t*)malloc(size);
    uint32_t* v = (uint32_t*)malloc(size);
    uint32_t* v_D;
    cudaMalloc(&v_D, size);

    randomInit<uint32_t>(u, m);
    cudaMemcpy(v_D, u, size, cudaMemcpyHostToDevice);

    multd(u, 8, u, m);

    int threadsPerBlock = 256;
    CallMultD<<<1, threadsPerBlock>>>(v_D, 8, v_D, m);
    cudaDeviceSynchronize();

    gpuAssert( cudaPeekAtLastError() );
    cudaMemcpy(v, v_D, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        if (v[i] != u[i]) {
            printf("INVALID AT INDEX %u: [%u/%u]\n", i, v[i], u[i]);
            printSlice(v, 'v', i, m);
            printSlice(u, 'u', i, m);

            // free(u);
            free(v);
            cudaFree(v_D);
            return 1;
        }
    }
    // free(u);
    free(v);
    cudaFree(v_D);
    printf("multd: VALID\n");
    return 0;
}








