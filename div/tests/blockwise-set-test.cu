#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-division.cu.h"
#include "../../cuda/helper.h"


template<class T, uint32_t Q>
__global__ void CallSet(
    T* u,
    const T d,
    const T m) {
        extern __shared__ char sh_mem[];
        volatile T* shmem_u32 = (T*)sh_mem;

        copyFromGlb2ShrMem<T, Q>(0, m, 0, u, shmem_u32);
        set<T, Q>(shmem_u32, d, m);
        shmem_u32[0] = d;
        copyFromShr2GlbMem<T, Q>(0, m, u, shmem_u32);
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

void sequential_set(uint32_t* u, uint32_t d, uint32_t m) {
    for (int i=1; i<m; i++) { u[i] = 0; }
    u[0] = d;
}

int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }


    uint32_t m = 1000;
    int size = m * sizeof(uint32_t);
    uint32_t* u = (uint32_t*)malloc(size);
    uint32_t* v = (uint32_t*)malloc(size);
    uint32_t* v_D;
    cudaMalloc(&v_D, size);

    for (int j = 0; j < 100; j++) {
        int randInt = (rand() % 110) - 10;

        randomInit<uint32_t>(u, m);
        cudaMemcpy(v_D, u, size, cudaMemcpyHostToDevice);

        sequential_set(u, randInt, m);

        int threadsPerBlock = 256;
        CallSet<uint32_t, 4><<<1, threadsPerBlock, size>>>(v_D, randInt, m);
        cudaDeviceSynchronize();

        gpuAssert( cudaPeekAtLastError() );
        cudaMemcpy(v, v_D, size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < m; i++) {
            if (v[i] != u[i]) {
                printf("ERROR AT ITERATION: %d\n", j);
                printSlice(v, 'v', i, m);
                printSlice(u, 'u', i, m);

                printf("INVALID AT INDEX %d: [%d/%d]\n", i, v[i], u[i]);

                free(v);
                cudaFree(v_D);
                return 1;
            }
        }
    }

    free(v);
    cudaFree(v_D);
    printf("set: VALID\n");
    return 0;
}








