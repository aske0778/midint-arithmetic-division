#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-division.cu.h"
#include "../../cuda/helper.h"


template<class T, class T2, uint32_t Q>
__global__ void CallMultD(
        T* u,
        const uint32_t d,
        T* v,
        const uint32_t m) {
    extern __shared__ char sh_mem[];
    volatile T* shmem_u = (T*)sh_mem;
    volatile T* shmem_v = (T*)(sh_mem + m*sizeof(T));
    volatile T2* shmem_buf = (T2*)(sh_mem + 2*m*sizeof(T));

    copyFromGlb2ShrMem<T, Q>(0, m, 0, u, shmem_u);
    multd<T, T2, Q>(shmem_u, d, shmem_v, shmem_buf, m);
    copyFromShr2GlbMem<T, Q>(0, m, v, shmem_v);
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

void sequential_multd(uint32_t* a, uint32_t b, uint32_t* r, uint32_t m)
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


    uint32_t m = 1000;
    int size = m * sizeof(uint32_t);
    uint32_t* u = (uint32_t*)malloc(size);
    uint32_t* v = (uint32_t*)malloc(size);
    uint32_t* v_D;
    cudaMalloc(&v_D, size);

    for (int j = 0; j < 100; j++) {
        srand(time(NULL));
        int randInt = (rand() % 110) - 10;

        randomInit<uint32_t>(u, m);
        cudaMemcpy(v_D, u, size, cudaMemcpyHostToDevice);

        sequential_multd(u, randInt, u, m);

        int threadsPerBlock = 256;
        CallMultD<uint32_t, uint64_t, 8><<<1, threadsPerBlock, 8*size>>>(v_D, randInt, v_D, m);
        cudaDeviceSynchronize();

        gpuAssert( cudaPeekAtLastError() );
        cudaMemcpy(v, v_D, size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < m; i++) {
            if (v[i] != u[i]) {
                printf("%d\n", randInt);
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








