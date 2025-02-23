#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-division.cu.h"
// #include "../sequential/helper.h"
// #include "../sequential/div.h"
#include "../../cuda/helper.h"

template<class T, uint32_t Q>
__global__ void CallShift(
    const int n,
    T* u,
    T* r,
    const uint32_t m) {
        extern __shared__ char sh_mem[];
        volatile T* shmem_u = (T*)sh_mem;
        volatile T* shmem_v = (T*)(sh_mem + m*sizeof(T));

        copyFromGlb2ShrMem<T, Q>(0, m, 0, u, shmem_u);
        shift<T, Q>(n, shmem_u, shmem_v, m);
        copyFromShr2GlbMem<T, Q>(0, m, r, shmem_v);
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

void sequential_shift(int n, uint32_t* u, uint32_t* r, uint32_t m)
{
    if (n >= 0)
    { // Right shift
        for (int i = m - 1; i >= 0; i--)
        {
            int offset = i - n;
            r[i] = (offset >= 0) ? u[offset] : 0;
        }
    }
    else
    { // Left shift
        for (int i = 0; i < m; i++)
        {
            int offset = i - n;
            r[i] = (offset < m) ? u[offset] : 0;
        }
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
        // int randInt = (rand() % m) - (m/2);
        int randInt = -500;

        randomInit<uint32_t>(u, m);
        cudaMemcpy(v_D, u, size, cudaMemcpyHostToDevice);

        sequential_shift(randInt, u, u, m);

        int threadsPerBlock = 256;

        CallShift<uint32_t, 2><<<1, threadsPerBlock, 4*size>>>(randInt, v_D, v_D, m);
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
        printf("%d\n", randInt);
    }

    free(v);
    cudaFree(v_D);
    printf("shift: VALID\n");
    return 0;
}








