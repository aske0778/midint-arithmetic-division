#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-division.cu.h"
#include "../../sequential/helper.h"


template<class T, uint32_t Q>
__global__ void CallPrec(
        T* u,
        T* buf,
        const uint32_t m) {
    extern __shared__ char sh_mem[];
    // volatile T* shmem_u = (T*)sh_mem;
    // volatile T* shmem_buf = (T*)(sh_mem + m*sizeof(T));
    // volatile T* shmem_buf = (T*)sh_mem;
    T* shmem_buf = (T*)sh_mem;
    volatile T* shmem_u = (T*)(sh_mem + sizeof(T));

    copyFromGlb2ShrMem<T, Q>(0, m, 0, u, shmem_u);
    prec<T, Q>(shmem_u, shmem_buf, m);
    copyFromShr2GlbMem<T, Q>(0, sizeof(T), buf, shmem_buf);
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

uint32_t sequential_prec(uint32_t* u, uint32_t m)
{
    uint32_t acc = 0;
    for (int i = 0; i < m; i++)
    {
        if (u[i] != 0)
        {
            acc = i;
        }
    }
    return acc + 1;
}



int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }


    uint32_t m = 100;
    int size = m * sizeof(uint32_t);
    uint32_t* u = (uint32_t*)malloc(size);
    uint32_t* v = (uint32_t*)malloc(1 * sizeof(uint32_t));
    uint32_t* v_D;
    uint32_t* prec;
    cudaMalloc(&v_D, size);
    cudaMalloc(&prec, 1 * sizeof(uint32_t));

    for (int j = 0; j < 100; j++) {
        randomInit<uint32_t>(u, m);
        cudaMemcpy(v_D, u, size, cudaMemcpyHostToDevice);

        u[0] = sequential_prec(u, m);

        int threadsPerBlock = 32;
        CallPrec<uint32_t, 4><<<1, threadsPerBlock, size + sizeof(uint32_t)>>>(v_D, prec, m);
        cudaDeviceSynchronize();

        gpuAssert( cudaPeekAtLastError() );
        cudaMemcpy(v, prec, 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // v[0] = m;

        for (int i = 0; i < 1; i++) {
            if (v[i] != u[i]) {
                printf("ERROR AT ITERATION: %d\n", j);
                printf("INVALID: [%u/%u]\n", v[i], u[i]);

                // printf("%u\n", u + size);
                // printf("%u\n", v);

                // printSlice(u, 'u', i, m);
                // printSlice(v, 'v', i, m);
                // printSlice(u, 'v', v[0], m);

                free(v);
                cudaFree(v_D);
                return 1;
            }
        }
    }

    free(v);
    cudaFree(v_D);
    printf("multd: VALID\n");
    return 0;
}








