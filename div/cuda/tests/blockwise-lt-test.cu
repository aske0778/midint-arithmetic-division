#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-div-helper.cu.h"

template<class T, class T2, uint32_t Q>
__global__ void Call_lt (
                T *u,
                T *v,
                const uint32_t m,
                int64_t* retval) {
    extern __shared__ char sh_mem[];
    volatile T* shmem_u = (T*)sh_mem;
    volatile T* shmem_v = (T*)(sh_mem + m*sizeof(T));
    volatile T2* shmem_buf = (T2*)(sh_mem + 2*m*sizeof(T));

    copyFromGlb2ShrMem<T, Q>(0, m, 0, u, shmem_u);
    copyFromGlb2ShrMem<T, Q>(0, m, 0, v, shmem_v);
    blockwise_lt<T, T2, Q>(shmem_u, shmem_v, m, retval, shmem_buf);

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

/**
 * @brief Returns a < b for two bigint_ts
 */
bool sequential_lt(uint32_t *a, uint32_t *b, uint32_t m)
{
    for (int i = m - 1; i >= 0; i--)
    {
        if (a[i] < b[i])
        {
            return 1;
        }
        else if (a[i] > b[i])
        {
            return 0;
        }
    }
    return 0;
}


int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }
    uint32_t m = 10;
    int size = m * sizeof(uint32_t);
    uint32_t* u = (uint32_t*)malloc(size);
    uint32_t* v = (uint32_t*)malloc(size);
    uint32_t* u_D;
    uint32_t* v_D;
    cudaMalloc(&u_D, size);
    cudaMalloc(&v_D, size);
    
    int64_t* cuda_retval;
    int64_t retval;
    
    cudaMalloc(&cuda_retval, sizeof(int64_t));


    srand(time(NULL));
    for (int j = 0; j < 10; j++) {

        randomInit<uint32_t>(u, m);
        randomInit<uint32_t>(v, m);
        cudaMemcpy(u_D, u, size, cudaMemcpyHostToDevice);
        cudaMemcpy(v_D, v, size, cudaMemcpyHostToDevice);


        int threadsPerBlock = 256;
        Call_lt<uint32_t, int64_t, 12><<<1, threadsPerBlock, 8*size>>>(u_D, v_D, m, cuda_retval);
        cudaDeviceSynchronize();

        gpuAssert( cudaPeekAtLastError() );
        cudaMemcpy(&retval, cuda_retval, sizeof(int64_t), cudaMemcpyDeviceToHost);
        printf("was u < v ? : %d \n", (sequential_lt(u, v, m)));
        for (int i = 0; i < m; i++) {
                printf("u and v was the following \n");
                printSlice(v, 'v', i, m);
                printSlice(u, 'u', i, m);

            }
        }

    free(u);
    free(v);
    cudaFree(u_D);
    cudaFree(v_D);
    cudaFree(cuda_retval);
    printf("lt: VALID\n");
    return 0;
}

