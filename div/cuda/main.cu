#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helpers/helper.h"

int main()
{
    const uint32_t M = 4096;
    const uint32_t Q = 16;
    const uint32_t num_instances = 1;
    // const uint32_t total_work = M * num_instances;
    const uint32_t size = M * sizeof(uint32_t);

    uint32_t uPrec = M / 2;
    uint32_t* u = randBigInt(uPrec, M, num_instances);
    uint32_t* v = randBigInt(uPrec - 8, M, num_instances);
    uint32_t quo[M] = {0};
    uint32_t rem[M] = {0};

    uint32_t *d_u, *d_v, *d_quo, *d_rem;
    cudaMalloc((void **)&d_u, size);
    cudaMalloc((void **)&d_v, size);
    cudaMalloc((void **)&d_quo, size);
    cudaMalloc((void **)&d_rem, size);

    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);

    divShinv<M, Q><<<1, M/Q, 2 * size>>>(d_u, d_v, d_quo, d_rem, num_instances);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    cudaMemcpy(quo, d_quo, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(rem, d_rem, size, cudaMemcpyDeviceToHost);

    uint32_t quo_gmp[M] = {0};
    uint32_t rem_gmp[M] = {0};
    div_gmp(u, v, quo_gmp, rem_gmp, M);

    for (int i = 0; i < M; i++) {
        if (quo[i] != quo_gmp[i] || rem[i] != rem_gmp[i]) {
            printf("\nInputs:\n");
            printSlice(u, 'u', i, M);
            printSlice(v, 'v', i, M);
            printf("Output:\n");
            printSlice(quo, 'q', i, M);
            printSlice(rem, 'r', i, M);
            printf("GMP:\n");
            printSlice(quo_gmp, 'q', i, M);
            printSlice(rem_gmp, 'r', i, M);
            return 1;
        }
    }
    printf("Done\n");
    return 0;
}