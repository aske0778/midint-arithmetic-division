#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helpers/helper.h"

int main()
{
    using Base = U32bits;
    // using Base = U16bits;
    using uint_t = Base::uint_t;

    const uint32_t Q = 32;
    const uint32_t M = 8192;
    const uint32_t num_instances = 1;
    // const uint32_t total_work = M * num_instances;
    const uint32_t size = M * sizeof(uint_t);

    uint32_t uPrec = M;
    uint_t* u = randBigInt<uint_t>(uPrec, M, num_instances);
    uint_t* v = randBigInt<uint_t>(uPrec - 8, M, num_instances);
    uint_t quo[M] = {0};
    uint_t rem[M] = {0};

    uint_t *d_u, *d_v, *d_quo, *d_rem;
    cudaMalloc((void **)&d_u, size);
    cudaMalloc((void **)&d_v, size);
    cudaMalloc((void **)&d_quo, size);
    cudaMalloc((void **)&d_rem, size);

    cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);

    cudaFuncSetAttribute(divShinv<Base, M,Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98000);

    divShinv<Base, M, Q><<<1, M/Q, 2 * size>>>(d_u, d_v, d_quo, d_rem);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    cudaMemcpy(quo, d_quo, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(rem, d_rem, size, cudaMemcpyDeviceToHost);

    uint_t quo_gmp[M] = {0};
    uint_t rem_gmp[M] = {0};
    div_gmp<uint_t>(u, v, quo_gmp, rem_gmp, M);

    for (int i = 0; i < M; i++) {
        if (quo[i] != quo_gmp[i] || rem[i] != rem_gmp[i]) {
            printf("\nInputs:\n");
            printSlice<uint_t>(u, 'u', i, M);
            printSlice<uint_t>(v, 'v', i, M);
            printf("Output:\n");
            printSlice<uint_t>(quo, 'q', i, M);
            printSlice<uint_t>(rem, 'r', i, M);
            printf("GMP:\n");
            printSlice<uint_t>(quo_gmp, 'q', i, M);
            printSlice<uint_t>(rem_gmp, 'r', i, M);
            return 1;
        }
    }
    printf("Done\n");
    return 0;
}