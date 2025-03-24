#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helper.h"

int main()
{
    const uint32_t M = 1024;
    const uint32_t Q = 8;

    uint32_t uPrec = (M);
    uint32_t* u = randBigInt(uPrec, M);
    uint32_t* v = randBigInt(uPrec * 0.9, M);
    uint32_t* quo = (uint32_t*)calloc(M, sizeof(uint32_t));
    uint32_t* rem = (uint32_t*)calloc(M, sizeof(uint32_t));

    uint32_t *d_u, *d_v, *d_quo, *d_rem;
    cudaMalloc((void **)&d_u, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_v, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_quo, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_rem, M * sizeof(uint32_t));

    cudaMemcpy(d_u, u, M * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, M * sizeof(uint32_t), cudaMemcpyHostToDevice);

    divShinv<M, Q><<<1, M/Q>>>(d_u, d_v, d_quo, d_rem, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(quo, d_quo, M * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(rem, d_rem, M * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t quo_gmp[M] = {0};
    uint32_t rem_gmp[M] = {0};
    div_gmp(u, v, quo_gmp, rem_gmp, M);

    printf("GMP:\n");
    prnt("quo", quo_gmp, M);
    prnt("rem", rem_gmp, M);
    printf("Cuda:\n");
    prnt("quo", quo, M);
    prnt("rem", rem, M);

    return 0;
}