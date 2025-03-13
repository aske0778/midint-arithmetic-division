#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"

void prnt(const char *str, uint32_t *u, uint32_t m)
{
    printf("%s: [", str);
    for (int i = 0; i < m; i++)
    {
        printf("%u", u[i]);
        if (i < m - 1)
            printf(", ");
    }
    printf("]\n");
}


int main()
{
    uint32_t u[] = {1234, 5678, 91011, 53, 0, 0, 0, 0};
    uint32_t v[] = {1234, 5678, 91011, 54, 0, 0, 0, 0};
    uint32_t res[] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t m = 8;

    uint32_t *d_u, *d_v, *d_quo, *d_rem;
    cudaMalloc((void **)&d_u, m * sizeof(uint32_t));
    cudaMalloc((void **)&d_v, m * sizeof(uint32_t));
    cudaMalloc((void **)&d_quo, m * sizeof(uint32_t));
    cudaMalloc((void **)&d_rem, m * sizeof(uint32_t));

    cudaMemcpy(d_u, u, m * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, m * sizeof(uint32_t), cudaMemcpyHostToDevice);

    divShinv<m, 2><<<1, 4>>>(d_u, d_v, d_quo, d_rem);
    cudaDeviceSynchronize();

    cudaMemcpy(res, d_quo, m * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    prnt("n", res, 8);
    return 0;
}