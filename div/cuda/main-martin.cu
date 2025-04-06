#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helpers/helper.h"

int main()
{
    const uint32_t M = 32;
    const uint32_t Q = 8;

    uint32_t u[32] = {1681692777, 1714636915, 1957747793, 424238335, 719885386, 1649760492, 596516649, 1189641421, 1025202362, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t v[32] = {1350490027, 783368690, 1102520059, 2044897763, 1967513926, 1365180540, 1540383426, 304089172, 1303455736, 35005211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    uint32_t quo[M] = {0};
    uint32_t rem[M] = {0};

    uint32_t *d_u, *d_v, *d_quo, *d_rem;
    cudaMalloc((void **)&d_u, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_v, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_quo, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_rem, M * sizeof(uint32_t));

    cudaMemcpy(d_u, u, M * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, M * sizeof(uint32_t), cudaMemcpyHostToDevice);

    divShinv<M, Q><<<1, M/Q, 2 * M * sizeof(uint32_t)>>>(d_u, d_v, d_quo, d_rem);
    cudaDeviceSynchronize();

    cudaMemcpy(quo, d_quo, M * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(rem, d_rem, M * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t quo_gmp[M] = {0};
    uint32_t rem_gmp[M] = {0};
    div_gmp(u, v, quo_gmp, rem_gmp, M);

    // printf("GMP:\n");
    // prnt("quo", quo_gmp, M);
    // prnt("rem", rem_gmp, M);
    // printf("Cuda:\n");
    // prnt("quo", quo, M);
    // prnt("rem", rem, M);
    for (int i = 0; i < M; i++)
    {
        if (quo[i] != quo_gmp[i] || rem[i] != rem_gmp[i])
        {
            printf("INVALID \n");
            break;
        }
    }
    return 0;
}