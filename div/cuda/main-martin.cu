#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helpers/helper.h"

int main()
{
    using Base = U32bits;
  //  using Base = U64bits;
   // using Base = U16bits;
    using uint_t = Base::uint_t;
    const uint32_t M = 32;
    const uint32_t Q = 4;

    uint_t u[32] = {0, 805251743, 1198720172, 1805613091, 144874089, 1510906527, 473903566, 13798878, 94255812, 1564003050, 99885196, 2081362124, 636453333, 363304213, 79065186, 1360478499, 604263370, 775056794, 1588695568, 1155465115, 535286141, 1389079342, 442982639, 1582482437, 4744263, 1642663198, 1153263590, 844169939, 1033206202, 0, 0, 0};
    uint_t v[32] = {0, 181226513, 286791631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


    // uint_t u[16] = {0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // uint_t v[16] = {0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // uint_t u[16] = {37826, 28157, 12125, 41481, 25946, 5930, 13477, 2530, 9635, 36859, 16311, 28179, 0, 0, 0, 0};
    // uint_t v[16] = {32071, 19796, 21146, 45873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    uint_t quo[M] = {0};
    uint_t rem[M] = {0};

    uint_t *d_u, *d_v, *d_quo, *d_rem;
    cudaMalloc((void **)&d_u, M * sizeof(uint_t));
    cudaMalloc((void **)&d_v, M * sizeof(uint_t));
    cudaMalloc((void **)&d_quo, M * sizeof(uint_t));
    cudaMalloc((void **)&d_rem, M * sizeof(uint_t));

    cudaMemcpy(d_u, u, M * sizeof(uint_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, M * sizeof(uint_t), cudaMemcpyHostToDevice);

    divShinv<Base, M, Q><<<1, M/Q, 2 * M * sizeof(uint_t)>>>(d_u, d_v, d_quo, d_rem);
    cudaDeviceSynchronize();

    cudaMemcpy(quo, d_quo, M * sizeof(uint_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(rem, d_rem, M * sizeof(uint_t), cudaMemcpyDeviceToHost);

    uint_t quo_gmp[M] = {0};
    uint_t rem_gmp[M] = {0};
    div_gmp(u, v, quo_gmp, rem_gmp, M);

    // prnt<uint_t>("u", u, M);
    // prnt<uint_t>("v", v, M);

    // prnt<uint_t>("quo", quo, M);
    // prnt<uint_t>("rem", rem, M);
    // prnt<uint_t>("quo_gmp", quo_gmp, M);
    // prnt<uint_t>("rem_gmp", rem_gmp, M);

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