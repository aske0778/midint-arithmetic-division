#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../ker-division.cu.h"
#include "../helpers/helper.h"

int main()
{
    //using Base = U32bits;
  //  using Base = U64bits;
   // using Base = U16bits;
    using Base = U8bits;
    using uint_t = Base::uint_t;
    const uint32_t M = 16;
    const uint32_t Q = 4;

    uint_t u[16] = {4, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint_t v[16] = {4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


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

    prnt<uint_t>("u", u, M);
    prnt<uint_t>("v", v, M);

    prnt<uint_t>("quo", quo, M);
    prnt<uint_t>("rem", rem, M);
    prnt<uint_t>("quo_gmp", quo_gmp, M);
    prnt<uint_t>("rem_gmp", rem_gmp, M);

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