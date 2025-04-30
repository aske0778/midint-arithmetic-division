#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../ker-division.cu.h"
//#include "../ker-division-64.cu.h"
#include "../helpers/helper.h"

int main()
{
  //  using Base = U32bits;
    using Base = U64bits;
   // using Base = U16bits;
   // using Base = U8bits;
    using uint_t = Base::uint_t;
    const uint32_t M = 16;
    const uint32_t Q = 4;

    // uint_t u[16] = {35165, 45317, 41751, 43096, 23273, 33886, 43220, 48555, 36018, 53453, 57542, 0, 0, 0, 0, 0};
    // uint_t v[16] = {30363, 40628, 9300, 34321, 50190, 7554, 63604, 34369, 0, 0, 0, 0, 0, 0, 0, 0};


    // uint_t u[8] = {4, 2, 2, 2, 0, 0, 0, 0};
    // uint_t v[8] = {4, 1, 1, 1, 0, 0, 0, 0};

    // uint_t u[16] = {39017, 18547, 56401, 23807, 37962, 22764, 7977, 31949, 0, 0, 0, 0, 0, 0, 0, 0};
    // uint_t v[16] = {22714, 55211, 16882, 7931, 43491, 57670, 124, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // uint_t u[16] = {7, 3, 5, 10, 7, 9, 7, 9, 7, 2, 2, 10, 0, 0, 0, 0};
    // uint_t v[16] = {4, 4, 5, 1, 7, 2, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0};

    // uint_t u[16] = {4, 2, 3, 6, 9, 6, 10, 10, 10, 9, 9, 9, 0, 0, 0, 0};
    // uint_t v[16] = {10, 1, 1, 5, 10, 2, 4, 4, 1, 1, 1, 0, 0, 0, 0, 0};


    uint_t u[16] = {1681692777, 1714636915, 1957747793, 424238335, 719885386, 1649760492, 596516649, 1189641421, 0, 0, 0, 0, 0, 0, 0, 0};
    uint_t v[16] = {1025202362, 1350490027, 783368690, 1102520059, 2044897763, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    


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