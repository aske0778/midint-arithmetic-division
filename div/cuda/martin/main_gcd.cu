#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../ker-division.cu.h"
//#include "../ker-division-64.cu.h"
#include "../helpers/helper.h"

int main()
{
   // using Base = U32bits;
   // using Base = U64bits;
    using Base = U16bits;
   // using Base = U8bits;
    using uint_t = Base::uint_t;
    const uint32_t M = 16;
    const uint32_t Q = 4;

    uint_t u[16] = {18, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0};
    uint_t v[16] = {48, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0};


    uint_t res[M] = {0};

    uint_t *d_u, *d_v, *d_res;
    cudaMalloc((void **)&d_u, M * sizeof(uint_t));
    cudaMalloc((void **)&d_v, M * sizeof(uint_t));
    cudaMalloc((void **)&d_res, M * sizeof(uint_t));

    cudaMemcpy(d_u, u, M * sizeof(uint_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, M * sizeof(uint_t), cudaMemcpyHostToDevice);

    gcd<Base, M, Q><<<1, M/Q, 2 * M * sizeof(uint_t)>>>(d_u, d_v, d_res);
    cudaDeviceSynchronize();

    cudaMemcpy(res, d_res, M * sizeof(uint_t), cudaMemcpyDeviceToHost);

    uint_t res_gmp[M] = {0};
   // gmpGCD(u, v, quo_gmp, rem_gmp, M);

    gmpGCD<uint_t, M>(1, u, v, res_gmp);

    prnt<uint_t>("u", u, M);
    prnt<uint_t>("v", v, M);

    prnt<uint_t>("res", res, M);
    prnt<uint_t>("res_gmp", res_gmp, M);

    // for (int i = 0; i < M; i++)
    // {
    //     if (quo[i] != quo_gmp[i] || rem[i] != rem_gmp[i])
    //     {
    //         printf("INVALID \n");
    //         break;
    //     }
    // }
    return 0;
}