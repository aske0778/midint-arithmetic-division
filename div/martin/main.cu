#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helper.h"

int main()
{
    const uint32_t M = 96;
    const uint32_t Q = 4;

    uint32_t u[96] = {
    724803052, 756165936, 1836347364, 1132624577, 412559579, 362594798, 351040828, 0,
    1046741222, 337739299, 1896306640, 1343606042, 1111783898, 446340713, 1197352298,
    915256190, 1782280524, 846942590, 524688209, 700108581, 1566288819, 1371499336,
    2114937732, 726371155, 1927495994, 292218004, 882160379, 11614769, 1682085273,
    1662981776, 630668850, 246247255, 1858721860, 1548348142, 105575579, 964445884,
    2118421993, 1520223205, 452867621, 1017679567, 1857962504, 201690613, 213801961,
    822262754, 648031326, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    uint32_t v[96] = {
    1829237727, 310022529, 1091414751, 0, 0, 0, 0, 0,  
    1411154259, 1737518944, 282828202, 110613202, 114723506, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    uint32_t quo[M] = {0};
    uint32_t rem[M] = {0};

    uint32_t *d_u, *d_v, *d_quo, *d_rem;
    cudaMalloc((void **)&d_u, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_v, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_quo, M * sizeof(uint32_t));
    cudaMalloc((void **)&d_rem, M * sizeof(uint32_t));

    cudaMemcpy(d_u, u, M * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, M * sizeof(uint32_t), cudaMemcpyHostToDevice);

    divShinv<M, Q><<<1, M/Q>>>(d_u, d_v, d_quo, d_rem);
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