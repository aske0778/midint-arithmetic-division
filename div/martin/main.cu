#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helpers/helper.h"

int main()
{
    const uint32_t M = 128;
    const uint32_t Q = 4;

    uint32_t u[128] = {1681692777, 1714636915, 1957747793, 424238335, 719885386, 1649760492, 596516649, 1189641421, 1025202362, 1350490027, 783368690, 1102520059, 2044897763, 1967513926, 1365180540, 1540383426, 304089172, 1303455736, 35005211, 521595368, 294702567, 1726956429, 336465782, 861021530, 278722862, 233665123, 2145174067, 468703135, 1101513929, 1801979802, 1315634022, 635723058, 1369133069, 1125898167, 1059961393, 2089018456, 628175011, 1656478042, 1131176229, 1653377373, 859484421, 1914544919, 608413784, 756898537, 1734575198, 1973594324, 149798315, 2038664370, 1129566413, 184803526, 412776091, 1424268980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t v[128] = {1911759956, 749241873, 137806862, 42999170, 982906996, 135497281, 511702305, 2084420925, 1937477084, 1827336327, 572660336, 1159126505, 805750846, 1632621729, 1100661313, 1433925857, 1141616124, 84353895, 939819582, 2001100545, 1998898814, 1548233367, 610515434, 1585990364, 1374344043, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
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