#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../ker-division.cu.h"
//#include "../ker-division-64.cu.h"
#include "../helpers/helper.h"


int main() {

 //  using Base = U32bits;
    using Base = U64bits;
   // using Base = U16bits;
  //  using Base = U8bits;
    using uint_t = Base::uint_t;

   // srand(time(NULL));
    bool stop = false;
    const uint32_t Q = 4;
    const uint32_t M = 2048;

    for (int i = 0; i < 1 && !stop; i++) {
        printf("\rIteration: %u", i);
        uint32_t uPrec = min((rand() % M)+1, M-Q);
        uint32_t vPrec = (rand() % uPrec) + 1;
        uint_t* u = randBigInt<uint_t>(uPrec, M);
        uint_t* v = randBigInt<uint_t>(vPrec, M);
        // prnt<uint_t>("u", u, M);
        // prnt<uint_t>("v", v, M);
        uint_t* quo = (uint_t*)calloc(M, sizeof(uint_t));
        uint_t* rem = (uint_t*)calloc(M, sizeof(uint_t));

        uint_t *d_u, *d_v, *d_quo, *d_rem;
        cudaMalloc((void **)&d_u, M * sizeof(uint_t));
        cudaMalloc((void **)&d_v, M * sizeof(uint_t));
        cudaMalloc((void **)&d_quo, M * sizeof(uint_t));
        cudaMalloc((void **)&d_rem, M * sizeof(uint_t));

        cudaMemcpy(d_u, u, M * sizeof(uint_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v, M * sizeof(uint_t), cudaMemcpyHostToDevice);

        cudaFuncSetAttribute(divShinv<Base,M,Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

        divShinv<Base, M, Q><<<1, M/Q,  2 * M * sizeof(uint_t)>>>(d_u, d_v, d_quo, d_rem);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        cudaMemcpy(quo, d_quo, M * sizeof(uint_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(rem, d_rem, M * sizeof(uint_t), cudaMemcpyDeviceToHost);

        uint_t* quo_gmp = (uint_t*)calloc(M, sizeof(uint_t));
        uint_t* rem_gmp = (uint_t*)calloc(M, sizeof(uint_t));
        div_gmp<uint_t>(u, v, quo_gmp, rem_gmp, M);

        for (int i = 0; i < M; i++) {
            if (quo[i] != quo_gmp[i] || rem[i] != rem_gmp[i]) {
                stop = true;
                // printf("Inputs:\n");
                // prnt("  u", u, M);
                // prnt("  v", v, M);
                // printf("Output:\n");
                // prnt("  q", quo, M);
                // prnt("  r", rem, M);
                // printf("GMP:\n");
                // prnt("  q", quo_gmp, M);
                // prnt("  r", rem_gmp, M);
                // printf("Iteration: %u \n", i);
                printf("INVALID \n");
                break;
            }
        }
        free(u); free(v); free(quo); free(rem); free(quo_gmp); free(rem_gmp);
    }
    return 0;
}