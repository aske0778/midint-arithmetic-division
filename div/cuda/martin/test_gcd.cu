#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "../ker-division.cu.h"
//#include "../ker-division-64.cu.h"
#include "../helpers/helper.h"


int main() {
   // cudaSetDevice(1);
   // using Base = U32bits;
    using Base = U64bits;
  //  using Base = U16bits;
   // using Base = U8bits;
    using uint_t = Base::uint_t;

   // srand(time(NULL));
    bool stop = false;
    const uint32_t Q = 4;
    const uint32_t M = 4096;

    for (int i = 0; i < 1000000 && !stop; i++) {
        printf("Iteration: %u \n", i);
        uint32_t uPrec = 1056; //min((rand() % M)+3, M/4);
        uint32_t vPrec = 1024; //(rand() % uPrec) + 2;
        uint_t* u = randBigInt<uint_t>(uPrec, M);
        uint_t* v = randBigInt<uint_t>(vPrec, M);
        // prnt<uint_t>("u", u, M);
        // prnt<uint_t>("v", v, M);
        uint_t* res = (uint_t*)calloc(M, sizeof(uint_t));

        uint_t *d_u, *d_v, *d_res;
        cudaMalloc((void **)&d_u, M * sizeof(uint_t));
        cudaMalloc((void **)&d_v, M * sizeof(uint_t));
        cudaMalloc((void **)&d_res, M * sizeof(uint_t));

        cudaMemcpy(d_u, u, M * sizeof(uint_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v, M * sizeof(uint_t), cudaMemcpyHostToDevice);

        cudaFuncSetAttribute(gcd<Base,M,Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

        gcd<Base, M, Q><<<1, M/Q,  2 * M * sizeof(uint_t)>>>(d_u, d_v, d_res);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel Launch Error: %s\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();

        cudaMemcpy(res, d_res, M * sizeof(uint_t), cudaMemcpyDeviceToHost);

        uint_t* res_gmp = (uint_t*)calloc(M, sizeof(uint_t));
        gmpGCD<uint_t, M>(1, u, v, res_gmp);

        for (int i = 0; i < M; i++) {
            if (res[i] != res_gmp[i]) {
                stop = true;
                printf("Inputs:\n");
                prnt("  u", u, M);
                prnt("  v", v, M);
                printf("Output:\n");
                prnt("res", res, M);
                printf("GMP:\n");
                prnt("res_gmp", res_gmp, M);
              //  prnt("  r", rem_gmp, M);
                // printf("Iteration: %u \n", i);
                printf("INVALID \n");
                break;
            }
        }
        free(u); free(v); free(res); free(res_gmp);
    }
    return 0;
}