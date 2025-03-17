#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helper.h"


int main() {
  //  srand(time(NULL));
    bool stop = false;
    const uint32_t M = 1024;
    const uint32_t Q = 4;

    for (int i = 0; i < 100 && !stop; i++) {
        printf("Iteration: %u \n", i);
        uint32_t uPrec = (rand() % M/2) + 1;
        uint32_t vPrec = (rand() % uPrec) + 3;
        uint32_t* u = randBigInt(uPrec, M);
        uint32_t* v = randBigInt(vPrec, M);
        uint32_t* quo = (uint32_t*)calloc(M, sizeof(uint32_t));
        uint32_t* rem = (uint32_t*)calloc(M, sizeof(uint32_t));

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

        uint32_t* quo_gmp = (uint32_t*)calloc(M, sizeof(uint32_t));
        uint32_t* rem_gmp = (uint32_t*)calloc(M, sizeof(uint32_t));
        div_gmp(u, v, quo_gmp, rem_gmp, M);
        
        for (int i = 0; i < M; i++) {
            if (quo[i] != quo_gmp[i] || rem[i] != rem_gmp[i]) {
                stop = true;
                printf("Inputs:\n");
                prnt("  u", u, M);
                prnt("  v", v, M);
                printf("Output:\n");
                prnt("  q", quo, M);
                prnt("  r", rem, M);
                printf("GMP:\n");
                prnt("  q", quo_gmp, M);
                prnt("  r", rem_gmp, M);
                break;
            }
        }
        free(u); free(v); free(quo); free(rem); free(quo_gmp); free(rem_gmp);
    }
    return 0;
}