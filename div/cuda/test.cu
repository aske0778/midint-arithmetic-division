#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helper.h"

void printSlice(uint32_t* u, char name, int i, uint32_t m) {
    int min = i-3 < 0 ? 0 : i-3;
    int max = i+3 > m ? m : i+3;

    printf("%c[%u-%u]: [", name, min, max);
    for (int i = min; i < max; i++) {
        printf("%u, ", u[i]);
    }
    printf("]\n");
}

int main() {
  //  srand(time(NULL));
    bool stop = false;
    const uint32_t num_instances = 4;
    const uint32_t M = 32;
    const uint32_t Q = 4;
    const uint32_t total_work = M * num_instances;

    for (int i = 0; i < 100 && !stop; i++) {
        printf("\rIteration: %u", i);
        uint32_t uPrec = (total_work);
        uint32_t vPrec = (uPrec);
        uint32_t* u = randBigInt(uPrec, total_work);
        uint32_t* v = randBigInt(vPrec, total_work);
        uint32_t* quo = (uint32_t*)calloc(total_work, sizeof(uint32_t));
        uint32_t* rem = (uint32_t*)calloc(total_work, sizeof(uint32_t));

        uint32_t *d_u, *d_v, *d_quo, *d_rem;
        cudaMalloc((void **)&d_u, total_work * sizeof(uint32_t));
        cudaMalloc((void **)&d_v, total_work * sizeof(uint32_t));
        cudaMalloc((void **)&d_quo, total_work * sizeof(uint32_t));
        cudaMalloc((void **)&d_rem, total_work * sizeof(uint32_t));

        cudaMemcpy(d_u, u, total_work * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v, total_work * sizeof(uint32_t), cudaMemcpyHostToDevice);

        for (int i = 0; i < num_instances; i++) {
            printf("block %d has: \n [", i);
            for (int j = 0; j < M; j++) {
                printf("%u, ", v[j + (i * M)]);
            }
            printf("] \n");
        }


        divShinv<M, Q><<<num_instances, M/Q, 2 * M * sizeof(uint32_t)>>>(d_u, d_v, d_quo, d_rem, num_instances);
        cudaDeviceSynchronize();

        cudaMemcpy(quo, d_quo, M * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(rem, d_rem, M * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint32_t* quo_gmp = (uint32_t*)calloc(M, sizeof(uint32_t));
        uint32_t* rem_gmp = (uint32_t*)calloc(M, sizeof(uint32_t));
        div_gmp(u, v, quo_gmp, rem_gmp, M);
        
        for (int i = 0; i < M; i++) {
            if (quo[i] != quo_gmp[i] || rem[i] != rem_gmp[i]) {
                stop = true;
                printf("\nInputs:\n");
                // prnt("  u", u, M);
                // prnt("  v", v, M);
                printSlice(u, 'u', i, M);
                printSlice(v, 'v', i, M);
                printf("Output:\n");
                printSlice(quo, 'q', i, M);
                printSlice(rem, 'r', i, M);
                // prnt("  q", quo, M);
                // prnt("  r", rem, M);
                printf("GMP:\n");
                printSlice(quo_gmp, 'q', i, M);
                printSlice(rem_gmp, 'r', i, M);
                // prnt("  q", quo_gmp, M);
                // prnt("  r", rem_gmp, M);
                break;
            }
        }
        free(u); free(v); free(quo); free(rem); free(quo_gmp); free(rem_gmp);
    }
    return 0;
}