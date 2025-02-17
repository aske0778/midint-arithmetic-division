#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>
#include "../ker-division.cu.h"
#include "../../sequential/helper.h"
#include "../../cuda/helper.h"
#include "../../sequential/div.h"


__global__ void CallShift(
    const n,
    const digit* u,
    digit_t* r,
    const prec m) {
        BlockwiseShift(n, u, r, m);
    }



int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }

    prec_t m = 10;

    bigint_t u = init(m);
    bigint_t v = init(m);
    bigint_t v_D;

    set(u, 1, m);
    set(v, 1, m);

    cudaMalloc($v_D, m * sizeof(digit_t));
    cudaMemcpy(v_D, v, m * sizeof(digit_t), cudaMemcpyHostToDevice);

    shift(2, u, u, m);
    shift(4, u, u, m);


    int threadsPerBlock = 256;
    CallShift<<<1, threadsPerBlock>>>(6, v_D, v_D, m);
    cudaMemcpy(v, v_D, m * sizeof(digit_t), cudaMemcpyDeviceToHost);
    gpuAssert( cudaPeekAtLastError() );

    for (int i = 0; i < m; i++) {
        if (v[i] != u[i]) {
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, v[i], u[i]);

            free(u);
            free(v);
            return 1;
        }
    }
    free(u);
    cudaFree(v);
    return 0;
}








