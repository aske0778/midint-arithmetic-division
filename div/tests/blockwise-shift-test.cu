#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-division.cu.h"
// #include "../sequential/helper.h"
// #include "../sequential/div.h"
#include "../../cuda/helper.h"

__global__ void CallShift(
    const int n,
    const uint32_t* u,
    uint32_t* r,
    const uint32_t m) {
        BlockwiseShift<8>(n, u, r, m);
    }

void printSlice(uint32_t* u, char name, int i, uint32_t m) {
    int min = i-3 < 0 ? 0 : i-3;
    int max = i+3 > m ? m : i+3;

    printf("%c[%d-%d]: [", name, min, max);
    for (int i = min; i < max; i++) {
        printf("%d, ", u[i]);
    }
    printf("]\n");
}

void shift(int n, uint32_t* u, uint32_t* r, uint32_t m)
{
    if (n >= 0)
    { // Right shift
        for (int i = m - 1; i >= 0; i--)
        {
            int offset = i - n;
            r[i] = (offset >= 0) ? u[offset] : 0;
        }
    }
    else
    { // Left shift
        for (int i = 0; i < m; i++)
        {
            int offset = i - n;
            r[i] = (offset < m) ? u[offset] : 0;
        }
    }
}



int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }


    uint32_t m = 100;
    int size = m * sizeof(uint32_t);

    // uint32_t u[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t* u = (uint32_t*)malloc(size);
    uint32_t* v = (uint32_t*)malloc(size);
    uint32_t* v_D;
    cudaMalloc(&v_D, size);

    for (int j = 0; j < 10; j++) {
        srand(time(NULL));
        int shiftBy = (rand() % 110) - 10;

        randomInit<uint32_t>(u, m);
        cudaMemcpy(v_D, u, size, cudaMemcpyHostToDevice);

        shift(shiftBy, u, u, m);

        int threadsPerBlock = 256;
        CallShift<<<1, threadsPerBlock>>>(shiftBy, v_D, v_D, m);
        cudaDeviceSynchronize();

        gpuAssert( cudaPeekAtLastError() );
        cudaMemcpy(v, v_D, size, cudaMemcpyDeviceToHost);

        printf("%d\n", shiftBy);
        for (int i = 0; i < m; i++) {
            if (v[i] != u[i]) {
                printf("ERROR AT ITERATION: %d\n", j);
                printSlice(u, 'u', i, m);
                printSlice(v, 'v', i, m);

                printf("INVALID AT INDEX %d: [%d/%d]\n", i, v[i], u[i]);

                // free(u);
                free(v);
                cudaFree(v_D);
                return 1;
            }
        }
    }

    // free(u);
    free(v);
    cudaFree(v_D);
    printf("shift: VALID\n");
    return 0;
}








