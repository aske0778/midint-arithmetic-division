#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-division.cu.h"
#include "../ker-div-helper.cu.h"
#include "../../../cuda/helper.h"
#include "../../sequential/helper.h"

template<class T, class T2, uint32_t Q>
__global__ void Call_quo(
    T* n,
    uint32_t d,
    T* q,
    const uint32_t m) {
        extern __shared__ char sh_mem[];
        volatile T* shmem_u = (T*)sh_mem;
        volatile T* shmem_buf = (T*)(sh_mem + 2*m*sizeof(T));

        copyFromGlb2ShrMem<T, Q>(0, m, 0, n, shmem_u);
        Blockwise_quo<T, T2, Q>(shmem_u, d, q, m, shmem_buf);
        copyFromShr2GlbMem<T, Q>(0, m, q, shmem_buf);
    }

void printSlice(uint32_t* u, char name, int i, uint32_t m) {
    int min = i-3 < 0 ? 0 : i-3;
    int max = i+3 > m ? m : i+3;

    printf("%c[%u-%u]: [", name, min, max);
    for (int i = min; i < max; i++) {
        printf("%u, ", u[i]);
    }
    printf("]\n");
}

/*
 * @brief bigint_t division
 * @note Uses long division algorithm
 * https://en.wikipedia.org/wiki/Division_algorithm#Long_division
 *
 * @param n numerator
 * @param d denominator
 * @param q quotient
 * @param r remainder
 * @param m Total size of bigint_ts
 */
void quo(uint32_t* n, uint32_t d, uint32_t* q, uint32_t m)
{
    if (d == 0)
    {
        printf("Division by zero\n");
        return;
    }
    uint64_t r = 0;


    for (int i = m - 1; i >= 0; i--)
    {
        r = (r << 32) + n[i];
        if (r >= d)
        {
            q[i] = r / d;
            r = r % d;
        }
    }
}

/**
 * @brief Initializes a bigint_t with m digits
 *
 * @param m The number of digits
 * @return bigint_t The initialized bigint_t
 */
void init2(uint32_t* n, uint32_t m)
{
    for (int i = 0; i < m; i++)
    {
        n[i] = 0;
    }
}

void shift2(int n, uint32_t* u, uint32_t* r, uint32_t m)
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




/**
 * @brief Computes the base og bigint_t to the power of n
 *
 * @param n the power to raise the base to
 * @param m the total number of digits in the bigint_t
 */
void bpow2(uint32_t* number, int n, uint32_t m)
{
    init2(number, m);
    number[0] = 1;
    shift2(n, number, number, m);
}



void sequential_multd(uint32_t* a, uint32_t b, uint32_t* r, uint32_t m)
{
    uint64_t buf[m];

    for (int i = 0; i < m; i++) {
        buf[i] = ((uint64_t)a[i]) * (uint64_t)b;
    }

    for (int i = 0; i < m - 1; i++) {
        buf[i + 1] += buf[i] >> 32;
    }

    for (int i = 0; i < m; i++) {
        r[i] = (uint32_t)buf[i];
    }
}

uint32_t sequential_prec(uint32_t* u, uint32_t m)
{
    uint32_t acc = 0;
    for (int i = 0; i < m; i++)
    {
        if (u[i] != 0)
        {
            acc = i;
        }
    }
    return acc + 1;
}



int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }

    const uint32_t m = 10;
    int size = m * sizeof(uint32_t);

    uint32_t* u = (uint32_t*)malloc(size);
    uint32_t* v = (uint32_t*)malloc(size);
    uint32_t* B = (uint32_t*)malloc(size);
    uint32_t* Bh = (uint32_t*)malloc(size);
    uint32_t* v2 = (uint32_t*)malloc(size);
    uint32_t* w = (uint32_t*)malloc(size);
    uint32_t* w2 = (uint32_t*)malloc(size);
    uint32_t* v_D;
    uint32_t* Bh_D;
    uint32_t* Bh_D2;

    randomInit<uint32_t>(v, 1);

    int h = sequential_prec(u, m);
    bpow2(B ,1, m);


    bpow2(Bh, h, m);

    init2(v2, m);
    init2(w, m);
    sequential_multd(v, 2, v2, m);
    quo(Bh, v[0], w, m);



    //cudaMalloc(&v_D, size);

    //cudaMemcpy(v_D, v, size, cudaMemcpyHostToDevice);

    cudaMalloc(&Bh_D, size);

    cudaMemcpy(Bh_D, Bh, size, cudaMemcpyHostToDevice);

    cudaMalloc(&Bh_D2, size);

    //cudaMemcpy(Bh_D2, Bh, size, cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    Call_quo<uint32_t, uint64_t, 1><<<1, threadsPerBlock, 8*size>>>(Bh_D, v[0], Bh_D2, m);
    cudaDeviceSynchronize();

    gpuAssert( cudaPeekAtLastError() );
    cudaMemcpy(w2, Bh_D2, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        if (w[i] != w2[i]) {
            printf("INVALID AT INDEX %u: [%u/%u]\n", i, w[i], w2[i]);
            printSlice(w, 'w', i, m);
            printSlice(w2, 'w2', i, m);

            // free(u);
            free(v);
            cudaFree(v_D);
            return 1;
        }
    }
    // free(u);
    free(v);
    cudaFree(v_D);
    printf("multd: VALID\n");
    return 0;
}

