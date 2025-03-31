
template<uint32_t Q>
__device__ inline void printLhcs(const char *str, uint32_t lhcs[2][Q+2], volatile uint32_t* sh_mem, uint32_t M)
{
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[Q * threadIdx.x + i] = lhcs[0][i];
    }
    __syncthreads();
    
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[M/2 - 1 + (Q * threadIdx.x + i)] = lhcs[1][i];
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        printf("%s: [", str);
        for (int i = 0; i < M; i++)
        {
            printf("%u", sh_mem[i]);
            if (i < M - 1)
                printf(", ");
        }
        printf("]\n");
    }
    __syncthreads();
}

template<uint32_t Q>
__device__ inline void printRegs1(const char *str, uint32_t u[Q], uint32_t* sh_mem, uint32_t M)
{
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[Q * threadIdx.x + i] = u[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        printf("%s: [", str);
        for (int i = 0; i < M; i++)
        {
            printf("%u", sh_mem[i]);
            if (i < M - 1)
                printf(", ");
        }
        printf("]\n");
    }
    __syncthreads();
}
