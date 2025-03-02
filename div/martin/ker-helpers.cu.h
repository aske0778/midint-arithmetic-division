template <uint32_t Q>
__device__ inline void cpyGlb2Sh2Reg(const uint32_t* AGlb, volatile uint32_t* ASh, uint32_t AReg[Q], const uint32_t m) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m) {
            ASh[idx] = AGlb[idx];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < m) {
            AReg[i] = ASh[idx];
        }
    }
}

template <uint32_t Q>
__device__ inline uint32_t prec(uint32_t u[Q], uint32_t* sh_mem, const uint32_t m) {

    uint32_t highest_idx = 0;
    sh_mem[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < m && u[i] != 0) {
            highest_idx = idx;
        }
    }
    atomicMax(sh_mem, highest_idx);
    __syncthreads();
    return sh_mem[0] + 1;
}

// template <uint32_t Q>
// __device__ inline bool lt(uint32_t u[Q], uint32_t bpow, uint32_t* sh_mem, const uint32_t m) {
//     return true;
// }

// template <uint32_t Q>
// __device__ inline bool lt(uint32_t bpow, uint32_t u[Q], uint32_t* sh_mem, const uint32_t m) {
//     return false;
// }

template <uint32_t Q>
void quo(uint32_t bpow, uint32_t v, bigint_t q, prec_t m)
{
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