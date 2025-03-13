
template <class T, uint32_t Q>
__device__ inline void cpySh2Glb(volatile uint32_t* volatile ASh, uint32_t* AGlb, const uint32_t m) {
    #pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
        uint32_t idx = blockDim.x * i + threadIdx.x;
        if (idx < m) {
            AGlb[idx] = ASh[idx];
        }
    }
}



template <class T, uint32_t Q>
__device__ inline void ltBpow(volatile T* u, int k, bool* shmem, const uint32_t m) {
    bool tmp = sh_mem[0];
    shmem[0] = true;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && idx >= k && u[idx] != 0) {
            shmem[0] = false;
        }
    }
    __syncthreads();

    res = shmem[0];
    shmem[0] = tmp;
    return res
}


template <class T, uint32_t Q>
__device__ inline void prec(volatile T* u, uint32_t *res, const uint32_t m) {
    int highest_idx = -1;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && u[idx] != 0) {
            highest_idx = idx;
        }
    }
    atomicMax(res, highest_idx + 1);
}

template <class T, uint32_t M, uint32_t Q>
__device__ inline T
prec4Reg( T u[Q]
        , T* p
) {

    int highest_idx = -1;
    int old = p[0];
    p[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M && u[idx] != 0) {
            highest_idx = max(highest_idx, idx);
        }
    }
    atomicMax(p, highest_idx + 1);
    highest_idx = p[0];
    p[0] = old;
    return highest_idx;
}



template <class T, uint32_t Q>
__device__ inline void shift(const int n, const volatile T* u, volatile T* res, const uint32_t m) {
    #pragma unroll
    for (int i = 0; i < Q; i++)
    {
        int idx = i * blockDim.x + threadIdx.x;  
        if (idx < m) {
            int offset = idx - n;
            if (n >= 0)
            { // Right shift
                res[idx] = (offset >= 0) ? u[offset] : 0;
            }
            else
            { // Left shift
                res[idx] = (offset < m) ? u[offset] : 0;
            }
        }
    }
}


template <uint32_t M, uint32_t Q>
__device__ inline void cpyGlb2Sh2Reg(uint32_t* AGlb, uint32_t* ASh, uint32_t AReg[Q]) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M) {
            ASh[idx] = AGlb[idx];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M) {
            AReg[i] = ASh[idx];
        }
    }
}

template <uint32_t M, uint32_t Q>
__device__ inline void cpyReg2Sh2Glb(uint32_t AReg[Q], uint32_t* ASh, uint32_t* AGlb) {
    #pragma unroll
    for (int i=0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M) {
            ASh[idx] = AReg[i];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M) {
            AGlb[idx] = ASh[idx];
        }
    }
}

template <uint32_t M, uint32_t Q>
__device__ inline uint32_t prec(uint32_t u[Q], volatile uint32_t* sh_mem) {
    uint32_t highest_idx = 0;
    sh_mem[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M && u[i] != 0) {
            highest_idx = idx;
        }
    }
 //   atomicMax(sh_mem, highest_idx);
    if (threadIdx.x == 0) {
        printf("Value of h:");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        printf("Value of h:");
    }
    
    return sh_mem[0] + 1;
}

// template <uint32_t Q>
// __device__ inline bool lt(uint32_t u[Q], uint32_t bpow, uint32_t* sh_mem, uint32_t m) {
//     return true;
// }

// template <uint32_t Q>
// __device__ inline bool lt(uint32_t bpow, uint32_t u[Q], uint32_t* sh_mem, uint32_t m) {
//     return false;
// }

// template <uint32_t Q>
// void quo(uint32_t bpow, uint32_t v, bigint_t q, prec_t m)
// {
//     uint64_t r = 0;
//     for (int i = m - 1; i >= 0; i--)
//     {
//         r = (r << 32) + n[i];
//         if (r >= d)
//         {
//             q[i] = r / d;
//             r = r % d;
//         }
//     }
// }



// template <uint32_t Q>
// __device__ inline bool lt(uint32_t u[Q], uint32_t bpow, uint32_t* sh_mem) {
//     uint32_t highest_idx = 0;
//     sh_mem[0] = 0;
    
//     #pragma unroll
//     for (int i = 0; i < Q; i++) {
//         if (u[i] != 0) {
//             highest_idx = Q * threadIdx.x + i;
//         }
//     }
//     atomicMax(sh_mem, highest_idx);
//     __syncthreads();    
//     return sh_mem[0] + 1;
// }

// template <uint32_t Q>
// __device__ inline bool lt(uint32_t bpow, uint32_t u[Q], uint32_t* sh_mem, uint32_t m) {
//     return false;
// }


template<uint32_t Q>
__device__ inline uint8_t lessThanWarp(uint8_t u, uint32_t idx, uint32_t lane) {
    #pragma unroll
    for (int i = 2; i <= WARP; i *= 2) {
        if ((lane & (i-1)) == (i-1)) {
            uint8_t elm = __shfl_down_sync(0xFFFFFFFF, u, i - 1);
            u |= (u & 0b10) && (elm & 0b01);
            u = (u & ~0b10) | (u & elm & 0b10);
        }
    }
    return u;
}



