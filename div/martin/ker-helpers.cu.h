template <class T, uint32_t Q>
__device__ inline void cpyGlb2Sh(const uint32_t* AGlb, volatile uint32_t* ASh, const uint32_t m) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m) {
            ASh[idx] = AGlb[idx];
        }
    }
}

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