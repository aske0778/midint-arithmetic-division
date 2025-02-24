#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

// #include "../../cuda/ker-fft-help.cu.h"
// #include "../../cuda/ker-helpers.cu.h"
// #include "../../cuda/helper.h"

template <class T, uint32_t Q>
__device__ inline void cpyGlb2Sh(uint_t* AGlb, uint_t* BGlb, uint_t* ASh, uint_t* BSh, const uint32_t m) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx >= m) return;

        ASh[idx] = AGlb[idx];
        BSh[idx] = BGlb[idx];
    }
}

template <class T, uint32_t Q>
__device__ inline void cpySh2Glb(uint_t* ASh, uint_t* AGlb, const uint32_t m) {
    #pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
        uint32_t idx = blockDim.x * i + threadIdx.x;
        if (idx >= m) return;

        AGlb[idx] = ASh[idx];
    }
}

template <class T, uint32_t Q>
__device__ inline void prec(const volatile T* u, volatile int* p, const uint32_t m) {
    int highest_idx = -1;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && u[idx] != 0) {
            highest_idx = max(highest_idx, idx);
        }
    }
    atomicMax(p, highest_idx + 1);
}



template <class T, uint32_t Q>
__device__ inline void shift(const int n, const volatile T* u, volatile T* res, const uint32_t m) {
    #pragma unroll
    for (int i = 0; i < Q; i++)
    {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx >= m) return;   

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

template <class T, uint32_t Q>
__global__ void div_shinv(const uint32_t* u, const uint32_t* v, uint32_t* res, const uint32_t m)
{
    extern __shared__ char sh_mem[];
    volatile T* USh = (T*)sh_mem;
    volatile T* VSh = (T*)(USh + m * 2);
    volatile T* TmpSh = (T*)(VSh + m * 2);
    cpyGlb2Sh<T, Q>(u, v, USh, VSh, m);

    __shared__ int32_t h;
    prec<T,Q>(USh, h, m);
    __syncthreads();

    __shared__ int32_t k;
    prec<T,Q>(VSh, k, m);
    __syncthreads();



    cpySh2Glb<T, Q>(Tmp, res, m);
}

#endif // KERNEL_DIVISION