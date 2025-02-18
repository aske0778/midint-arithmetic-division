#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

#include "../cuda/helper.h"
// #include "../cuda/ker-fft-help.cu.h"
// #include "../cuda/ker-helpers.cu.h"


template<uint32_t Q>
__device__ inline void
BlockwiseShift( const int n,
           const uint32_t* u,
           uint32_t* res,
           const uint32_t m ) {

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        
        if (idx >= m) {
           continue; 
        }

        int offset = idx - n;
        if (n >= 0) {   // Right shift
            res[idx] = (offset >= 0) ? u[offset] : 0;
        } else {        // Left shift
            res[idx] = (offset < m) ? u[offset] : 0;
        }
    }
}



template<int Q>
__device__ inline void
BlockwiseMultD( uint32_t* a,
                uint32_t  b,
                uint32_t* v,
                const uint32_t m ) {
    uint64_t buf[m];

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx] = ((uint64_t)a[idx]) * (uint64_t)b;
    }

    #pragma unroll
    for (int i = 1; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx + 1] += buf[idx ] >> 32;
    }

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        v[idx] = (uint32_t)buf[idx];
    }
}






#endif // KERNEL_DIVISION