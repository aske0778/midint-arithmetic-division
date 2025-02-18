#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

#include "../cuda/helper.h"
// #include "../cuda/ker-fft-help.cu.h"
// #include "../cuda/ker-helpers.cu.h"


/**
 * @brief Sets the first element to d and zeros the rest
 */
template<uint32_t Q>
__device__ inline void
set( uint32_t* u,
     const uint32_t d,
     const uint32_t m ) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m) {
            u[idx] = 0;
        }
    }
    if (threadIdx.x == 0) {
        u[0] = d;
    }
}



/**
 * @brief 
 * @todo Contains bug
 * 
 * @tparam Q 
 * @param n 
 * @param u 
 * @param res 
 * @param m 
 * @return __device__ 
 */
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


/**
 * @brief 
 * @todo Contains bug
 * 
 * @tparam Q 
 * @param u 
 * @param d 
 * @param v 
 * @param buf 
 * @param m 
 * @return __device__ 
 */
template<int Q>
__device__ inline void
BlockwiseMultD( uint32_t* u,
                uint32_t  d,
                uint32_t* v,
                volatile uint64_t* buf,
                const uint32_t m ) {
                    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx] = ((uint64_t)u[idx]) * (uint64_t)d;
    }
    __syncthreads();

    // #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx + 1] += (buf[idx] >> 32);
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        v[idx] = (uint32_t)buf[idx];
    }
    __syncthreads();
}






#endif // KERNEL_DIVISION