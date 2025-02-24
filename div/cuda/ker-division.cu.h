#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

// #include "../../cuda/helper.h"
#include "ker-helper.cu.h"
// #include "../cuda/ker-fft-help.cu.h"
// #include "../cuda/ker-helpers.cu.h"


/**
 * @brief
 * @todo Contains bug when n < 0
 *
 * @tparam Q
 * @param n
 * @param u
 * @param res
 * @param m
 * @return __device__
 */
template <class T, uint32_t Q>
__device__ inline void
shift(const int n,
      volatile T *u,
      volatile T *res,
      const uint32_t m) {

    if (n >= 0) { // Left shift
        #pragma unroll
        for (int i = 0; i < Q; i++) {
            int idx = i * blockDim.x + threadIdx.x;
            if (idx < m) {
                int offset = idx - n;
                res[idx] = (offset < m) ? u[offset] : 0;
            }
            __syncthreads();
        }
    } else { // Right shift
        #pragma unroll
        for (int i = Q; i >= 0; i--) {
            int idx = i * blockDim.x + threadIdx.x;
            if (idx < m) {
                int offset = idx - n;
                res[idx] = (offset >= 0) ? u[offset] : 0;
            }
            __syncthreads();
        }
    }
    __syncthreads();
}

/**
 * @brief
 * @todo Contains bug when d < 0 where
 * sometimes the result is -1 less than expected
 *
 * @tparam Q
 * @param u
 * @param d
 * @param v
 * @param buf
 * @param m
 * @return __device__
 */
template <class T, class T2, uint32_t Q>
__device__ inline void
multd(volatile T *u,
      const uint32_t d,
      volatile T *v,
      volatile T2 *buf,
      const uint32_t m)
{

#pragma unroll
    for (int i = 0; i < Q; i++)
    {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx] = ((T2)u[idx]) * (T2)d;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Q; i++)
    {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx + 1] += (buf[idx] >> 32);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Q; i++)
    {
        int idx = i * blockDim.x + threadIdx.x;
        v[idx] = (T)buf[idx];
    }
    __syncthreads();
}

#endif // KERNEL_DIVISION