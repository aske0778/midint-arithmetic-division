#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

// #include "../../cuda/helper.h"
#include "ker-div-helper.cu.h"
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



template <class Base, uint32_t M, uint32_t Q>
__global__ inline void
divShinvClassical( typename Base::uint_t* ass
                 , typename Base::uint_t* bss
                 , typename Base::uint_t* rss
) {

    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;

    const uint32_t M_lft = LIFT_LEN(M, Q);
    const uint32_t shmem_len = IPB*M_lft;

    __shared__ uint_t Ash[shmem_len];
    __shared__ uint_t Bsh[shmem_len];
    volatile carry_t* carry_shm = (volatile carry_t*)Ash;

    uint_t Arg[Q];
    uint_t Brg[Q];

    { // read from global memory
        copyFromGlb2Shr2RegMem<uint_t, M, Q>(0, 0, Ass, Ash, Arg);
        copyFromGlb2Shr2RegMem<uint_t, M, Q>(0, 0, Bss, Bsh, Brg);
        __syncthreads();
    }





    { // write to global memory
        copyFromReg2GlbMem<uint_t, M, Q>(Ass, Ash, Arg);
    }

}




#endif // KERNEL_DIVISION