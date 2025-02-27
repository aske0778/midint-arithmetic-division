#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

// #include "../../cuda/helper.h"
// #include "../../cuda/ker-helpers.cu.h"
#include "ker-div-helper.cu.h"
#include "ker-bpow.cu.h"
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
__device__ inline typename Base::uint_t
multmod4Reg( typename Base::uint_t Arg[Q]
       , typename Base::uint_t Brg[Q]
       , uint32_t d
       , volatile typename Base::uint_t* Ash
       , volatile typename Base::uint_t* Bsh
) {
    using uint_t = typename Base::uint_t;

    uint_t Rrg[Q]; 
    zero4Reg(Rrg);
    bmulRegsQ<Base, 1, M, Q/2>(Ash, Bsh, Arg, Brg, Rrg);
    for (int i = 0; i < Q; i++) {
        uint32_t idx = Q * threadIdx.x + i;
        if (i >= d) {
            Rrg[idx] = 0;
        }
    }
    return Rrg;
}




template <class T, uint32_t M, uint32_t Q>
__device__ inline void
shinv( T v[Q]
     , T w[Q]
     , uint32_t h
     , uint32_t k
     , volatile T* shmem
) {



    { // Early termination checks
        if (lt4Reg2Bpow<T, M, Q>(v, 0, shmem)) {
            // TODO: return quo and write result to w
            return;
        } else if (lt4Bpow2Reg<T, M, Q>(h, v, shmem)) {
            zero4Reg<T, M, Q>(w);
            return;
        } else if (lt4Bpow2RegMul2<T, M, Q>(h, v, shmem)) {
            set4Reg<T, M, Q>(w, 1);
            return;
        } else if (eq4Reg2Bpow<T, M, Q>(v, k, shmem)) {
            bpow4Reg<T, M, Q>(w, h-k);
            return;
        }
    }


}


template <class Base, uint32_t M, uint32_t Q>
__global__ void
divShinvClassical( typename Base::uint_t* ass
                 , typename Base::uint_t* bss
                 , typename Base::uint_t* rss
) {

    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;

    const uint32_t shmem_len = LIFT_LEN(M, Q);

    __shared__ uint_t Ash[shmem_len];
    __shared__ uint_t Bsh[shmem_len];
    volatile carry_t* carry_shm = (volatile carry_t*)Ash;

    uint_t Arg[Q];
    uint_t Brg[Q];

    { // read from global memory
        copyFromGlb2Shr2RegMem<uint_t, M, Q>(0, 0, ass, Ash, Arg);
        copyFromGlb2Shr2RegMem<uint_t, M, Q>(0, 0, bss, Bsh, Brg);
        __syncthreads();
    }




    { // write to global memory
        copyFromReg2Shr2GlbMem<uint_t, M, Q>(rss, Ash, Arg);
    }

}




#endif // KERNEL_DIVISION