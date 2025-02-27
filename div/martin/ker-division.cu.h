#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

#include "ker-helpers.cu.h"


template <class T, uint32_t Q>
__device__ inline void
shinv(uint32_t* VReg[Q], uint32_t h, vuint32_t* RReg[Q], const uint32_t m) {

    int k = prec4Reg()
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

    int l = min(k, 2);    
    {
        __uint128_t V = 0;
        for (int i = 0; i <= l; i++)
        {
            V += ((__uint128_t)v[k - l + i]) << (32 * i);
        }

        __uint128_t b2l = (__uint128_t)1 << 32 * 2 * l;
        __uint128_t tmp = (b2l - V) / V + 1;

        RReg[0] = (digit_t)(tmp);
        RReg[1] = (digit_t)(tmp >> 32);
        RReg[2] = (digit_t)(tmp >> 64);
        RReg[3] = (digit_t)(tmp >> 96);
    }




}

template <class T, uint32_t Q>
__global__ void div_shinv(uint32_t* u, uint32_t* v, uint32_t* res, const uint32_t m)
{
    extern __shared__ char sh_mem[];
    volatile T* VSh = (T*)sh_mem;
   // volatile T* TmpSh = (T*)(VSh + m);
    uint32_t VReg[Q];
    cpyGlb2Sh2Reg<T, Q>(v, VSh, VReg, m);


    uint16_t h = 
    
    prec<T,Q>(USh, &h, m);
    __syncthreads();

    uint_t Rrg[Q];
    shinv<T,Q>(VReg, h, RReg, m);



    // __shared__ uint32_t k;
    // prec<T,Q>(VSh, &k, m);
    // __syncthreads();

    // if (threadIdx.x == 0) {
    //     printf("Value of h: %u\n", h);
    // }



    cpySh2Glb<T, Q>(, res, 1);
}

#endif // KERNEL_DIVISION