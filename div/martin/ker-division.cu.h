#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

#include "ker-helpers.cu.h"


template <uint32_t Q>
__device__ inline void
shinv(uint32_t* VSh, uint32_t* USh, uint32_t VReg[Q], uint32_t h, uint32_t RReg[Q], const uint32_t m) {

    uint16_t k = prec<Q>(VReg, USh, m) - 1;

    if (k == 0) {
      quo<Q>(h, v, RReg, m);
      return;
    }
    if (lt<Q>(v, h, USh, m)) {

    }
    if (2v > Bh) {

    }
    if (v = Bk) {

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

        #pragma unroll
        for (uint32_t i = 0; i < Q; i++) {
            int x = Q * threadIdx.x + i;
            if (x < 4) {
                RReg[i] = (digit_t)(tmp >> x);
            }
        }
    }

    if (h - k <= l) {
       // shift(h - k - l, w, w, m); use cuda implementation
    }
    else {
        // call refine
    }



}

template <uint32_t Q>
__global__ void div_shinv(uint32_t* u, uint32_t* v, uint32_t* res, const uint32_t m)
{
    extern __shared__ char sh_mem[];
    uint32_t* VSh = (uint32_t*)sh_mem;  //volatile?
    uint32_t* USh = (uint32_t*)(VSh + m); //volatile?
    uint32_t VReg[Q];
    uint32_t UReg[Q];

    cpyGlb2Sh2Reg<Q>(v, VSh, VReg, m);
    cpyGlb2Sh2Reg<Q>(u, USh, UReg, m);

    uint16_t h = prec<Q>(UReg, USh, m);


    uint_t RReg[Q];
    shinv<Q>(VSh, USh, VReg, h, RReg, m);

  
    // if (threadIdx.x == 0) {
    //     printf("Value of h: %u\n", h);
    // }
}

#endif // KERNEL_DIVISION