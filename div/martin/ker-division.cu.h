#include "ker-helpers.cu.h"
#include "../../cuda/helper.h"
#include "../../cuda/ker-classic-mul.cu.h"

template<uint32_t M, uint32_t Q>
__device__ inline void
multMod(uint32_t* USh, uint32_t* VSh, uint32_t UReg[Q], uint32_t VReg[Q], uint32_t d, uint32_t RReg[Q]) {
    bmulRegsQ<U32bits, 1, M, Q/2>(USh, VSh, UReg, VReg, RReg); 
    #pragma unroll
    for (int i=0; i < Q; i++) {
        if (Q * threadIdx.x + i >= d)
        {
            RReg[i] = 0;
        }
    }
}

template<uint32_t M, uint32_t Q>
__device__ inline bool
powDiff(uint32_t* USh, uint32_t* VSh, uint32_t VReg[Q], uint32_t RReg[Q], uint32_t h, uint32_t l) {
    uint16_t vPrec = prec<Q>(VReg, USh);
    uint16_t rPrec = prec<Q>(RReg, VSh);
    uint32_t L = vPrec + rPrec - l + 1;
    bool sign = 1;
    if (vPrec == 0 || rPrec == 0) {
        zeroAndSet<Q>(VReg, 1, h);
    }
    else if (L >= h) {
        bmulRegsQ<U32bits, 1, M, Q/2>(USh, VSh, VReg, RReg, VReg); 
        sub<Q>(h, VReg, USh);
    }
    else {
        multMod<M, Q>(USh, VSh, VReg, RReg, L, VReg);
        if (!ez<Q>(VReg, USh)) {
            if (ez<Q>(VReg, L-1, VSh)) {
                sign = 0;
            }
            else {
                sub<Q>(L, VReg, USh);
            }
        }
    }
    return sign;
}

template<uint32_t M, uint32_t Q>
__device__ inline void
step(uint32_t* USh, uint32_t* VSh, uint32_t h, uint32_t VReg[Q], uint32_t RReg[Q], uint32_t n, uint32_t l, uint32_t g) {
    bool sign = powDiff<M, Q>(USh, VSh, VReg, RReg, h - n, l - g);
    bmulRegsQ<U32bits, 1, M, Q/2>(USh, VSh, RReg, VReg, VReg); 
    shift<M, Q>(2 * n - h, VReg, VSh, VReg);
    shift<M, Q>(n, RReg, USh, RReg);

    if (sign) {
        baddRegs<uint32_t, uint32_t, uint32_t, M, Q, UINT32_MAX>(VSh, RReg, VReg, RReg);
    }
    else {
        bsubRegs<uint32_t, uint32_t, uint32_t, M, Q, UINT32_MAX>(VSh, RReg, VReg, RReg);
    }
}

template<uint32_t M, uint32_t Q>
__device__ inline void
refine(uint32_t* USh, uint32_t* VSh, uint32_t VReg[Q], uint32_t TReg[Q], uint16_t h, uint16_t k, uint8_t l, uint32_t RReg[Q]) {
    shift<M, Q>(2, RReg, USh, RReg);
    while (h - k > l) {
        uint32_t n = min(h - k + 1 - l, l);
        uint32_t s = max(0, k - 2 * l + 1 - 2);
        shift<M, Q>(-s, VReg, USh, TReg);
        step<M, Q>(USh, VSh, k + l + n - s + 2, TReg, RReg, n, l, 2);
        shift<M, Q>(-1, RReg, USh, RReg);
        //printRegs<M,Q>("res", RReg, USh);
        l = l + n - 1;
    }
    shift<M, Q>(-2, RReg, USh, RReg);
}

template<uint32_t M, uint32_t Q>
__device__ inline void
shinv(uint32_t* USh, uint32_t* VSh, uint32_t VReg[Q], uint32_t TReg[Q], uint16_t h, uint32_t RReg[Q]) {

    uint16_t k = prec<Q>(VReg, USh) - 1;

    if (k == 0) {
        quo<Q>(h, VSh[0], RReg);
        return;
    }
    if (k >= h && !eq<Q>(VReg, h, USh)) {
        return;
    }
    if (k == h-1 && VSh[k] > UINT32_MAX / 2 ) {
        set<Q>(RReg, 1, 0);
        return;
    }
    if (eq<Q>(VReg, k, USh)) {
        set<Q>(RReg, 1, h - k);
        return;
    }
    
    uint8_t l = min(k, 2);    
    {
        if (threadIdx.x < (Q+3) / Q) {
            __uint128_t V = 0;
            for (int i = 0; i <= l; i++)
            {
                V += ((__uint128_t)VSh[k - l + i]) << (32 * i);
            }

            __uint128_t b2l = (__uint128_t)1 << 32 * 2 * l;
            __uint128_t tmp = (b2l - V) / V + 1;

            #pragma unroll
            for (int i = 0; i < Q; i++) {
                int x = Q * threadIdx.x + i;
                if (x < 4) {
                    RReg[i] = (uint32_t)(tmp >> 32*x);
                }
            }
        }
    }

    if (h - k <= l) {
        shift<M, Q>(h-k-l, RReg, USh, RReg);
    }
    else {
        refine<M, Q>(USh, VSh, VReg, TReg, h, k, l, RReg);
    }
}

template<uint32_t M, uint32_t Q>
__global__ void divShinv(uint32_t* u, uint32_t* v, uint32_t* quo, uint32_t* rem) {
    extern __shared__ char sh_mem[];
    uint32_t* VSh = (uint32_t*)sh_mem;
    uint32_t* USh = (uint32_t*)(VSh + M);
    uint32_t VReg[Q];
    uint32_t UReg[Q];
    uint32_t RReg1[Q] = {0};
    uint32_t RReg2[Q] = {0};

    cpyGlb2Sh2Reg<Q>(v, VSh, VReg);
    cpyGlb2Sh2Reg<Q>(u, USh, UReg);
    __syncthreads();
    
    printRegs<M,Q>("res", VReg, USh);

    uint16_t h = prec<Q>(UReg, USh);

    shinv<M, Q>(USh, VSh, VReg, RReg2, h, RReg1);
    __syncthreads();
    bmulRegsQ<U32bits, 1, M, Q/2>(USh, VSh, UReg, RReg1, RReg1);
    
    shift<M, Q>(-h, RReg1, USh, RReg1);
    __syncthreads();

    bmulRegsQ<U32bits, 1, M, Q/2>(USh, VSh, VReg, RReg1, RReg2); 
    bsubRegs<uint32_t, uint32_t, uint32_t, M, Q, UINT32_MAX>(VSh, UReg, RReg2, RReg2);

    if (!lt<Q>(RReg2, VReg, USh)) {
        add1<Q>(RReg1, USh);
        bsubRegs<uint32_t, uint32_t, uint32_t, M, Q, UINT32_MAX>(VSh, RReg2, VReg, RReg2);
    }

    cpyReg2Sh2Glb<Q>(RReg1, VSh, quo);
    cpyReg2Sh2Glb<Q>(RReg2, USh, rem);
    __syncthreads();
}