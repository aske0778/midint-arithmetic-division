#include "helpers/types.cu.h"
#include "helpers/scan_reduce.cu.h"
#include "helpers/ker-helpers.cu.h"
#include "binops/add.cu.h"
#include "binops/sub.cu.h"
#include "binops/mult.cu.h"

/**
 * Calculates (a * b) rem B^d
 */
template<typename Base, uint32_t M, uint32_t Q>
__device__ inline void
multMod( volatile typename Base::uint_t* USh
       , volatile typename Base::uint_t* VSh
       , typename Base::uint_t UReg[Q]
       , typename Base::uint_t VReg[Q]
       , int d
       , typename Base::uint_t RReg[Q]
) {
    bmulRegsQ<U32bits, 1, Q/2>(USh, VSh, UReg, VReg, RReg, M); 

    #pragma unroll
    for (int i=0; i < Q; i++) {
        if (Q * threadIdx.x + i >= d)
        {
            RReg[i] = 0;
        }
    }
}

/**
 * Calculates B^h-v*w
 */
template<typename Base, uint32_t M, uint32_t Q>
__device__ inline bool
powDiff( volatile typename Base::uint_t* USh
       , volatile typename Base::uint_t* VSh
       , typename Base::uint_t VReg[Q]
       , typename Base::uint_t RReg[Q]
       , int h
       , int l
) {
    using uint_t = typename Base::uint_t;

    int vPrec = prec<uint_t, Q>(VReg, USh);
    int rPrec = prec<uint_t, Q>(RReg, VSh);
    int L = vPrec + rPrec - l + 1;
    bool sign = 1;
    
    if (vPrec == 0 || rPrec == 0) {
        zeroAndSet<uint_t, Q>(VReg, 1, h);
    }
    else if (L >= h) {
        __syncthreads();
        bmulRegsQ<Base, 1, Q/2>(USh, VSh, VReg, RReg, VReg, M);
        sub<Base, Q>(h, VReg, USh);
    }
    else {
        __syncthreads();
        multMod<Base, M, Q>(USh, VSh, VReg, RReg, L, VReg);
        if (!ez<uint_t, Q>(VReg, USh)) {
            if (ez<uint_t, Q>(VReg, L-1, VSh)) {
                sign = 0;
            }
            else {
                sub<Base, Q>(L, VReg, VSh);
            }
        }
    }
    return sign;
}

/**
 * Iterative towards an approximation in at most log(M) steps
 */
template<typename Base, uint32_t M, uint32_t Q>
__device__ inline void
step( volatile typename Base::uint_t* USh
    , volatile typename Base::uint_t* VSh
    , int h
    , typename Base::uint_t VReg[Q]
    , typename Base::uint_t RReg[Q]
    , int n
    , int l
) {
    using uint_t = typename Base::uint_t;
    using carry_t = typename Base::carry_t;

    bool sign = powDiff<Base, M, Q>(USh, VSh, VReg, RReg, h - n, l - 2);
    __syncthreads();
    bmulRegsQ<U32bits, 1, Q/2>(USh, VSh, RReg, VReg, VReg, M); 
    shift<uint_t, M, Q>(2 * n - h, VReg, VSh, VReg);
    shift<uint_t, M, Q>(n, RReg, USh, RReg);
    __syncthreads();

    if (sign) {
        baddRegs<uint_t, uint_t, carry_t, Q, UINT32_MAX>(VSh, RReg, VReg, RReg, M);
    }
    else {
        bsubRegs<uint_t, uint_t, carry_t, Q, UINT32_MAX>(VSh, RReg, VReg, RReg, M);
    }
}

/**
 * Refine the approximation of the quotient
 */
template<typename Base, uint32_t M, uint32_t Q>
__device__ inline void
refine( volatile typename Base::uint_t* USh
      , volatile typename Base::uint_t* VSh
      , typename Base::uint_t VReg[Q]
      , typename Base::uint_t TReg[Q]
      , int h
      , int k
      , int l
      , typename Base::uint_t RReg[Q]
) {
    using uint_t = typename Base::uint_t;

    shift<uint_t, M, Q>(2, RReg, USh, RReg);

    #pragma unroll
    for (int i = 0; i < log2f(h-k); i++) {
        int n = min(h - k + 1 - l, l);
        int s = max(0, k - 2 * l + 1 - 2);
        shift<uint_t, M, Q>(-s, VReg, VSh, TReg);
        __syncthreads();
        step<Base, M, Q>(USh, VSh, k + l + n - s + 2, TReg, RReg, n, l);
        __syncthreads();
        shift<uint_t, M, Q>(-1, RReg, USh, RReg);
        l = l + n - 1;
    }
    shift<uint_t, M, Q>(-2, RReg, VSh, RReg);
}

/**
 * Calculates the shifted inverse
 */
template<typename Base, uint32_t M, uint32_t Q>
__device__ inline void
shinv( volatile typename Base::uint_t* USh
     , volatile typename Base::uint_t* VSh
     , typename Base::uint_t VReg[Q]
     , typename Base::uint_t TReg[Q]
     , int h
     , typename Base::uint_t RReg[Q]
) {
    using uint_t = typename Base::uint_t;
    using uquad_t = typename Base::uquad_t;
    
    int k = prec<uint_t, Q>(VReg, USh) - 1;

    if (k == 0) {
        quo<Base, Q>(h, VSh[0], RReg);
        return;
    }
    __syncthreads();
    if (k >= h && !eq<uint_t, Q>(VReg, h, USh)) {
        return;
    }
    if (k == h-1 && VSh[k] > Base::HIGHEST / 2 ) {
        set<uint_t, Q>(RReg, 1, 0);
        return;
    }
    __syncthreads();
    if (eq<uint_t, Q>(VReg, k, USh)) {
        set<uint_t, Q>(RReg, 1, h - k);
        return;
    }
    int l = min(k, 2);    
    {
        if (threadIdx.x < (Q+3) / Q) {
            uquad_t V = 0;
            #pragma unroll
            for (int i = 0; i <= l; i++)
            {
                V += ((uquad_t)VSh[k - l + i]) << (Base::bits * i);
            }
        
            uquad_t b2l = (uquad_t)1 << Base::bits * 2 * l;
            uquad_t tmp = (b2l - V) / V + 1;

            #pragma unroll
            for (int i = 0; i < Q; i++) {
                int x = Q * threadIdx.x + i;
                if (x < 4) {
                    RReg[i] = (uint_t)(tmp >> Base::bits*x);
                }
            }
        }
    }
    __syncthreads();

    if (h - k <= l) {
        shift<uint_t, M, Q>(h-k-l, RReg, USh, RReg);
    }
    else {
        refine<Base, M, Q>(USh, VSh, VReg, TReg, h, k, l, RReg);
    }
}

/**
 * Implementation of multi-precision integer division using
 * the shifted inverse and classical multiplication
 */
template<typename Base, uint32_t M, uint32_t Q>
__global__ void 
divShinv( typename Base::uint_t* u
        , typename Base::uint_t* v
        , typename Base::uint_t* quo
        , typename Base::uint_t* rem
) {
    using uint_t = typename Base::uint_t;
    using carry_t = typename Base::carry_t;

    extern __shared__ char sh_mem[];
    volatile uint_t* VSh = (uint_t*)sh_mem;
    volatile uint_t* USh = (uint_t*)(VSh + M);
    uint_t VReg[Q];
    uint_t UReg[Q];
    uint_t RReg1[2*Q] = {0};
    uint_t* RReg2 = &RReg1[Q];

    cpyGlb2Sh2Reg<uint_t, M, Q>(v, VSh, VReg);
    cpyGlb2Sh2Reg<uint_t, M, Q>(u, USh, UReg);
    __syncthreads();

    int h = prec<uint_t, Q>(UReg, USh);
    __syncthreads();

    shinv<Base, M, Q>(USh, VSh, VReg, RReg2, h, RReg1);
    __syncthreads();

    bmulRegsQComplete<Base, 1, Q/2>(USh, VSh, UReg, RReg1, RReg1, M);
    __syncthreads();
    shiftDouble<uint_t, M, Q>(-h, RReg1, VSh, RReg1);
    __syncthreads();

    bmulRegsQ<Base, 1, Q/2>(USh, VSh, VReg, RReg1, RReg2, M); 
    __syncthreads();
    bsubRegs<uint_t, uint_t, carry_t, Q, Base::HIGHEST>(VSh, UReg, RReg2, RReg2, M);

    if (!lt<uint_t, Q>(RReg2, VReg, USh)) {
        __syncthreads();
        add1<Base, Q>(RReg1, USh);
        bsubRegs<uint_t, uint_t, carry_t, Q, Base::HIGHEST>(VSh, RReg2, VReg, RReg2, M);
    }

   __syncthreads();
    cpyReg2Sh2Glb<uint_t, M, Q>(quo, VSh, RReg1);
    cpyReg2Sh2Glb<uint_t, M, Q>(rem, USh, RReg2);
}

/**
 * Implementation of multi-precision integer quotient using the shifted inverse 
 * the shifted inverse and classical multiplication
 */
template<typename Base, uint32_t M, uint32_t Q>
__global__ void
quoShinv( typename Base::uint_t* u
        , typename Base::uint_t* v
        , typename Base::uint_t* quo
) {
    using uint_t = typename Base::uint_t;
    using carry_t = typename Base::carry_t;

    extern __shared__ char sh_mem[];
    volatile uint_t* VSh = (uint_t*)sh_mem;
    volatile uint_t* USh = (uint_t*)(VSh + M);
    uint_t VReg[Q];
    uint_t UReg[Q];
    uint_t RReg1[2*Q] = {0};
    uint_t* RReg2 = &RReg1[Q];

    cpGlb2Reg<uint_t, 1, M, Q>(1, VSh, v, VReg);
    cpGlb2Reg<uint_t, 1, M, Q>(1, USh, u, UReg);
    __syncthreads();

    int h = prec<uint_t, Q>(UReg, USh);
    __syncthreads();

    shinv<Base, M, Q>(USh, VSh, VReg, RReg2, h, RReg1);
    __syncthreads();

    bmulRegsQComplete<Base, 1, Q/2>(USh, VSh, UReg, RReg1, RReg1, M);
    __syncthreads();

    shiftDouble<uint_t, M*2, Q>(-h, RReg1, VSh, RReg1);
    __syncthreads();

    bmulRegsQ<Base, 1, Q/2>(USh, VSh, UReg, RReg1, RReg1, M);
    __syncthreads();
    
    bsubRegs<uint_t, uint_t, carry_t, Q, Base::HIGHEST>(VSh, UReg, RReg2, RReg2, M);
    
    if (!lt<uint_t, Q>(RReg2, VReg, USh)) { 
        __syncthreads();
        add1<Base, Q>(RReg1, VSh); 
    }

    __syncthreads();
    cpReg2Glb<uint_t, 1, M, Q>(1, VSh, RReg1, quo);
}


