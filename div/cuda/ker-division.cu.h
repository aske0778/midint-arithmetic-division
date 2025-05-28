#include "helpers/types.cu.h"
#include "helpers/copy.cu.h"
#include "helpers/scan_reduce.cu.h"
#include "helpers/ker-helpers.cu.h"
#include "binops/add.cu.h"
#include "binops/sub.cu.h"
#include "binops/mult.cu.h"

#define BLOCKS_PER_SM 1

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
    if (d <= blockDim.x) {
        smallMult<Base, Q>(USh, VSh, UReg, VReg, RReg, d); 
    } else if (d <= 2*blockDim.x){
        smallMult2x<Base, Q>(USh, VSh, UReg, VReg, RReg, d); 
    } else {
        bmulRegsQ<Base, 1, Q/2>(USh, VSh, UReg, VReg, RReg, M);
        #pragma unroll
        for (int i=0; i < Q; i++) {
            if (Q * threadIdx.x + i >= d) {
                RReg[i] = 0;
            }
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

    int vPrec = prec<uint_t, Q>(VReg, (uint32_t*)USh);
    int rPrec = prec<uint_t, Q>(RReg, (uint32_t*)VSh);
    int L = vPrec + rPrec - l + 1;                       
    bool sign = 1;                              

    if (vPrec == 0 || rPrec == 0) {
        zeroAndSet<uint_t, Q>(VReg, 1, h);
    } else if (L >= h) {
        __syncthreads();
        int maxMul = vPrec + rPrec;
        if (maxMul <= blockDim.x) {
            smallMult<Base, Q>(USh, VSh, VReg, RReg, VReg, maxMul); 
        } else if (maxMul <= 2*blockDim.x) {
            smallMult2x<Base, Q>(USh, VSh, VReg, RReg, VReg, maxMul); 
        } else {
            bmulRegsQ<Base, 1, Q/2>(USh, VSh, VReg, RReg, VReg, M); 
        }
        __syncthreads();
        if (lt<uint_t, Q>(VReg, h, (uint32_t*)USh)) {  
            sub<Base, Q>(h, VReg, VSh);
        } else {
            sub<Base, Q>(VReg, h, VSh);
            sign = 0;
        }
    } else {
        __syncthreads();
        multMod<Base, M, Q>(USh, VSh, VReg, RReg, L, VReg);
        __syncthreads();

        if (!ez<uint_t, Q>(VReg, USh)) {
            if (ez<uint_t, Q>(VReg, L-1, VSh)) {
                sign = 0;
            } else {
                sub<Base, Q>(L, VReg, &VSh[4]);
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
    , int g
) {
    using uint_t  = typename Base::uint_t; 
    using ubig_t  = typename Base::ubig_t;
    using carry_t = typename Base::carry_t;

    bool sign = powDiff<Base, M, Q>(USh, VSh, VReg, RReg, h - n, l - g);
    __syncthreads();

    int rPrec = prec<uint_t, Q>(RReg, (uint32_t*)VSh);
    int vPrec = prec<uint_t, Q>(VReg, (uint32_t*)USh);
    __syncthreads();
    int maxMul = rPrec+vPrec;                            
    if (maxMul <= blockDim.x) {
        smallMult<Base, Q>(USh, VSh, RReg, VReg, VReg, maxMul); 
    } else if (maxMul <= 2*blockDim.x){
        smallMult2x<Base, Q>(USh, VSh, RReg, VReg, VReg, maxMul); 
    } else {
        bmulRegsQ<Base, 1, Q/2>(USh, VSh, RReg, VReg, VReg, M);
    }
    __syncthreads();

    shift<uint_t, M, Q>(n, RReg, USh, RReg);
    if (sign) {
        shift<uint_t, M, Q>(2 * n - h, VReg, VSh, VReg);
        __syncthreads();
        baddRegs<uint_t, uint_t, carry_t, Q, Base::HIGHEST>((carry_t*)VSh, RReg, VReg, RReg);
    } else {
        bool isZero = ezShift<uint_t, Q>(VReg, 2 * n - h, VSh);
        __syncthreads();
        shift<uint_t, M, Q>(2 * n - h, VReg, USh, VReg);
        if (!isZero) add1<Base, Q>(VReg, VSh); 
        __syncthreads();
        bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, RReg, VReg, RReg);
    }
}

/**
 * Refine the approximation of the quotient
 */
template<typename Base, uint32_t M, uint32_t Q>
__device__ inline void
refine3( volatile typename Base::uint_t* USh
      , volatile typename Base::uint_t* VSh
      , typename Base::uint_t VReg[Q]
      , typename Base::uint_t TReg[Q]
      , int h
      , int k
      , int l                               
      , typename Base::uint_t RReg[Q]
) {
    using uint_t = typename Base::uint_t;

    shift<uint_t, M, Q>(2, RReg, (uint_t*)USh, RReg);

    for (int i = 0; i < (int)ceilf(max(log2f(h-k-1), 0.0f)) + 2; i++) {
        int n = min(h - k + 1 - l, l);      
        int s = max(0, k - 2 * l + 1 - 2);       
        shift<uint_t, M, Q>(-s, VReg, VSh, TReg);
        __syncthreads();
        step<Base, M, Q>(USh, VSh, k + l + n - s + 2, TReg, RReg, n, l, 2);
        __syncthreads();
        if (i < 2) {
            shift<uint_t, M, Q>(-n, RReg, USh, RReg);
        }
        else {
            shift<uint_t, M, Q>(-1, RReg, USh, RReg);
            l = l + n - 1;
        }
    }
    shift<uint_t, M, Q>((h - k < 2) ? h - k - 4 : -2, RReg, VSh, RReg);
}

template<typename Base, uint32_t M, uint32_t Q>
__device__ inline void
refine2( volatile typename Base::uint_t* USh
      , volatile typename Base::uint_t* VSh
      , typename Base::uint_t VReg[Q]
      , typename Base::uint_t TReg[Q]
      , int h
      , int k
      , int l                               
      , typename Base::uint_t RReg[Q]
) {
    using uint_t = typename Base::uint_t;

    shift<uint_t, M, Q>(2, RReg, (uint_t*)USh, RReg);

    for (int i = 0; i < (int)ceilf(max(log2f(h-k-1), 0.0f)) + 2; i++) {
        int n = min(h - k + 1 - l, l);      
        int s = 0;       
        shift<uint_t, M, Q>(-s, VReg, VSh, TReg);
        __syncthreads();
        step<Base, M, Q>(USh, VSh, k + l + n - s + 2, TReg, RReg, n, l, 2);
        __syncthreads();
        if (i < 2) {
            shift<uint_t, M, Q>(-(n > 0) - (n > 1), RReg, USh, RReg);
        }
        else {
            shift<uint_t, M, Q>(-1, RReg, USh, RReg);
            l = l + n - 1;
        }
    }
    shift<uint_t, M, Q>((h - k < 2) ? h - k - 4 : -2, RReg, VSh, RReg);
}

template<typename Base, uint32_t M, uint32_t Q>
__device__ inline void
refine1( volatile typename Base::uint_t* USh
      , volatile typename Base::uint_t* VSh
      , typename Base::uint_t VReg[Q]
      , typename Base::uint_t TReg[Q]
      , int h
      , int k
      , int l                               
      , typename Base::uint_t RReg[Q]
) {
    using uint_t = typename Base::uint_t;
    h = h+1;
    shift<uint_t, M, Q>(h-k-l, RReg, (uint_t*)USh, RReg);

    for (int i = 0; i < (int)ceilf(max(log2f(h-k-1), 0.0f)) + 2; i++) {
        __syncthreads();
        shift<uint_t, M, Q>(0, VReg, VSh, TReg);
        __syncthreads();
        step<Base, M, Q>(USh, VSh, h, TReg, RReg, 0, l, 0);
        __syncthreads();
        if (i > 1) {
            l = min(2*l-1, h-k);
        }
    }
    shift<uint_t, M, Q>(-1, RReg, VSh, RReg);
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
     , int k
     , typename Base::uint_t RReg[Q]
) {
    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using uquad_t = typename Base::uquad_t;
    
    if (k == 0) {
        quo<Base, Q>(h, VSh[0], VSh, RReg);
        return;
    }
    if (k >= h && !eq<uint_t, Q>(VReg, h, &USh[2])) {
        return;
    }
    if (k == h-1 && VSh[k] > Base::HIGHEST / 2 ) {
        set<uint_t, Q>(RReg, 1, 0);
        return;
    }
    if (eq<uint_t, Q>(VReg, k, &USh[3])) {
        set<uint_t, Q>(RReg, 1, h - k);
        return;
    }

    if (threadIdx.x == 0) {
        ubig_t tmp;
        ubig_t V = (ubig_t)VSh[k - 1] | (ubig_t)VSh[k] << Base::bits;

        if (Base::bits == 64) {
            tmp = divide_u256_by_u128((__uint128_t)1 << 64, 0, V);
        } else {
            tmp = ((uquad_t)1 << 3*Base::bits) / V;
        }
        RReg[0] = (uint_t)(tmp);
        RReg[1] = (uint_t)(tmp >> Base::bits);
        if (tmp == 0) RReg[2] = 1;
    }
    __syncthreads();

    refine3<Base, M, Q>(USh, VSh, VReg, TReg, h, k, 2, RReg);
}

/**
 * Implementation of multi-precision integer division using
 * the shifted inverse and classical multiplication
 */
template<typename Base, uint32_t M, uint32_t Q>
__device__ inline void
divShinv( volatile typename Base::uint_t* USh
        , volatile typename Base::uint_t* VSh
        , typename Base::uint_t UReg[Q]
        , typename Base::uint_t VReg[Q]
        , typename Base::uint_t RReg1[2*Q]
        , typename Base::uint_t RReg2[Q]
) {
    using uint_t = typename Base::uint_t;
    using carry_t = typename Base::carry_t;

    int h = prec<uint_t, Q>(UReg, (uint32_t*)USh);     
    int k = prec<uint_t, Q>(VReg, (uint32_t*)&USh[1]) - 1; 

    shinv<Base, M, Q>(USh, VSh, VReg, RReg2, h, k, RReg1);
    __syncthreads();

    bmulRegsQComplete<Base, 1, Q/2>(USh, VSh, UReg, RReg1, RReg1, M);
    __syncthreads();

    shiftDouble<uint_t, M, Q>(-h, RReg1, VSh, RReg1);
    __syncthreads();

    bmulRegsQ<Base, 1, Q/2>(USh, VSh, VReg, RReg1, RReg2, M); 
    __syncthreads();

    if(lt<uint_t, Q>(UReg, RReg2, USh)) {
        __syncthreads();
        sub<Base, Q>(RReg1, 0, VSh);
        bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, RReg2, VReg, RReg2);
    }
    __syncthreads();
    bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, UReg, RReg2, RReg2);
    if (!lt<uint_t, Q>(RReg2, VReg, USh)) {
        __syncthreads();
        add1<Base, Q>(RReg1, USh);
        bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, RReg2, VReg, RReg2);
    }
}

template<typename Base, uint32_t M, uint32_t Q>
__global__ void 
__launch_bounds__(M/Q, BLOCKS_PER_SM*1024*Q/M)
divShinvKer( typename Base::uint_t* u
        , typename Base::uint_t* v
        , typename Base::uint_t* quo
        , typename Base::uint_t* rem
) {
    using uint_t = typename Base::uint_t;

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

    divShinv<Base, M, Q>(USh, VSh, UReg, VReg, RReg1, RReg2);
    __syncthreads();

    cpyReg2Sh2Glb<uint_t, M, Q>(quo, VSh, RReg1);
    cpyReg2Sh2Glb<uint_t, M, Q>(rem, USh, RReg2);
}

/**
 * Implementation of Euclidean algorithm (GCD)
 */
template<typename Base, uint32_t M, uint32_t Q>
__global__ void 
__launch_bounds__(M/Q, BLOCKS_PER_SM*1024*Q/M)
gcd( typename Base::uint_t* u
   , typename Base::uint_t* v
   , typename Base::uint_t* res
) {
    using uint_t = typename Base::uint_t;

    extern __shared__ char sh_mem[];
    volatile uint_t* VSh = (uint_t*)sh_mem;
    volatile uint_t* USh = (uint_t*)(VSh + M);
    uint_t VReg[Q];
    uint_t UReg[Q];
    uint_t RReg1[2*Q] = {0};
    uint_t* RReg2 = &RReg1[Q];
    
    cpyGlb2Sh2Reg<uint_t, M, Q>(u, USh, VReg);
    cpyGlb2Sh2Reg<uint_t, M, Q>(v, VSh, RReg2);
    __syncthreads();

    while(!(ez<uint_t, Q>(VReg, USh))) {
        cpyReg2Reg<uint_t, Q>(VReg, UReg);
        cpyReg2Reg<uint_t, Q>(RReg2, VReg);
        cpyReg2Shm<uint_t,Q>(VReg, VSh);
        zeroReg<uint_t, Q>(RReg1);
        zeroReg<uint_t, Q>(RReg2);
        __syncthreads();
        divShinv<Base, M, Q>(USh, VSh, UReg, VReg, RReg1, RReg2);
        __syncthreads();
    }
    __syncthreads();
    cpyReg2Sh2Glb<uint_t, M, Q>(res, VSh, UReg);
}