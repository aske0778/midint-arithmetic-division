#include "ker-helpers.cu.h"
#include "../../cuda/helper.h"
#include "../../cuda/ker-classic-mul.cu.h"

template<uint32_t M, uint32_t Q>
__device__ inline void
multMod(volatile uint32_t* USh, volatile uint32_t* VSh, uint32_t UReg[Q], uint32_t VReg[Q], int d, uint32_t RReg[Q]) {
    bmulRegsQ<U32bits, 1, M, Q/2>((uint32_t*)USh, (uint32_t*)VSh, UReg, VReg, RReg); 
    #pragma unroll
    for (int i=0; i < Q; i++) {
        if (Q * threadIdx.x + i >= d)
        {
            RReg[i] = 0;
        }
    }
  //  __syncthreads();
}

template<uint32_t M, uint32_t Q>
__device__ inline bool
powDiff(volatile uint32_t* USh, volatile uint32_t* VSh, uint32_t VReg[Q], uint32_t RReg[Q], int h, int l) {
    int vPrec = prec<Q>(VReg, USh);
    __syncthreads();
    int rPrec = prec<Q>(RReg, VSh);
    __syncthreads();
    int L = vPrec + rPrec - l + 1;
    bool sign = 1;
 //   __syncthreads();
    
    if (vPrec == 0 || rPrec == 0) {
        zeroAndSet<Q>(VReg, 1, h);
    }
    else if (L >= h) {
        bmulRegsQ<U32bits, 1, M, Q/2>((uint32_t*)USh, (uint32_t*)VSh, VReg, RReg, VReg); 
        sub<Q>(h, VReg, USh);
    }
    else {
        multMod<M, Q>(USh, VSh, VReg, RReg, L, VReg);
      //  __syncthreads();
        if (!ez<Q>(VReg, USh)) {
          //  __syncthreads();
            if (ez<Q>(VReg, L-1, VSh)) {
                sign = 0;
            }
            else {
            //    __syncthreads();
                sub<Q>(L, VReg, USh);
            }
        }
    }
    __syncthreads();
    return sign;
}

template<uint32_t M, uint32_t Q>
__device__ inline void
step(volatile uint32_t* USh, volatile uint32_t* VSh, int h, uint32_t VReg[Q], uint32_t RReg[Q], int n, int l) {
    bool sign = powDiff<M, Q>(USh, VSh, VReg, RReg, h - n, l - 2);
    __syncthreads();
    bmulRegsQ<U32bits, 1, M, Q/2>((uint32_t*)USh, (uint32_t*)VSh, RReg, VReg, VReg); 
    shift<M, Q>(2 * n - h, VReg, VSh, VReg);
    __syncthreads();
    shift<M, Q>(n, RReg, USh, RReg);
    __syncthreads();

    if (sign) {
        baddRegs<uint32_t, uint32_t, uint32_t, M, Q, UINT32_MAX>((uint32_t*)VSh, RReg, VReg, RReg);
    }
    else {
        bsubRegs<uint32_t, uint32_t, uint32_t, M, Q, UINT32_MAX>((uint32_t*)VSh, RReg, VReg, RReg);
    }
}

template<uint32_t M, uint32_t Q>
__device__ inline void
refine(volatile uint32_t* USh, volatile uint32_t* VSh, uint32_t VReg[Q], uint32_t TReg[Q], int h, int k, int l, uint32_t RReg[Q]) {
    shift<M, Q>(2, RReg, USh, RReg);
    __syncthreads();
    while (h - k > l) {
        int n = min(h - k + 1 - l, l);
        int s = max(0, k - 2 * l + 1 - 2);
        shift<M, Q>(-s, VReg, USh, TReg);
        __syncthreads();
        step<M, Q>(USh, VSh, k + l + n - s + 2, TReg, RReg, n, l);
        __syncthreads();
        shift<M, Q>(-1, RReg, USh, RReg);
        __syncthreads();
        l = l + n - 1;
    }
    shift<M, Q>(-2, RReg, USh, RReg);
}

template<uint32_t M, uint32_t Q>
__device__ inline void
shinv(volatile uint32_t* USh, volatile uint32_t* VSh, uint32_t VReg[Q], uint32_t TReg[Q], int h, uint32_t RReg[Q]) {

    int k = prec<Q>(VReg, USh) - 1;
    __syncthreads();

    if (k == 0) {
        quo<Q>(h, VSh[0], RReg);
        return;
    }
    __syncthreads();
    if (k >= h && !eq<Q>(VReg, h, USh)) {
        return;
    }
    __syncthreads();
    if (k == h-1 && VSh[k] > UINT32_MAX / 2 ) {
     //   __syncthreads();
        set<Q>(RReg, 1, 0);
        return;
    }
    __syncthreads();
    if (eq<Q>(VReg, k, USh)) {
     //   __syncthreads();
        set<Q>(RReg, 1, h - k);
        return;
    }
  //  __syncthreads();
    int l = min(k, 2);    
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
    __syncthreads();

    if (h - k <= l) {
        shift<M, Q>(h-k-l, RReg, USh, RReg);
    }
    else {
        refine<M, Q>(USh, VSh, VReg, TReg, h, k, l, RReg);
    }
}

template<uint32_t M, uint32_t Q>
__global__ void divShinv(uint32_t* u, uint32_t* v, uint32_t* quo, uint32_t* rem, const uint32_t num_instances) {
    extern __shared__ char sh_mem[];
    volatile uint32_t* VSh = (uint32_t*)sh_mem;
    volatile uint32_t* USh = (uint32_t*)(VSh + M);
    uint32_t VReg[Q];
    uint32_t UReg[Q];
    uint32_t RReg1[Q] = {0};
    uint32_t RReg2[Q] = {0};



    cpyGlb2Sh2Reg<M, Q>(v, VSh, VReg);
    cpyGlb2Sh2Reg<M, Q>(u, USh, UReg);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf(" \n [");
        for (int i = 0; i < M; i++){
            printf("( %u at BLOCK(%d) ), ", VSh[i], blockIdx.x);
        }
        printf("] \n");
    }
    __syncthreads();


    int h = prec<Q>(UReg, USh);
    __syncthreads();

    shinv<M, Q>(USh, VSh, VReg, RReg2, h, RReg1);
    __syncthreads();

    bmulRegsQ<U32bits, 1, M, Q/2>((uint32_t*)USh, (uint32_t*)VSh, UReg, RReg1, RReg1);
    __syncthreads();
    
    shift<M, Q>(-h, RReg1, USh, RReg1);
    __syncthreads();

    bmulRegsQ<U32bits, 1, M, Q/2>((uint32_t*)USh, (uint32_t*)VSh, VReg, RReg1, RReg2); 
    __syncthreads();
    bsubRegs<uint32_t, uint32_t, uint32_t, M, Q, UINT32_MAX>((uint32_t*)VSh, UReg, RReg2, RReg2);
    __syncthreads();

    if (!lt<Q>(RReg2, VReg, USh)) {
        __syncthreads();
        add1<Q>(RReg1, USh);
        __syncthreads();
        bsubRegs<uint32_t, uint32_t, uint32_t, M, Q, UINT32_MAX>((uint32_t*)VSh, RReg2, VReg, RReg2);
    }
    __syncthreads();
    cpyReg2Sh2Glb<M, Q>(RReg1, VSh, quo);
    cpyReg2Sh2Glb<M, Q>(RReg2, USh, rem);
    __syncthreads();
}