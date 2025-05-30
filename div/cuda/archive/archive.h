template<class S, uint32_t Q>
__device__ inline 
void from4Reg2ShmQ1( S lhcs[2][Q+2], S Lrg[2*Q], S Hrg[2*Q], volatile S* Lsh, volatile S* Hsh, uint32_t n ) {
    __syncthreads();

    const uint32_t Q2 = 2*Q;
    const uint32_t offset = ( threadIdx.x / (n/Q2) ) * n;
    uint32_t tid_mod_m = threadIdx.x % (n/Q2);
    int32_t n_m_2ltid = offset + n - Q*tid_mod_m - Q;
    int32_t twoltid = offset + Q*tid_mod_m;
    {    
        #pragma unroll
        for(int q=0; q<Q; q++) {
            Lsh[twoltid+q] = lhcs[0][q];
        }
        #pragma unroll
        for(int q=0; q<Q; q++) {
            Lsh[n_m_2ltid + q] = lhcs[1][q];
        }

        __syncthreads();
        cpyShm2Reg<S,2*Q>( Lsh, Lrg );
        __syncthreads();
    
    }
    //__syncthreads();
    {
        #pragma unroll
        for(int q=2; q<Q; q++) {
            Hsh[twoltid+q] = 0;
        }
        Hsh[twoltid+Q]   = lhcs[0][Q];
        Hsh[twoltid+Q+1] = lhcs[0][Q+1];

        #pragma unroll
        for(int q=2; q<Q; q++) {
            Hsh[n_m_2ltid + q] = 0;
        }
        S high = lhcs[1][Q];
        S carry= lhcs[1][Q+1];
        uint32_t ind = n_m_2ltid + Q;
        if( tid_mod_m == 0 ) {
            high  = 0;
            carry = 0;
            ind   = offset;
        }
        Hsh[ind]   = high;
        Hsh[ind+1] = carry;
        
        __syncthreads();
        cpyShm2Reg<S,2*Q>( Hsh, Hrg );
        __syncthreads();
    }
}


template<class Base, uint32_t IPB, uint32_t Q>
__device__ 
void bmulRegsQComplete( volatile typename Base::uint_t* Ash
              , volatile typename Base::uint_t* Bsh
              , typename Base::uint_t Arg[2*Q]
              , typename Base::uint_t Brg[2*Q]
              , typename Base::uint_t Rrg[4*Q]
              , uint32_t M
              ) 
{
    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;
    
    // 1. copy from global to shared to register memory
    cpyReg2Shm<uint_t,2*Q>( Arg, Ash );
    cpyReg2Shm<uint_t,2*Q>( Brg, Bsh );
    __syncthreads();
  
    // 2. perform the convolution
    uint_t lhcs[2][2*Q+2];

    wrapperConvQ1<uint_t, ubig_t, 2*Q>( Ash, Bsh, lhcs, M );
    __syncthreads();

    volatile typename Base::uint_t* Lsh = Bsh;

    uint_t Lrg[4*Q];
    uint_t Hrg[4*Q];

    from4Reg2ShmQ1<uint_t, Q*2>( lhcs, Lrg, Hrg, Lsh, Lsh, M*2 );
    __syncthreads();

    baddRegs<uint_t, uint_t, carry_t, 4*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, Rrg, M*2 );
}


/**
 * An inefficient multiplication implementation
 * utilizing shared memory, but allows dangling threads.
 */
template<class Base>
__device__ 
void naiveMult( volatile typename Base::uint_t* Ash
              , volatile typename Base::uint_t* Bsh
              , volatile typename Base::ubig_t* Csh
              , typename Base::uint_t Arg[]
              , typename Base::uint_t Brg[]
              , typename Base::uint_t Rrg[]
              , uint32_t M
) {
    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;
    
    int Q = (M + blockDim.x - 1) / blockDim.x;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M) {
            Ash[idx] = Arg[i];
            Bsh[idx] = Brg[i];
        }
    }
    __syncthreads();

    // Not using Q offset
    if (threadIdx.x < M) {
        ubig_t acc = 0;
        for (int i = 0; i <= threadIdx.x; i++) {
            int j = threadIdx.x - i;    
            if (j < M) {
                acc += Ash[i] * Bsh[j];
            }
        }
        Csh[threadIdx.x] = acc;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M) {
            Ash[2 * idx]     = (uint_t) Csh[idx];
            Ash[2 * idx + 1] = (uint_t) (Csh[idx] >> Base::bits);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < 2*M) {
            Rrg[i] = Ash[idx];
        }
    }
}


/**
 * Branchless version of from4Reg2ShmQ2
 */
template<class S, uint32_t Q>
__device__ inline 
void from4Reg2ShmQ2Brnchless( S lhcs[Q+2]
                            , volatile S* Lsh
                            , volatile S* Hsh
                            , S highCarry[2]
                            , bool isFirst
                            , uint32_t n
) {
    const uint32_t Q2 = 2*Q;
    uint32_t tid_mod_m = threadIdx.x % (n/Q2);
    int32_t twoltid = isFirst ? Q*tid_mod_m : n/2 - Q*tid_mod_m - Q;

    #pragma unroll
    for(int q=0; q<Q; q++) {
        Lsh[twoltid+q] = lhcs[q];
    }
    
    #pragma unroll
    for(int q=2; q<Q; q++) {
        Hsh[twoltid+q] = 0;
    }
    // __syncthreads(); // Can maybe be omitted

    int condition = (threadIdx.x != n/Q2 - 1);
    Hsh[twoltid+Q]   = lhcs[Q]   * condition + Hsh[twoltid+Q]   * (1 - condition);
    Hsh[twoltid+Q+1] = lhcs[Q+1] * condition + Hsh[twoltid+Q+1] * (1 - condition);
    __syncthreads();

    int condition1 = (isFirst && threadIdx.x == n/Q2 - 1);
    int condition2 = (!isFirst && threadIdx.x == n/Q2 - 1);

    highCarry[0] = lhcs[Q]   * condition1 + highCarry[0] * (1 - condition1);
    highCarry[1] = lhcs[Q+1] * condition1 + highCarry[1] * (1 - condition1);
    Hsh[0]       = 0         * condition1 + Hsh[0]       * (1 - condition1);
    Hsh[1]       = 0         * condition1 + Hsh[1]       * (1 - condition1);

    Hsh[Q]       = lhcs[Q]   * condition2 + Hsh[Q]       * (1 - condition2);
    Hsh[Q+1]     = lhcs[Q+1] * condition2 + Hsh[Q+1]     * (1 - condition2);
    Lsh[0]       = lhcs[0]   * condition2 + Lsh[0]       * (1 - condition2);
    Lsh[1]       = lhcs[1]   * condition2 + Lsh[1]       * (1 - condition2);
    Hsh[0]       = highCarry[0] * condition2 + Hsh[0]       * (1 - condition2);
    Hsh[1]       = highCarry[1] * condition2 + Hsh[1]       * (1 - condition2);
    __syncthreads();

    condition = isFirst && threadIdx.x == 0;
    Hsh[0] = 0 * condition + Hsh[0] * (1 - condition);
    Hsh[1] = 0 * condition + Hsh[1] * (1 - condition);
}


template<class uint_t, uint32_t Q>
 __device__ inline bool
 eq( uint_t u[Q]
   , uint32_t bpow
   , volatile uint_t* sh_mem
 ) {
     sh_mem[0] = true;
     __syncthreads(); 

     #pragma unroll
     for (int i = 0; i < Q; i++) {
         if (u[i] != (bpow == (i * blockDim.x + threadIdx.x))) {
            sh_mem[0] = false;
            break;
         }
     }
     __syncthreads();    
     return sh_mem[0];  
 }

 bsubaddRegs<uint_t, uint_t, carry_t, Q, Base::HIGHEST>(sign, (carry_t*)VSh, RReg, VReg, RReg);

 /**
 * Subtraction implementation between two
 * bigints stored in register memory; uses
 * redundant computation to aleviate register
 * pressure.
 */
template<class D, class S, class CT, uint32_t Q, D HIGHEST>
__device__ inline void
bsubaddRegs(
          bool is_add 
        , volatile CT* Csh
        , D Arg[Q]
        , S Brg[Q]
        , D rs[Q]
) {
    {
        CT accum = CarrySegBop<CT>::identity();
        #pragma unroll
        for(int i=0; i<Q; i++) {
            D a = Arg[i];
            S b = Brg[i];
            CT c;
            
            if(is_add) {
                rs[i] = a + (D)b;
                c = (CT) ( (rs[i] < a) );
                c = c | ((rs[i] == HIGHEST) << 1);
            } else {
                rs[i] = a - (D)b;
                c = (CT) ( (rs[i] > a) );
                c = c | ((rs[i] == 0) << 1);
            }
            accum = CarrySegBop<CT>::apply(accum, c);
        }
        Csh[threadIdx.x] = accum;
    }
    __syncthreads();
   
    scanIncBlock< CarrySegBop<CT> >(Csh, threadIdx.x);
        
    {
        CT carry = CarrySegBop<CT>::identity();
        if(threadIdx.x > 0) {
            carry = Csh[threadIdx.x - 1];
        }
        #pragma unroll
        for(int i=0; i<Q; i++) {
            D a = Arg[i];
            S b = Brg[i];
            CT c;
            
            if(is_add) {
                rs[i] = a + (D)b;
                c = (CT) ( (rs[i] < a) );
                c = c | ((rs[i] == HIGHEST) << 1);
                rs[i] += (carry & 1);
            } else {
                rs[i] = a - (D)b;
                c = (CT) ( (rs[i] > a) );
                c = c | ((rs[i] == 0) << 1);
                rs[i] -= (carry & 1);
            }
            carry = CarrySegBop<CT>::apply(carry, c);         
        }
    }
}




// /**
//  * Implementation of multi-precision integer quotient using
//  * the shifted inverse and classical multiplication
//  */
// template<typename Base, uint32_t M, uint32_t Q>
// __global__ void
// __launch_bounds__(M/Q, BLOCKS_PER_SM*1024*Q/M)
// quoShinv( typename Base::uint_t* u
//         , typename Base::uint_t* v
//         , typename Base::uint_t* quo
// ) {
//     using uint_t = typename Base::uint_t;
//     using carry_t = typename Base::carry_t;

//     extern __shared__ char sh_mem[];
//     volatile uint_t* VSh = (uint_t*)sh_mem;
//     volatile uint_t* USh = (uint_t*)(VSh + M);
//     uint_t VReg[Q];
//     uint_t UReg[Q];
//     uint_t RReg1[2*Q] = {0};
//     uint_t* RReg2 = &RReg1[Q];

//     cpyGlb2Sh2Reg<uint_t, M, Q>(v, VSh, VReg);
//     cpyGlb2Sh2Reg<uint_t, M, Q>(u, USh, UReg);
//     __syncthreads();

//     int h = prec<uint_t, Q>(UReg, (uint32_t*)USh);
//     int k = prec<uint_t, Q>(VReg, (uint32_t*)&USh[4]) - 1;

//     shinv<Base, M, Q>(USh, VSh, VReg, RReg2, h, k, RReg1);
//     __syncthreads();

//     bmulRegsQComplete<Base, 1, Q/2>(USh, VSh, UReg, RReg1, RReg1, M);
//     __syncthreads();

//     shiftDouble<uint_t, M, Q>(-h, RReg1, VSh, RReg1);
//     __syncthreads();

//     bmulRegsQ<Base, 1, Q/2>(USh, VSh, VReg, RReg1, RReg2, M);
//     __syncthreads();

//     if(lt<uint_t, Q>(UReg, RReg2, USh)) {
//         __syncthreads();
//         sub<Base, Q>(RReg1, 0, VSh);
//         bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, RReg2, VReg, RReg2);
//     }
//     __syncthreads();
    
//     bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, UReg, RReg2, RReg2);
//     if (!lt<uint_t, Q>(RReg2, VReg, USh)) { 
//         __syncthreads();
//         add1<Base, Q>(RReg1, VSh); 
//     }

//     __syncthreads();
//     cpyReg2Sh2Glb<uint_t, M, Q>(quo, VSh, RReg1);
// }



// /**
//  * Implementation of greatest common divisor 
//  */
// template<typename Base, uint32_t M, uint32_t Q>
// __global__ void 
// __launch_bounds__(M/Q, BLOCKS_PER_SM*1024*Q/M)
// gcd( typename Base::uint_t* u
//    , typename Base::uint_t* v
//    , typename Base::uint_t* r
// ) {
//     using uint_t = typename Base::uint_t;
//     using carry_t = typename Base::carry_t;

//     extern __shared__ char sh_mem[];
//     volatile uint_t* VSh = (uint_t*)sh_mem;
//     volatile uint_t* USh = (uint_t*)(VSh + M);
//     uint_t VReg[Q];
//     uint_t UReg[Q];
//     uint_t RReg1[2*Q] = {0};
//     uint_t* RReg2 = &RReg1[Q];
//     uint_t TReg[Q] = {0};

//     cpyGlb2Sh2Reg<uint_t, M, Q>(v, VSh, VReg);
//     cpyGlb2Sh2Reg<uint_t, M, Q>(u, USh, UReg);
//     __syncthreads();

//     do {
//         int h = prec<uint_t, Q>(UReg, (uint32_t*)USh);     
//         int k = prec<uint_t, Q>(VReg, (uint32_t*)&USh[4]) - 1; 

//         shinv<Base, M, Q>(USh, VSh, VReg, RReg2, h, k, RReg1);
//         __syncthreads();

//         bmulRegsQComplete<Base, 1, Q/2>(USh, VSh, UReg, RReg1, RReg1, M);
//         __syncthreads();

//         shiftDouble<uint_t, M, Q>(-h, RReg1, VSh, RReg1);
//         __syncthreads();

//         bmulRegsQ<Base, 1, Q/2>(USh, VSh, VReg, RReg1, RReg2, M); 
//         __syncthreads();

//         if(lt<uint_t, Q>(UReg, RReg2, USh)) {
//             __syncthreads();
//             sub<Base, Q>(RReg1, 0, VSh);
//             bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, RReg2, VReg, RReg2);
//         }
//         __syncthreads();
//         bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, UReg, RReg2, RReg2);
//         if (!lt<uint_t, Q>(RReg2, VReg, USh)) {
//             __syncthreads();
//             add1<Base, Q>(RReg1, USh);
//             bsubRegs<uint_t, uint_t, carry_t, Q>((carry_t*)VSh, RReg2, VReg, RReg2);
//         }

//         #pragma unroll
//         for (int i = 0; i < Q; i++) {
//             UReg[i] = VReg[i];
//             VReg[i] = RReg2[i];
//         }
//         if (!lt<uint_t, Q>(RReg2, VReg, VSh)){
//             zeroAndSet<uint_t, Q>(UReg,1,0);
//             break;
//         }
//     } while ((!(ez<uint_t, Q>(VReg, VSh))));

//     __syncthreads();
//     cpyReg2Sh2Glb<uint_t, M, Q>(r, USh, UReg);
// }


template<class Base, uint32_t Q>
__device__ 
void naiveMult( volatile typename Base::uint_t* Ash
              , volatile typename Base::uint_t* Bsh
              , typename Base::uint_t Arg[Q]
              , typename Base::uint_t Brg[Q]
              , typename Base::uint_t Rrg[Q]
              , uint32_t M
) {
    using uint_t = typename Base::uint_t;
    using ubig_t = typename Base::ubig_t;
    using carry_t= typename Base::carry_t;
    
    if (threadIdx.x < (M+Q-1)/Q) {
        #pragma unroll
        for(int i=0; i<Q; i++) {
            Ash[Q*threadIdx.x + i] = Arg[i];
            Bsh[Q*threadIdx.x + i] = Brg[i];
        }
    }
    __syncthreads();

    ubig_t accum = 0;
    carry_t carry = 0;
    if (threadIdx.x < M) {
        #pragma unroll
        for (int i = 0; i <= threadIdx.x; i++) {
            ubig_t ck = (ubig_t)Ash[i] * (ubig_t)Bsh[threadIdx.x - i];
            uint_t accum_prev = (uint_t) (accum >> Base::bits);
            accum += ck;
            carry += (  ((uint_t)(accum >> Base::bits)) < accum_prev );
        }
    }
    if (threadIdx.x == 0) {
        Bsh[M] = 0;
        Ash[2*M] = 0;
        Ash[2*M + 1] = 0;
    }
    if (threadIdx.x < M) {
        Ash[M + threadIdx.x] = (uint_t)accum;
        Bsh[M + threadIdx.x + 1] = (uint_t) (accum >> Base::bits);
        Ash[2*M + threadIdx.x + 2] = carry;
    }
    __syncthreads();

    uint_t arg;
    uint_t brg;
    uint_t crg;
    if (threadIdx.x < M) {
        arg = Ash[M + threadIdx.x];
        brg = Bsh[M + threadIdx.x];
        crg = Ash[2*M + threadIdx.x];
    }

    uint_t res = baddRegsNaive<uint_t, uint_t, carry_t, Base::HIGHEST>( (carry_t*)&Bsh[2*blockDim.x], arg, brg );
    res = baddRegsNaive<uint_t, uint_t, carry_t, Base::HIGHEST>( (carry_t*)Bsh, res, crg );

    Ash[threadIdx.x] = res;
    __syncthreads();

    if (threadIdx.x < (M+Q-1)/Q) {
        #pragma unroll
        for(int i=0; i<Q; i++) {
            int idx = Q*threadIdx.x + i;
            if (idx < M) {
                Rrg[i] = Ash[idx];
            }
        }
    }
}