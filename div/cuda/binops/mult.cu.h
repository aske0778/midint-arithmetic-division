/**
 * 
 */
template<class S, uint32_t Q>
__device__ inline 
void from4Reg2ShmQ( S lhcs[2][Q+2]
                  , volatile S* Lsh
                  , volatile S* Hsh
                  , uint32_t n
) {
    const uint32_t Q2 = 2*Q;
    const uint32_t offset = ( threadIdx.x / (n/Q2) ) * n;
    uint32_t tid_mod_m = threadIdx.x % (n/Q2);

    {    
        int32_t twoltid = offset + Q*tid_mod_m;
        #pragma unroll
        for(int q=0; q<Q; q++) {
            Lsh[twoltid+q] = lhcs[0][q];
        }
        #pragma unroll
        for(int q=2; q<Q; q++) {
            Hsh[twoltid+q] = 0;
        }
        Hsh[twoltid+Q]   = lhcs[0][Q];
        Hsh[twoltid+Q+1] = lhcs[0][Q+1];
    }
    __syncthreads();
    {
        int32_t n_m_2ltid = offset + n - Q*tid_mod_m - Q;
        #pragma unroll
        for(int q=0; q<Q; q++) {
            Lsh[n_m_2ltid + q] = lhcs[1][q];
        }
        #pragma unroll
        for(int q=2; q<Q; q++) {
            Hsh[n_m_2ltid + q] = 0;
        }
        S high = lhcs[1][Q];
        uint32_t carry= lhcs[1][Q+1];
        uint32_t ind = n_m_2ltid + Q;
        if( tid_mod_m == 0 ) {
            high  = 0;
            carry = 0;
            ind   = offset;
        }
        Hsh[ind]   = high;
        Hsh[ind+1] = carry;
    }
}

/**
 * 
 */
template<class S, class D>
__device__ inline
void computeIter64( uint32_t i
                  , uint32_t j
                  , volatile S* Ash
                  , volatile S* Bsh
                  , D& accum
                  , uint32_t& carry
) {
    const uint32_t SHFT = 8*sizeof(S);
    S ai = Ash[i];
    S bj = Bsh[j];
    D ck = ((D)ai) * ((D)bj);

    S accum_prev = (S) (accum>>SHFT);
    accum += ck;
    carry += (  ((S)(accum>>SHFT)) < accum_prev );
    //if (accum < ck) carry++;
}

/**
 * 
 */
template<class S, class D, uint32_t Q>
__device__ inline
void combineQ( D accums[Q]
             , uint32_t carrys[Q]
             , S lhcs[Q+2]
) {
    uint32_t SHFT = 8 * sizeof(S);
    
    lhcs[0] = (S) accums[0];
    S h_res = (S) (accums[0] >> SHFT);
    S c_res = carrys[0];

    #pragma unroll
    for(int q=1; q<Q; q++) {
        S l = (S) accums[q];
        S h = (S) (accums[q] >> SHFT);
        lhcs[q] = l + h_res;
        h_res = h + (c_res + (lhcs[q] < l));
        c_res = carrys[q] + (h_res < h);
    }
    lhcs[Q]   = h_res;
    lhcs[Q+1] = c_res;
}

/**
 * 
 */
template<class S, class D, uint32_t Q>
__device__ inline
void convolutionQ( uint32_t k1
                 , volatile S* Ash
                 , volatile S* Bsh
                 , S lhcs[Q+2]
) {
    D        accums[Q]; 
    uint32_t carrys[Q];
    
    #pragma unroll
    for(int q=0; q<Q; q++) { 
        accums[q] = 0; 
        carrys[q] = 0; 
    }

    for(int kk = 0; kk <= k1; kk++) {
        uint32_t i = kk;
        uint32_t j = k1 - i;

        #pragma unroll
        for(int q=0; q<Q; q++) {
            computeIter64<S,D>( i, j+q,   Ash, Bsh, accums[q], carrys[q] );
        }
    }

    #pragma unroll
    for(int q=1; q<Q; q++) {
        #pragma unroll
        for(int i=0; i<Q-q; i++) {
            computeIter64<S,D>(k1+q, i, Ash, Bsh, accums[i+q], carrys[i+q]);
        }
    }

    combineQ<S,D,Q>(accums, carrys, lhcs);
}

/**
 * 
 */
template<class S, class D, uint32_t Q>
__device__ inline
void wrapperConvQ( volatile S* Ash0
                 , volatile S* Bsh0
                 , S lhcs[2][Q+2]
                 , uint32_t M
) {
    const uint32_t offset = ( threadIdx.x / (M/(2*Q)) ) * M;
    volatile S* Ash = Ash0 + offset;
    volatile S* Bsh = Bsh0 + offset;

    uint32_t ltid = threadIdx.x % (M/(2*Q));
    { // first half
        uint32_t k1 = Q*ltid;
        convolutionQ<S,D,Q>(k1, Ash, Bsh, lhcs[0]);
    }

    { // second half
        uint32_t k2 = M - Q*ltid - Q;
        convolutionQ<S,D,Q>(k2, Ash, Bsh, lhcs[1]);
    }
}

/**
 * 
 */
template<class Base, uint32_t IPB, uint32_t Q>
__device__ 
void bmulRegsQ( volatile typename Base::uint_t* Ash
              , volatile typename Base::uint_t* Bsh
              , typename Base::uint_t Arg[2*Q]
              , typename Base::uint_t Brg[2*Q]
              , typename Base::uint_t Rrg[2*Q]
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
    uint_t lhcs[2][Q+2];

    wrapperConvQ<uint_t, ubig_t, Q>( Ash, Bsh, lhcs, M );
    __syncthreads();

    volatile typename Base::uint_t* Lsh = Ash;
    volatile typename Base::uint_t* Hsh = Bsh;

    // 3. publish the low parts normally, and the high and carry shifted by one.
    from4Reg2ShmQ<uint_t, Q>( lhcs, Lsh, Hsh, M );
    __syncthreads();


    // 4. load back to register and perform the addition of the carries.
    uint_t Lrg[2*Q];
    cpyShm2Reg<uint_t,2*Q>( Lsh, Lrg );
    uint_t Hrg[2*Q];
    cpyShm2Reg<uint_t,2*Q>( Hsh, Hrg );
    __syncthreads();

    // Hsh[0] = 0;
    // __syncthreads();
    // for (int i=0; i < Q*2; i++) {
    //     int idx = Q*2 * threadIdx.x + i;
    //     if (idx == M/2 && (Lrg[i] != 0 || Rrg[i] != 0)) {
    //         Hsh[0] = 1;
    //     }
    // }

    // __syncthreads();
    // if (Hsh[0] == 1) {
    //     __syncthreads();
    //     printRegs1<uint_t, 2*Q>("left", Lrg, Lsh, M);
    //     __syncthreads();
    //     __syncthreads();
    //     printRegs1<uint_t, 2*Q>("right", Hrg, Lsh, M);
    //     __syncthreads();
    // }
    // printRegs<uint_t, Q*2>("Lrg", Lrg, Hsh, M);
    // __syncthreads();

    baddRegs<uint_t, uint_t, carry_t, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, Rrg, M );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 
 */
template<class S, class D>
__device__ inline
void computeIter641( uint32_t i
                   , uint32_t j
                   , volatile S* Ash
                   , volatile S* Bsh
                   , D& accum
                   , D& carry
) {
    const uint32_t SHFT = 8*sizeof(S);
    S ai = Ash[i];
    S bj = Bsh[j];
    D ck = ((D)ai) * ((D)bj);

    S accum_prev = (S) (accum>>SHFT);
    accum += ck;
    carry += (  ((S)(accum>>SHFT)) < accum_prev );
    //if (accum < ck) carry++;
}

/**
 * 
 */
template<class S, class D, uint32_t Q>
__device__ inline
void combineQ1( D accums[Q]
              , D carrys[Q]
              , S lhcs[Q+2]
) {
    uint32_t SHFT = 8 * sizeof(S);
    
    lhcs[0] = (S) accums[0];
    S h_res = (S) (accums[0] >> SHFT);
    S c_res = carrys[0];

    #pragma unroll
    for(int q=1; q<Q; q++) {
        S l = (S) accums[q];
        S h = (S) (accums[q] >> SHFT);
        lhcs[q] = l + h_res;
        h_res = h + (c_res + (lhcs[q] < l));
        c_res = carrys[q] + (h_res < h);
    }
    lhcs[Q]   = h_res;
    lhcs[Q+1] = c_res;
}

/**
 * 
 */
template<class S, class D, uint32_t Q>
__device__ inline
void convolutionQ1( uint32_t k1
                  , volatile S* Ash
                  , volatile S* Bsh
                  , S lhcs[Q+2]
                  , uint32_t M
) {
    D accums[Q]; 
    D carrys[Q];
    
    #pragma unroll
    for(int q=0; q<Q; q++) { 
        accums[q] = 0; 
        carrys[q] = 0; 
    }

    for(int kk = 0; kk <= k1; kk++) {
        uint32_t i = kk;
        uint32_t j = k1 - i;

        #pragma unroll
        for(int q=0; q<Q; q++) {
            if (i < M && j+q < M) {
                computeIter641<S,D>(i, j+q, Ash, Bsh, accums[q], carrys[q]);
            }
        }
    }

    #pragma unroll
    for(int q=1; q<Q; q++) {
        #pragma unroll
        for(int i=0; i<Q-q; i++) {
            if (k1+q < M && i < M) {
                computeIter641<S,D>(k1+q, i, Ash, Bsh, accums[i+q], carrys[i+q]);
            }
        }
    }
    combineQ1<S,D,Q>(accums, carrys, lhcs);
}

/**
 * 
 */
template<class S, class D, uint32_t Q>
__device__ inline
void wrapperConvQ1( volatile S* Ash0
                  , volatile S* Bsh0
                  , S lhcs[2][Q+2]
                  , uint32_t M
) {
    volatile S* Ash = Ash0;
    volatile S* Bsh = Bsh0;

    { // first half
        uint32_t k1 = Q*threadIdx.x;
        convolutionQ1<S,D,Q>(k1, Ash, Bsh, lhcs[0], M);
    }

    { // second half
        uint32_t k2 = 2*M - Q*threadIdx.x - Q;
        convolutionQ1<S,D,Q>(k2, Ash, Bsh, lhcs[1], M);
    }
}

/**
 * 
 */
template<class S, uint32_t Q>
__device__ inline 
void from4Reg2ShmQ2( S lhcs[Q+2]
                   , volatile S* Lsh
                   , volatile S* Hsh
                   , S highCarry[2]
                   , bool isFirst
                   , uint32_t n
) {
    const uint32_t Q2 = 2*Q;
    uint32_t tid_mod_m = threadIdx.x % (n/Q2);
    //Refactoring in progress
    {    
        int32_t twoltid = isFirst ? Q*tid_mod_m : n/2 - Q*tid_mod_m - Q;
        #pragma unroll
        for(int q=0; q<Q; q++) {
            Lsh[twoltid+q] = lhcs[q];
        }
        __syncthreads();
        #pragma unroll
        for(int q=2; q<Q; q++) {
            Hsh[twoltid+q] = 0;
        }
        __syncthreads();
        if( threadIdx.x != n/Q2 - 1) {
            Hsh[twoltid+Q]   = lhcs[Q];
            Hsh[twoltid+Q+1] = lhcs[Q+1];
        }
        __syncthreads();
        if (isFirst && threadIdx.x == n/Q2 - 1){
            highCarry[0] = lhcs[Q];
            highCarry[1] = lhcs[Q+1];
            Hsh[0] = 0;
            Hsh[1] = 0;
        }
        else if (threadIdx.x == n/Q2 - 1){
            Hsh[Q]   = lhcs[Q];
            Hsh[Q+1] = lhcs[Q+1];
            Lsh[0] = lhcs[0];
            Lsh[1] = lhcs[1];
            Hsh[0] = highCarry[0];
            Hsh[1] = highCarry[1];
        }
        __syncthreads();
        if (isFirst && threadIdx.x == 0) {
            Hsh[0] = 0;
            Hsh[1] = 0;
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

/**
 * 
 */
template<class Base, uint32_t IPB, uint32_t Q>
__device__ 
void bmulRegsQComplete( volatile typename Base::uint_t* Ash
              , volatile typename Base::uint_t* Bsh
              , typename Base::uint_t Arg[2*Q]
              , typename Base::uint_t Brg[2*Q]
              , typename Base::uint_t Rrg[4*Q]
              , uint32_t M
) {
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

    volatile uint_t* Lsh = Ash;
    volatile uint_t* Hsh = Bsh;

    // 3. publish the low parts normally, and the high and carry shifted by one.
    uint_t highCarry[2];

    from4Reg2ShmQ2<uint_t, Q*2>( lhcs[0], Lsh, Hsh, highCarry, true, M*2 );
    __syncthreads();

    // 4. load back to register and perform the addition of the carries.
    uint_t Lrg[2*Q];
    uint_t Hrg[2*Q];
    cpyShm2Reg<uint_t,2*Q>( Lsh, Lrg );
    cpyShm2Reg<uint_t,2*Q>( Hsh, Hrg );
    __syncthreads();

    bool overflow = baddRegsOverflow<uint_t, uint_t, carry_t, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, (carry_t*)Hsh, Lrg, Hrg, Rrg, M );

    from4Reg2ShmQ2<uint_t, Q*2>( lhcs[1], Lsh, Hsh, highCarry, false, M*2 );
    __syncthreads();

    cpyShm2Reg<uint_t,2*Q>( Lsh, Lrg );
    cpyShm2Reg<uint_t,2*Q>( Hsh, Hrg );
    __syncthreads();

    baddRegs<uint_t, uint_t, carry_t, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, &Rrg[Q*2], M );

    if (overflow) {
        add1<Base, Q*2>(&Rrg[Q*2], Hsh);
    }
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
    //

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
 * Assumption: Q evenly divides M
 */
template<typename Base, uint32_t IPB, uint32_t M, uint32_t Q>
__global__ void bmulKerQ( uint32_t num_instances
                        , typename Base::uint_t* ass
                        , typename Base::uint_t* bss
                        , typename Base::uint_t* rss
                        )
{
    using uint_t = typename Base::uint_t;
    const uint32_t M_lft = LIFT_LEN(M, Q);
    const uint32_t shmem_len = IPB*M_lft;

    __shared__ uint_t Ash[shmem_len];
    __shared__ uint_t Bsh[shmem_len];

    uint_t Arg[Q];
    uint_t Brg[Q];
    { // read from global memory
        const uint32_t ipb = min(num_instances - IPB*blockIdx.x, IPB);
        cpGlb2Reg<uint_t,IPB,M,Q>(ipb, Ash, ass, Arg);
        cpGlb2Reg<uint_t,IPB,M,Q>(ipb, Bsh, bss, Brg);
    }
    __syncthreads();

    uint_t Rrg[Q];
    bmulRegsQ<Base, IPB, Q/2>(Ash, Bsh, Arg, Brg, Rrg, M);

    { // write to global memory
        const uint32_t ipb = min(num_instances - IPB*blockIdx.x, IPB);
        cpReg2Glb<uint_t,IPB,M,Q>(ipb, Ash, Rrg, rss);
    }
}