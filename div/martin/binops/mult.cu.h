template<uint32_t Q>
__device__ inline void printRegs1(const char *str, uint32_t u[Q*2], volatile uint32_t* sh_mem, uint32_t M)
{
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[Q * threadIdx.x + i] = u[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[M/2 + Q * threadIdx.x + i] = u[Q+i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("%s: [", str);
        for (int i = 0; i < M; i++)
        {
            printf("%u", sh_mem[i]);
            if (i < M - 1)
                printf(", ");
        }
        printf("]\n");
    }
    __syncthreads();
}


__device__ inline void printShMem(const char *str, volatile uint32_t* sh_mem, uint32_t M)
{
    if (threadIdx.x == 0) {
        printf("%s: [", str);
        for (int i = 0; i < M; i++)
        {
            printf("%u", sh_mem[i]);
            if (i < M - 1)
                printf(", ");
        }
        printf("]\n");
    }
    __syncthreads();
}



template<uint32_t Q>
__device__ inline void printLhcs(const char *str, uint32_t lhcs[2][Q+2], volatile uint32_t* sh_mem, uint32_t M)
{
    for (int i=0; i < Q; i++) {
        sh_mem[Q * threadIdx.x + i] = lhcs[0][i];
    }
    __syncthreads();
    
    for (int i=0; i < Q; i++) {
        sh_mem[M/2 + (Q * threadIdx.x + i)] = lhcs[1][i];
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        printf("%s: [", str);
        for (int i = 0; i < M; i++)
        {
            printf("%u", sh_mem[i]);
            if (i < M - 1)
                printf(", ");
        }
        printf("]\n");
    }
    __syncthreads();
}



template<uint32_t Q>
__device__ inline void printLhcs12(const char *str, uint32_t lhcs[2][Q+2], volatile uint32_t* sh_mem, uint32_t M)
{
    for (int i=0; i < M/Q*2; i++) {
        if (threadIdx.x == i) {
            printf("%s: [", str);
            for (int i = 0; i < Q+2; i++)
            {
                printf("%u", lhcs[0][i]);
                if (i < Q+2 - 1)
                    printf(", ");
            }
            printf("]\n");
        }
    }
    for (int i = (M / Q) * 2 - 1; i >= 0; i--) {
        if (threadIdx.x == i) {
            printf("%s: [", str);
            for (int i = 0; i < Q+2; i++)
            {
                printf("%u", lhcs[1][i]);
                if (i < Q+2 - 1)
                    printf(", ");
            }
            printf("]\n");
        }
    }
    
    __syncthreads();
}



template<class S, uint32_t Q>
__device__ inline 
void from4Reg2ShmQ( S lhcs[2][Q+2], volatile S* Lsh, volatile S* Hsh, uint32_t n ) {
    //__syncthreads();

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
        S carry= lhcs[1][Q+1];
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

template<class S, class D>
__device__ inline
void computeIter64( uint32_t i, uint32_t j
                  , volatile S* Ash, volatile S* Bsh
                  , D& accum, uint32_t& carry
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

template<class S, class D, uint32_t Q>
__device__ inline
void combineQ( D accums[Q], uint32_t carrys[Q], S lhcs[Q+2] ) {
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

template<class S, class D, uint32_t Q>
__device__ inline
void convolutionQ( uint32_t k1, volatile S* Ash, volatile S* Bsh, S lhcs[Q+2]) {
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


template<class S, class D, uint32_t Q>
__device__ inline
void wrapperConvQ( volatile S* Ash0, volatile S* Bsh0, S lhcs[2][Q+2], uint32_t M ) {
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


    // if (threadIdx.x == 0) {
    //     printf("lhcs: %u \n", lhcs[1][0]);
    // }

    // printLhcs12<Q>("res", lhcs, Bsh, M);
    // __syncthreads();

    volatile typename Base::uint_t* Lsh = Ash;
    volatile typename Base::uint_t* Hsh = Bsh;

    // 3. publish the low parts normally, and the high and carry shifted by one.
    from4Reg2ShmQ<uint_t, Q>( lhcs, Lsh, Hsh, M );
    __syncthreads();
    // printShMem("Lsh", Lsh, M);
    // __syncthreads();
    // printShMem("Hsh", Hsh, M);
    // __syncthreads();

    // 4. load back to register and perform the addition of the carries.
    uint_t Lrg[2*Q];
    cpyShm2Reg<uint_t,2*Q>( Lsh, Lrg );
    uint_t Hrg[2*Q];
    cpyShm2Reg<uint_t,2*Q>( Hsh, Hrg );
    __syncthreads();

    // printRegs1<Q*2>("res", Lrg, Lsh, M);
    // __syncthreads();

    // printRegs1<Q*2>("res", Hrg, Lsh, M);
    // __syncthreads();

    baddRegs<uint_t, uint_t, carry_t, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, Rrg, M );

    // printRegs1<Q*2>("res:", Rrg, Lsh, M);
    // __syncthreads();


}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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

template<class S, class D>
__device__ inline
void computeIter641( uint32_t i, uint32_t j
                  , volatile S* Ash, volatile S* Bsh
                  , D& accum, uint32_t& carry
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

template<class S, class D, uint32_t Q>
__device__ inline
void combineQ1( D accums[Q], uint32_t carrys[Q], S lhcs[Q+2] ) {
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

template<class S, class D, uint32_t Q>
__device__ inline
void convolutionQ1( uint32_t k1, volatile S* Ash, volatile S* Bsh, S lhcs[Q+2], uint32_t M) {
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


template<class S, class D, uint32_t Q>
__device__ inline
void wrapperConvQ1( volatile S* Ash0, volatile S* Bsh0, S lhcs[2][Q+2], uint32_t M ) {
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

    // printLhcs1<Q*2>("res", lhcs, 1, Bsh, M*2);
    // __syncthreads();

    // printLhcs12<Q*2>("res", lhcs, Bsh, M*2);
    // __syncthreads();

    // if (threadIdx.x == 8) {
    //     printf("lhcs: %u \n", lhcs[0][0]);
    // }
    volatile typename Base::uint_t* Lsh = Bsh;

    uint_t Lrg[4*Q];
    uint_t Hrg[4*Q];

    from4Reg2ShmQ1<uint_t, Q*2>( lhcs, Lrg, Hrg, Lsh, Lsh, M*2 );
    __syncthreads();

    // printRegs1<Q*4>("Lrg", Lrg, Lsh, M*2);
    // __syncthreads();
    // printRegs1<Q*4>("Hrg", Hrg, Lsh, M*2);
    // __syncthreads();



    baddRegs<uint_t, uint_t, carry_t, 4*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, Rrg, M*2 );

    // printRegs1<Q*4>("res", Rrg, Lsh, M*2);
    // __syncthreads();

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// template<class S, uint32_t Q>
// __device__ inline 
// void from4Reg2ShmQ( S lhcs[Q+2], volatile S* Lsh, volatile S* Hsh, S highCarry[2], bool isFirst, uint32_t n ) {
//     //__syncthreads();

//     const uint32_t Q2 = 2*Q;
//     uint32_t tid_mod_m = threadIdx.x % (n/Q2);

//     {    
//         int32_t twoltid = isfirst ? Q*tid_mod_m : n - Q*tid_mod_m - Q;;
//         #pragma unroll
//         for(int q=0; q<Q; q++) {
//             Lsh[twoltid+q] = lhcs[q];
//         }
//         #pragma unroll
//         for(int q=2; q<Q; q++) {
//             Hsh[twoltid+q] = 0;
//         }
//         if( threadIdx.x != n/Q2 - 1 ) {
//             Hsh[twoltid+Q]   = lhcs[Q];
//             Hsh[twoltid+Q+1] = lhcs[Q+1];
//         }
//         else if (isFirst){
//             highCarry[0] = lhcs[Q];
//             highCarry[1] = lhcs[Q+1];
//         }
//     }
//     __syncthreads();
//     {
//         if( tid_mod_m == 0 ) {
//             S high  = 0;
//             S carry = 0;
//             if (!isFirst) {
//                 S high  = highCarry[0];
//                 S carry = highCarry[1];  
//             }
//             uint32_t ind  = 0;
//             Hsh[ind]   = high;
//             Hsh[ind+1] = carry;
//         }
//     }
// }


template<class S, uint32_t Q>
__device__ inline 
void from4Reg2ShmQ2( S lhcs[Q+2], volatile S* Lsh, volatile S* Hsh,  S highCarry[2], bool isFirst, uint32_t n ) {
    //__syncthreads();

    const uint32_t Q2 = 2*Q;
    uint32_t tid_mod_m = threadIdx.x % (n/Q2);

    {    
        // if (threadIdx.x == n/Q2 - 1) {
        //     printf("HERE: %u \n", lhcs[0]);
        // }
        int32_t twoltid = isFirst ? Q*tid_mod_m : n/2 - Q*tid_mod_m - Q;
      //  printf("NOWN %u \n", twoltid);
        #pragma unroll
        for(int q=0; q<Q; q++) {
          //  if (lhcs[q] ==867742551 ) printf("HERE1 \n");
            Lsh[twoltid+q] = lhcs[q];
        }
        __syncthreads();
        #pragma unroll
        for(int q=2; q<Q; q++) {
            Hsh[twoltid+q] = 0;
        }
        __syncthreads();
        if( threadIdx.x != n/Q2 - 1) {
         //   printf("NOWN %u \n", twoltid);
            // if (lhcs[Q] ==867742551 ) printf("HERE2 %u\n", twoltid+Q);
            // if (lhcs[Q+1] ==867742551 ) printf("HERE3 \n");
            Hsh[twoltid+Q]   = lhcs[Q];
            Hsh[twoltid+Q+1] = lhcs[Q+1];
          //  printf("NOWN %u \n", Lsh[0]);
        }
        __syncthreads();
        if (isFirst && threadIdx.x == n/Q2 - 1){
            highCarry[0] = lhcs[Q];
            highCarry[1] = lhcs[Q+1];
            Hsh[0] = 0;
            Hsh[1] = 0;
        }
        else if (threadIdx.x == n/Q2 - 1){
         //   printf("NOWN %u \n", twoltid+Q);
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
    __syncthreads();
}



template<class Base, uint32_t IPB, uint32_t Q>
__device__ 
void bmulRegsQComplete1( volatile typename Base::uint_t* Ash
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
    __syncthreads();
    cpyReg2Shm<uint_t,2*Q>( Brg, Bsh );
    __syncthreads();
  
    // 2. perform the convolution
    uint_t lhcs[2][2*Q+2];

    wrapperConvQ1<uint_t, ubig_t, 2*Q>( Ash, Bsh, lhcs, M );
    __syncthreads();

    // printLhcs1<Q*2>("res", lhcs, 1, Bsh, M*2);
    // __syncthreads();

    // printLhcs<Q*2>("res", lhcs, Bsh, M*2);
    // __syncthreads();
    volatile typename Base::uint_t* Lsh = Ash;
    volatile typename Base::uint_t* Hsh = Bsh;

    // 3. publish the low parts normally, and the high and carry shifted by one.
    uint_t highCarry[2];
    __syncthreads();
    from4Reg2ShmQ2<uint_t, Q*2>( lhcs[0], Lsh, Hsh, highCarry, true, M*2 );
    __syncthreads();

    // printf("High: %u \n", highCarry[0]);
    // printf("Carry: %u \n", highCarry[1]);
    // printShMem("Lsh", Lsh, M);
    // __syncthreads();
    // printShMem("Hsh", Hsh, M);
    // __syncthreads();

    // 4. load back to register and perform the addition of the carries.
    uint_t Lrg[2*Q];
    uint_t Hrg[2*Q];
    cpyShm2Reg<uint_t,2*Q>( Lsh, Lrg );
    __syncthreads();
    cpyShm2Reg<uint_t,2*Q>( Hsh, Hrg );
    __syncthreads();

    // printRegs1<Q*2>("res", Lrg, Lsh, M);
    // __syncthreads();

    // printRegs1<Q*2>("res", Hrg, Lsh, M);
    // __syncthreads();

    bool overflow = baddRegsOverflow<uint_t, uint_t, carry_t, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, (carry_t*)Hsh, Lrg, Hrg, Rrg, M );
    __syncthreads();

    // printRegs1<Q*2>("res", Rrg, Lsh, M);
    // __syncthreads();

    from4Reg2ShmQ2<uint_t, Q*2>( lhcs[1], Lsh, Hsh, highCarry, false, M*2 );
    __syncthreads();

    // if (threadIdx.x == M/(Q*2) - 1) {
    //     printf("High: %u \n", highCarry[0]);
    //     printf("Carry: %u \n", highCarry[1]);
    // }
    // printShMem("Lsh", Lsh, M);
    // __syncthreads();
    // printShMem("Hsh", Hsh, M);
    // __syncthreads();

   // printLhcs12<Q*2>("res", lhcs, Bsh, M*2);

    cpyShm2Reg<uint_t,2*Q>( Lsh, Lrg );
    __syncthreads();
    cpyShm2Reg<uint_t,2*Q>( Hsh, Hrg );
    __syncthreads();

    baddRegs<uint_t, uint_t, carry_t, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, &Rrg[Q*2], M );
    __syncthreads();
    //printf("BOOL: %d \n", overflow);
    if (overflow) {
        add1<Q*2>(&Rrg[Q*2], Lsh);
    }
    __syncthreads();

    // printRegs1<Q*2>("res", Rrg, Hsh, M*2);
    // __syncthreads();
    // printf("HERE \n");
}




