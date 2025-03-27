template<class S, uint32_t Q>
__device__ inline 
void from4Reg2ShmQ( S lhcs[2][Q+2], S* Lsh, S* Hsh, uint32_t n ) {
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
    //__syncthreads();
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
                  , S* Ash, S* Bsh
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
void convolutionQ( uint32_t k1, S* Ash, S* Bsh, S lhcs[Q+2]) {
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
void wrapperConvQ( S* Ash0, S* Bsh0, S lhcs[2][Q+2], uint32_t M ) {
    const uint32_t offset = ( threadIdx.x / (M/(2*Q)) ) * M;
    S* Ash = Ash0 + offset;
    S* Bsh = Bsh0 + offset;
    
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
void bmulRegsQ( typename Base::uint_t* Ash
              , typename Base::uint_t* Bsh
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

    typename Base::uint_t* Lsh = Ash;
    typename Base::uint_t* Hsh = Bsh;

    // 3. publish the low parts normally, and the high and carry shifted by one.
    from4Reg2ShmQ<uint_t, Q>( lhcs, Lsh, Hsh, M );
    __syncthreads();

    // 4. load back to register and perform the addition of the carries.
    uint_t Lrg[2*Q];
    cpyShm2Reg<uint_t,2*Q>( Lsh, Lrg );
    uint_t Hrg[2*Q];
    cpyShm2Reg<uint_t,2*Q>( Hsh, Hrg );
    __syncthreads();

    baddRegs<uint_t, uint_t, carry_t, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, Rrg, M );
}







////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




// template<class S, uint32_t Q>
// __device__ inline 
// void from4Reg2ShmQ1( S lhcs[2][Q+2], S* Lsh, S* Hsh, uint32_t n ) {
//     //__syncthreads();

//     const uint32_t Q2 = 2*Q;
//     const uint32_t offset = ( threadIdx.x / (n/Q2) ) * n;
//     uint32_t tid_mod_m = threadIdx.x % (n/Q2);

//     {    
//         int32_t twoltid = offset + Q*tid_mod_m;
//         #pragma unroll
//         for(int q=0; q<Q; q++) {
//             Lsh[twoltid+q] = lhcs[0][q];
//         }
//         #pragma unroll
//         for(int q=2; q<Q; q++) {
//             Hsh[twoltid+q] = 0;
//         }
//         Hsh[twoltid+Q]   = lhcs[0][Q];
//         Hsh[twoltid+Q+1] = lhcs[0][Q+1];
//     }
//     //__syncthreads();
//     {
//         int32_t n_m_2ltid = offset + n - Q*tid_mod_m - Q;
//         #pragma unroll
//         for(int q=0; q<Q; q++) {
//             Lsh[n_m_2ltid + q] = lhcs[1][q];
//         }
//         #pragma unroll
//         for(int q=2; q<Q; q++) {
//             Hsh[n_m_2ltid + q] = 0;
//         }
//         S high = lhcs[1][Q];
//         S carry= lhcs[1][Q+1];
//         uint32_t ind = n_m_2ltid + Q;
//         if( tid_mod_m == 0 ) {
//             high  = 0;
//             carry = 0;
//             ind   = offset;
//         }
//         Hsh[ind]   = high;
//         Hsh[ind+1] = carry;
//     }
// }

// template<class S, class D>
// __device__ inline
// void computeIter641( uint32_t i, uint32_t j
//                   , S* Ash, S* Bsh
//                   , D& accum, uint32_t& carry
// ) {
//     const uint32_t SHFT = 8*sizeof(S);
//     S ai = Ash[i];
//     S bj = Bsh[j];
//     D ck = ((D)ai) * ((D)bj);

//     S accum_prev = (S) (accum>>SHFT);
//     accum += ck;
//     carry += (  ((S)(accum>>SHFT)) < accum_prev );
//     //if (accum < ck) carry++;
// }

// template<class S, class D, uint32_t Q>
// __device__ inline
// void combineQ1( D accums[Q], uint32_t carrys[Q], S lhcs[Q+2] ) {
//     uint32_t SHFT = 8 * sizeof(S);
    
//     lhcs[0] = (S) accums[0];
//     S h_res = (S) (accums[0] >> SHFT);
//     S c_res = carrys[0];

//     #pragma unroll
//     for(int q=1; q<Q; q++) {
//         S l = (S) accums[q];
//         S h = (S) (accums[q] >> SHFT);
//         lhcs[q] = l + h_res;
//         h_res = h + (c_res + (lhcs[q] < l));
//         c_res = carrys[q] + (h_res < h);
//     }
//     lhcs[Q]   = h_res;
//     lhcs[Q+1] = c_res;
// }

// template<class S, class D, uint32_t Q>
// __device__ inline
// void convolutionQ1( uint32_t k1, S* Ash, S* Bsh, S lhcs[Q+2]) {
//     D        accums[Q]; 
//     uint32_t carrys[Q];
    
//     #pragma unroll
//     for(int q=0; q<Q; q++) { 
//         accums[q] = 0; 
//         carrys[q] = 0; 
//     }

//     for(int kk = 0; kk <= k1; kk++) {
//         uint32_t i = kk;
//         uint32_t j = k1 - i;

//         #pragma unroll
//         for(int q=0; q<Q; q++) {
//             computeIter641<S,D>( i, j+q,   Ash, Bsh, accums[q], carrys[q] );
//         }
//     }

//     #pragma unroll
//     for(int q=1; q<Q; q++) {
//         #pragma unroll
//         for(int i=0; i<Q-q; i++) {
//             computeIter641<S,D>(k1+q, i, Ash, Bsh, accums[i+q], carrys[i+q]);
//         }
//     }
//     combineQ1<S,D,Q>(accums, carrys, lhcs);
// }


// template<class S, class D, uint32_t Q>
// __device__ inline
// void wrapperConvQ1( S* Ash0, S* Bsh0, S lhcs[2][Q+2], uint32_t M ) {
//     const uint32_t offset = ( threadIdx.x / (M/(2*Q)) ) * M;
//     S* Ash = Ash0 + offset;
//     S* Bsh = Bsh0 + offset;
    
//     uint32_t ltid = threadIdx.x % (M/(2*Q));
//     { // first half
//         uint32_t k1 = Q*ltid;
//         convolutionQ1<S,D,Q>(k1, Ash, Bsh, lhcs[0]);
//     }

//     { // second half
//         uint32_t k2 = M - Q*ltid - Q;
//         convolutionQ1<S,D,Q>(k2, Ash, Bsh, lhcs[1]);
//     }
// }


// template<class Base, uint32_t IPB, uint32_t Q>
// __device__ 
// void bmulRegsQ1( typename Base::uint_t* Ash
//               , typename Base::uint_t* Bsh
//               , typename Base::uint_t Arg[2*Q]
//               , typename Base::uint_t Brg[2*Q]
//               , typename Base::uint_t Rrg[2*Q]
//               , uint32_t M
//               ) 
// {
//     using uint_t = typename Base::uint_t;
//     using ubig_t = typename Base::ubig_t;
//     using carry_t= typename Base::carry_t;
    
//     // 1. copy from global to shared to register memory
//     cpyReg2Shm<uint_t,2*Q>( Arg, Ash );
//     cpyReg2Shm<uint_t,2*Q>( Brg, Bsh );
//     __syncthreads();
  
//     // 2. perform the convolution
//     uint_t lhcs[2][2*Q+2];
//     wrapperConvQ1<uint_t, ubig_t, Q*2>( Ash, Bsh, lhcs, M*2 );
//     __syncthreads();

//     typename Base::uint_t* Lsh = Ash;
//     typename Base::uint_t* Hsh = Bsh;

//     // 3. publish the low parts normally, and the high and carry shifted by one.
//     from4Reg2ShmQ1<uint_t, Q*2>( lhcs, Lsh, Hsh, M*2 );
//     __syncthreads();

//     // 4. load back to register and perform the addition of the carries.
//     uint_t Lrg[2*Q];
//     cpyShm2Reg<uint_t,2*Q>( Lsh, Lrg );
//     uint_t Hrg[2*Q];
//     cpyShm2Reg<uint_t,2*Q>( Hsh, Hrg );
//     __syncthreads();

//     baddRegs<uint_t, uint_t, carry_t, 2*Q, Base::HIGHEST>( (carry_t*)Lsh, Lrg, Hrg, Rrg, M );
// }