
template<uint32_t Q>
__device__ inline void printLhcs(const char *str, uint32_t lhcs[2][Q+2], volatile uint32_t* sh_mem, uint32_t M)
{
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[Q * threadIdx.x + i] = lhcs[0][i];
    }
    __syncthreads();
    
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[M/2 - 1 + (Q * threadIdx.x + i)] = lhcs[1][i];
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
__device__ inline void printRegs1(const char *str, uint32_t u[Q], uint32_t* sh_mem, uint32_t M)
{
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[Q * threadIdx.x + i] = u[i];
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

template<class uint_t, uint32_t Q>
__device__ inline void
printRegs( const char *str
         , uint_t u[Q]
         , volatile uint_t* sh_mem
         , uint32_t M
) {
    #pragma unroll
    for (int i=0; i < Q; i++) {
        sh_mem[Q * threadIdx.x + i] = u[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        printf("%s: [", str);
        for (int i = 0; i < M; i++) {
            printf("%u", sh_mem[i]);
            if (i < M - 1)
                printf(", ");
        }
        printf("]\n");
    }
}



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



template<class uint_t, uint32_t M, uint32_t Q>
__device__ inline void shiftDouble(int n, uint_t u[Q*2], volatile uint_t* sh_mem, uint_t RReg[Q]) {
    #pragma unroll
    for (int i = 0; i < Q*2; i++) {
        int idx = Q*2 * threadIdx.x + i;
        int offset = idx + n;

        if (offset >= 0 && offset < M) {
            sh_mem[offset] = u[i];
        }
        else {
            sh_mem[M-idx-1] = 0;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        RReg[i] = sh_mem[Q * threadIdx.x + i];
    }
    __syncthreads();
}


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

bigDigit_t tmp1 = (b2l - V) / V + 1;
bigDigit_t tmp2 = ((double)(b2l - V*tmp1) / (double)V) * pow(2.0, bits);; //<< bits;
tmp = tmp2 + (tmp1 << bits);

// uint64_t V_low  = (uint64_t)(tmp);
// uint64_t V_high = (uint64_t)(tmp >> 64);

// printf("Full V (uint128): high = %llu, low = %llu\n", (unsigned long long)V_high, (unsigned long long)V_low);
