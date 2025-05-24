
/**
 */
template<class D, class S, class CT, uint32_t Q, D HIGHEST>
__device__ inline void
baddRegs( volatile CT* Csh
        , D Arg[Q]
        , S Brg[Q]
        , D rs[Q]
) {
    CT cs[Q];
    {
        CT accum = CarrySegBop<CT>::identity();
        for(int i=0; i<Q; i++) {
            D a = Arg[i];
            S b = Brg[i];
            CT c;
            
            rs[i] = a + (D)b;
            c = (CT) ( (rs[i] < a) );
            c = c | ((rs[i] == HIGHEST) << 1);
            
            accum = CarrySegBop<CT>::apply(accum, c);
            cs[i] = c;
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
        for(int i=0; i<Q; i++) {
            rs[i] += (carry & 1);
            carry = CarrySegBop<CT>::apply(carry, cs[i]);         
        }
    }
}

/**
 * 
 */
template<class D, class S, class CT, uint32_t Q, D HIGHEST>
__device__ inline bool
baddRegsOverflow( volatile CT* Csh
                , volatile CT* Dsh
                , D Arg[Q]
                , S Brg[Q]
                , D rs[Q]
                , uint32_t m
) {
    Dsh[0] = false;
    __syncthreads();
    CT cs[Q];
    
    {
        CT accum = CarrySegBop<CT>::identity();
        for(int i=0; i<Q; i++) {
            uint32_t ind = threadIdx.x * Q + i;
            D a = Arg[i];
            S b = Brg[i];
            CT c;

            if( (ind % m) == m-1 && a + b < a) Dsh[0] = true;

            rs[i] = a + (D)b;
            c = (CT) ( (rs[i] < a) );
            c = c | ((rs[i] == HIGHEST) << 1);

            accum = CarrySegBop<CT>::apply(accum, c);
            cs[i] = c;
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
        for(int i=0; i<Q; i++) {
            if( (threadIdx.x * Q + i == m-1) && (rs[i] == HIGHEST) && (carry & 1) ) {
                Dsh[0] = true;
            }
            rs[i] += (carry & 1);
            carry = CarrySegBop<CT>::apply(carry, cs[i]);         
        }
    }
    __syncthreads();
    return Dsh[0];
}


/**
 */
template<class D, class S, class CT, D HIGHEST>
__device__ inline D
baddRegsNaive( volatile CT* Csh
        , D Arg
        , S Brg
) {
    D rs;
    {
        D a = Arg;
        S b = Brg;
        CT c;
        
        rs = a + (D)b;

        c = (CT) ( (rs < a) );
        c = c | ((rs == HIGHEST) << 1);

        Csh[threadIdx.x] = c;
    }
    __syncthreads();
   
    scanIncBlock< CarrySegBop<CT> >(Csh, threadIdx.x);

    {
        CT carry = CarrySegBop<CT>::identity();
        if(threadIdx.x > 0) {
            carry = Csh[threadIdx.x - 1];
        }
        rs += (carry & 1);
    }
    return rs;
}

template<class D, class S, class CT, D HIGHEST>
__device__ inline void
baddRegsNaive2x( volatile CT* Csh
        , D Arg[2]
        , S Brg[2]
        , D rs[2]
) {
    CT cs[2];
    {
        CT accum = CarrySegBop<CT>::identity();
        for(int i=0; i<2; i++) {
            D a = Arg[i];
            S b = Brg[i];
            CT c;
            
            rs[i] = a + (D)b;
            c = (CT) ( (rs[i] < a) );
            c = c | ((rs[i] == HIGHEST) << 1);
            
            accum = CarrySegBop<CT>::apply(accum, c);
            cs[i] = c;
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
        for(int i=0; i<2; i++) {
            rs[i] += (carry & 1);
            carry = CarrySegBop<CT>::apply(carry, cs[i]);         
        }
    }
}


template<class Base, uint32_t Q>
__device__ inline void 
add1( typename Base::uint_t u[Q]
    , volatile typename Base::uint_t* sh_mem
) {
    using uint_t = typename Base::uint_t;

    uint32_t tmp = UINT32_MAX;
    sh_mem[0] = Base::HIGHEST;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int rev_i = Q - i - 1;
        if (u[rev_i] != Base::HIGHEST) {
            tmp = Q * threadIdx.x + rev_i;
        }
    }
    atomicMin((uint32_t*)sh_mem, tmp);
    __syncthreads();

    uint32_t min_index = sh_mem[0];

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        uint32_t idx = Q * threadIdx.x + i;
        if (idx < min_index) {
            u[i] = 0;
        }
        else if (idx == min_index) {
            u[i] += 1;
        }
    }
}
