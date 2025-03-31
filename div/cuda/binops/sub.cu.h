template<class D, class S, class CT, uint32_t q, D HIGHEST>
__device__ void bsubRegs( volatile CT* Csh
                         , D Arg[q]
                         , S Brg[q]
                         , D rs[q]
                         , uint32_t m
                         ) {
    //D  rs[q];
    CT cs[q];
    
    // 1. map: subtract the digits pairwise, build the 
    //         partial results and the carries, and
    //         print carries to shmem
    {
        CT accum = CarrySegBop<CT>::identity();
        for(int i=0; i<q; i++) {
            uint32_t ind = threadIdx.x * q + i;
            D a = Arg[i];
            S b = Brg[i];
            CT c;
            
            rs[i] = a - (D)b;
            c = (CT) ( (rs[i] > a) );
            c = c | ((rs[i] == HIGHEST) << 1);
            //c = c | ( ((ind % m) == 0) << 2 );
            if( (ind % m) == 0 )
                c = c | 4;
            
            accum = CarrySegBop<CT>::apply(accum, c);
            cs[i] = c;
        }
        Csh[threadIdx.x] = accum;
    }
    
    __syncthreads();
   
    // 2. scan the carries
    scanIncBlock< CarrySegBop<CT> >(Csh, threadIdx.x);
        
    // 3. compute the final result by subtracting the carry from the previous element
    {
        CT carry = CarrySegBop<CT>::identity();
        if(threadIdx.x > 0) {
            carry = Csh[threadIdx.x - 1];
        }
        //CT carry = prefix;
        for(int i=0; i<q; i++) {
            // uint32_t c = ( (carry & 1) == 1 );
            if( (cs[i] & 4) == 0 )
                rs[i] -= (carry & 1);
            carry = CarrySegBop<CT>::apply(carry, cs[i]);         
        }
    }
    __syncthreads();
}

template<typename Base, uint32_t Q>
__device__ inline void
sub( uint32_t bpow
   , typename Base::uint_t u[Q]
   , volatile typename Base::uint_t* sh_mem
) {
    using uint_t = typename Base::uint_t;
    sh_mem[0] = Base::HIGHEST;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != Base::HIGHEST) {
            atomicMin((uint_t*)sh_mem, Q * threadIdx.x + i);
            break;
        }
    }
    __syncthreads();

    uint32_t min_index = sh_mem[0];
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        uint32_t idx = Q * threadIdx.x + i;
        if (idx < min_index) {
            u[i] = 0;
        }
        else if (idx < bpow) {
            u[i] = ~u[i] + (idx == min_index);
        }
    }
    __syncthreads();
}
