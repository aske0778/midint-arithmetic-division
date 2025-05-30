#define WARP (32)
#define lgWARP (5)

/**
 * Calculate the precision of a bigint in register memory
 */
template<class uint_t, uint32_t Q>
__device__ inline uint32_t 
prec( uint_t u[Q]
    , volatile uint32_t* sh_mem
) {
    uint32_t tmp = 0;
    sh_mem[0] = 0;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != 0) {
            tmp = Q * threadIdx.x + i + 1;
        }
    }
    atomicMax((uint32_t*)sh_mem, tmp);
    __syncthreads();
    
    return sh_mem[0];
}

/**
 * Tests if a bigint is equal to a bpow
 */
template<class uint_t, uint32_t Q>
__device__ inline bool 
eq( uint_t u[Q]
  , uint32_t bpow
  , volatile uint_t* sh_mem
) {
    bool tmp = true;
    sh_mem[0] = true;
    __syncthreads(); 
      
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != (bpow == (Q * threadIdx.x + i))) {
            tmp = false;
        }
    }
    if (tmp == false) {
        sh_mem[0] = false;
    }
    __syncthreads();    
    return sh_mem[0];
}

template<class uint_t, uint32_t Q>
__device__ inline bool 
ezShift( uint_t u[Q]
  , int ind
  , volatile uint_t* sh_mem
) {
    bool tmp = true;
    sh_mem[0] = true;
    __syncthreads();   
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != 0 && Q * threadIdx.x + i < ind) {
            tmp = false;
        }
    }
    if (tmp == false) {
        sh_mem[0] = false;
    }
    __syncthreads();   
    return sh_mem[0];
}



/**
 * Checks if a bigint is zero
 */
template<class uint_t, uint32_t Q>
__device__ inline bool 
ez( uint_t u[Q]
  , volatile uint_t* sh_mem
) {
    bool tmp = true;
    sh_mem[0] = true;
    __syncthreads();   
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != 0) {
            tmp = false;
        }
    }
    if (tmp == false) {
        sh_mem[0] = false;
    }
    __syncthreads();   
    return sh_mem[0];
}

/**
 * 
 */
// template<class uint_t, uint32_t Q>
// __device__ inline bool 
// ez( uint_t u[Q]
//   , uint32_t idx
//   , volatile uint_t* sh_mem
// ) {
//     sh_mem[0] = false;
//     __syncthreads();
//     if (threadIdx.x == idx / Q && u[idx % Q] == 0)
//         sh_mem[0] = true;
//     __syncthreads();
//     return sh_mem[0];
// }
template<class uint_t, uint32_t Q>
__device__ inline bool 
ez( uint_t u[Q]
  , uint32_t idx
  , volatile uint_t* sh_mem
) {
    sh_mem[0] = false;
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        if (threadIdx.x == idx / Q && i == idx % Q ) {
            sh_mem[0] = u[i] == 0;
        }
    }
    __syncthreads();

    return sh_mem[0];
}

/**
 * Sets a specific index of a bigint to value d
 * in register memory
 */
// template<class uint_t, uint32_t Q>
// __device__ inline void 
// set( uint_t u[Q]
//    , uint_t d
//    , uint32_t idx
// ) {
//     if (threadIdx.x == idx / Q) {
//         u[idx % Q] = d;
//     }
// }
template<class uint_t, uint32_t Q>
__device__ inline void 
set( uint_t u[Q]
   , uint_t d
   , uint32_t idx
) {
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        if (threadIdx.x == idx / Q && i == idx % Q) {
            u[i] = d;
        }
    }
}

/**
 * Zeros a bigint
 */
template<class uint_t, uint32_t Q>
__device__ inline void 
zeroReg(uint_t u[Q]) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        u[i] = 0;
    }
}

/**
 * Zeros a bigint and sets index idx to value d
 */
template<class uint_t, uint32_t Q>
__device__ inline void 
zeroAndSet( uint_t u[Q]
          , uint_t d
          , uint32_t idx
) {
    zeroReg<uint_t, Q>(u);
    set<uint_t, Q>(u, d, idx);
}

/**
 * Performs the shift operation on a bigint
 */
template<class uint_t, uint32_t M, uint32_t Q>
__device__ inline void 
shift( int n
     , uint_t u[Q]
     , volatile uint_t* sh_mem
     , uint_t RReg[Q]
) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        int offset = idx + n;
        uint_t val = 0;
        if (offset >= 0 && offset < M) {
            val = u[i];
        } else {
            offset = M-idx-1;
        }
        sh_mem[offset] = val;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        RReg[i] = sh_mem[Q * threadIdx.x + i];
    }
}

/**
 * Performs the shift operation on a bigint of size 2M
 */
template<class uint_t, uint32_t M, uint32_t Q>
__device__ inline void 
shiftDouble( int n
           , uint_t u[Q*2]
           , volatile uint_t* sh_mem
           , uint_t RReg[Q]
) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        int offset = idx + n;
        uint_t val = 0;

        if (offset >= 0 && offset < M) {
            val = u[i];
        } else {
            offset = M-idx-1;
        }
        sh_mem[offset] = val;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = M + Q * threadIdx.x + i;
        int offset = idx + n;

        if (offset >= 0 && offset < M) {
            sh_mem[offset] = u[Q+i];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        RReg[i] = sh_mem[Q * threadIdx.x + i];
    }
}


/**
 * Sequential quotient calculation of a bpow and divisor d
 */
// template<typename Base, uint32_t Q>
// __device__ inline void 
// quo( uint32_t bpow
//    , typename Base::uint_t d
//    , typename Base::uint_t RReg[Q]
// ) {
//     if (d==1) {
//         if (threadIdx.x == bpow / Q) {
//             RReg[bpow % Q] = 1;
//         }
//         return;
//     }

//     typename Base::ubig_t r = 1;

//     for (int i = bpow - 1; i >= 0; i--) {
//         r <<= Base::bits; 
//         if (r >= d) {
//             if (threadIdx.x == i / Q) {
//                 RReg[i % Q] = r / d;
//             }
//             r %= d;
//         }
//     }
// }
template<typename Base, uint32_t Q>
__device__ inline void 
quo( uint32_t bpow
   , typename Base::uint_t d
   , volatile typename Base::uint_t* sh_mem
   , typename Base::uint_t RReg[Q]
) {
    if (d == 1) {
        set<Base::uint_t, Q>(RReg, 1, bpow);
        return;
    }
    if (threadIdx.x == 0) {
        sh_mem[0] = 0;
        typename Base::ubig_t r = 1;
        for (int i = bpow - 1; i >= 0; i--) {
            r <<= Base::bits; 
            if (r >= d) {
                sh_mem[i] = r / d;
                r %= d;
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for(int i=0; i<Q; i++) {
        if (i < bpow) {
            RReg[i] = sh_mem[Q*threadIdx.x + i];
        }
    }
}

__device__ inline uint128_t divide_u256_by_u128(uint128_t high, uint128_t low, uint128_t divisor) {
    uint128_t quotient = 0;
    uint128_t rem = 0;
    
    bool overflow = false;
    for (int i = 192; i >= 0; i--) {
        if (rem & (__uint128_t)1 << 127) {
            overflow = true;
        }
        rem <<= 1;

        if (i == 192) {
            rem |= 1;
        } 

        quotient <<= 1;

        if (rem >= divisor || overflow) {
            rem -= divisor;
            quotient |= 1;
            overflow = false;
        }
    }
    return quotient;
}


template<class uint_t, uint32_t Q>
__device__ inline bool 
lt( uint_t u[Q]
  , uint32_t bpow
  , volatile uint32_t* sh_mem
) {
    uint32_t tmp = 0;
    sh_mem[0] = 0;
    __syncthreads();   

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != 0) {
            tmp = Q * threadIdx.x + i;
        }
    }
    atomicMax((uint32_t*)sh_mem, tmp);
    __syncthreads();
    
    return sh_mem[0] < bpow;
}

/**
 * Warp-level implementation of less than operation
 * between two bigints stored in register memory
 */
template<class uint_t, uint32_t Q>
__device__ inline bool 
lt( uint_t u[Q]
  , uint_t v[Q]
  , volatile uint_t* sh_mem
) {
    int RReg[Q] = {0};

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] < v[i]) {
            RReg[i] |= 0b01;
        }
        else if (u[i] == v[i]) {
            RReg[i] |= 0b10;
        }
        if (i != 0) {
            RReg[i] |= (RReg[i] & 0b10) && (RReg[i-1] & 0b01);
            RReg[i] = (RReg[i] & ~0b10) | (RReg[i] & RReg[i-1] & 0b10);
        }
    }
    return reduceBlock<LessThan, uint_t>(RReg[Q-1], sh_mem) & 0b01;
}

/**
 * Prints contents of register memory to stdout
 */
template<class uint_t, uint32_t M, uint32_t Q>
__device__ inline void
printRegs( const char *str
         , uint_t u[Q]
         , volatile uint_t* sh_mem
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
            // printf("%" PRIu64, sh_mem[i]);
            if (i < M - 1)
                printf(", ");
        }
        printf("]\n");
    }
}

