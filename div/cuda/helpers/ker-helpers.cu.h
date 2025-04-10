#define WARP (32)
#define lgWARP (5)

/**
 * Our own implementation for copying coalesced
 * from global to shared to register memory
 */
template<class uint_t, uint32_t M, uint32_t Q>
__device__ inline void 
cpyGlb2Sh2Reg( uint_t* AGlb
             , volatile uint_t* ASh
             , volatile uint_t AReg[Q]
) {
    const int glb_offs = blockIdx.x * M;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        ASh[idx] = AGlb[idx+glb_offs];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        AReg[i] = ASh[Q * threadIdx.x + i];
    }
}

/**
 * Our own implementation for copying coalesced
 * from register to shared to global memory
 */
template<class uint_t, uint32_t M, uint32_t Q>
__device__ inline void 
cpyReg2Sh2Glb( uint_t* AGlb
             , volatile uint_t* ASh
             , volatile uint_t AReg[Q]
) {
    const int glb_offs = blockIdx.x * M;

    #pragma unroll
    for (int i=0; i < Q; i++) {
        ASh[Q * threadIdx.x + i] = AReg[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        AGlb[idx+glb_offs] = ASh[idx];
    }
}

/**
 * Library implementation for copying coalesced
 * from global to shared to register memory
 */
template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpGlb2Reg ( uint32_t ipb
               , volatile S* shmem
               , S* ass
               , S Arg[Q]
) {
    const uint32_t M_lft = LIFT_LEN(M, Q); 
    // 1. read from global to shared memory
    const uint64_t glb_offs = blockIdx.x * (IPB * M);
    
    for(int i=0; i<Q; i++) {
        uint32_t loc_pos_sh = i*(IPB*M_lft/Q) + threadIdx.x;
        uint32_t r = loc_pos_sh % M_lft;

        uint32_t loc_pos_glb= (loc_pos_sh / M_lft) * M + r;
        S el = 0;
        if( (r < M) && (loc_pos_sh / M_lft < ipb) ) {
            el = ass[glb_offs + loc_pos_glb];
        }
        shmem[loc_pos_sh] = el;       
    }
    __syncthreads();
    // 2. read from shmem to regs
    for(int i=0; i<Q; i++) {
        Arg[i] = shmem[Q*threadIdx.x + i];
    }
}

/**
 * Library implementation for copying coalesced
 * from register to shared to global memory
 */
template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpReg2Glb ( uint32_t ipb
               , volatile S* shmem
               , S Rrg[Q]
               , S* rss
) { 
    const uint32_t M_lft = LIFT_LEN(M, Q);
    
    // 1. write from regs to shared memory
    uint32_t ind_ipb = threadIdx.x / (M_lft/Q);
    for(int i=0; i<Q; i++) {
        uint32_t r = (Q*threadIdx.x + i) % M_lft;
        uint32_t loc_ind = ind_ipb*M + r;
        if(r < M)
            shmem[loc_ind] = Rrg[i];
    }

    __syncthreads();

    // 2. write from shmem to global
    const uint64_t glb_offs = blockIdx.x * (IPB * M);
    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*(IPB*M_lft/Q) + threadIdx.x;
        if(loc_pos < ipb * M) {
            rss[glb_offs + loc_pos] = shmem[loc_pos];
        }
    }
}

/**
 * Copy from register to shared memory
 */
template<class uint_t, uint32_t Q>
__device__ inline void 
cpyReg2Shm ( uint_t Rrg[Q]
           , volatile uint_t* shmem
) { 
    for(int i=0; i<Q; i++) {
        shmem[Q*threadIdx.x + i] = Rrg[i];
    }
}

/**
 * Copy from shared to register memory
 */
template<class uint_t, uint32_t Q>
__device__ inline void 
cpyShm2Reg ( volatile uint_t* shmem
           , uint_t Rrg[Q]
) { 
    for(int i=0; i<Q; i++) {
        Rrg[i] = shmem[Q*threadIdx.x + i];
    }
}

/**
 * Calculate the precision of a bigint in register memory
 */
template<class uint_t, uint32_t Q>
__device__ inline uint32_t 
prec( uint_t u[Q]
    , volatile uint32_t* sh_mem
) {
    sh_mem[0] = 0;
    __syncthreads();   

    #pragma unroll
    for (int i = Q-1; i >= 0; i--) {
        if (u[i] != 0) {
            atomicMax((uint32_t*)sh_mem, Q * threadIdx.x + i + 1);
        }
    }
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
    sh_mem[0] = true;
    __syncthreads(); 
      
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != (bpow == (i * blockDim.x + threadIdx.x))) {
            sh_mem[0] = false;
        }
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
    sh_mem[0] = true;
    __syncthreads();   
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != 0) {
            sh_mem[0] = false;
        }
    }
    __syncthreads();   
    return sh_mem[0];
}

/**
 * 
 */
template<class uint_t, uint32_t Q>
__device__ inline bool 
ez( uint_t u[Q]
  , uint32_t idx
  , volatile uint_t* sh_mem
) {
    sh_mem[0] = (threadIdx.x == idx / Q && u[idx % Q] == 0);
    __syncthreads();
    return sh_mem[0];
}

/**
 * Sets a specific index of a bigint to value d
 * in register memory
 */
template<class uint_t, uint32_t Q>
__device__ inline void 
set( uint_t u[Q]
   , uint_t d
   , uint32_t idx
) {
    if (threadIdx.x == idx / Q) {
        u[idx % Q] = d;
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
    for (uint32_t i = 0; i < Q; ++i) {
        u[i] = 0;
    }
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

        // bool condition = (offset >= 0 && offset < M);
        // sh_mem[offset * condition + (M-idx-1) * (1-condition)] = u[i] * condition + 0 * (1-condition);

        if (offset >= 0 && offset < M) {
            sh_mem[offset] = u[i];
        } else {
            sh_mem[M-idx-1] = 0;
        }
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

        if (offset >= 0 && offset < M) {
            sh_mem[offset] = u[i];
        } else {
            sh_mem[M-idx-1] = 0;
        }
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
 * Quotient calculation of a bpow and divisor d
 */
template<typename Base, uint32_t Q>
__device__ inline void 
quo( uint32_t bpow
   , typename Base::uint_t d
   , typename Base::uint_t RReg[Q]
) {
    typename Base::ubig_t r = 1;

    #pragma unroll
    for (int i = bpow - 1; i >= 0; i--) {
        r <<= Base::bits; 
        if (r >= d) {
            if (threadIdx.x == i / Q) {
                RReg[i % Q] = r / d;
            }
            r %= d;
        }
    }
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
            if (i < M - 1)
                printf(", ");
        }
        printf("]\n");
    }
}