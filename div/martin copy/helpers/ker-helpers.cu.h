#define WARP (32)
#define lgWARP (5)

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

template<class uint_t, uint32_t Q>
__device__ inline void 
cpyReg2Shm ( uint_t Rrg[Q]
           , volatile uint_t* shmem
) { 
    for(int i=0; i<Q; i++) {
        shmem[Q*threadIdx.x + i] = Rrg[i];
    }
}

template<class uint_t, uint32_t Q>
__device__ inline void 
cpyShm2Reg ( volatile uint_t* shmem
           , uint_t Rrg[Q]
) { 
    for(int i=0; i<Q; i++) {
        Rrg[i] = shmem[Q*threadIdx.x + i];
    }
}

template<class uint_t, uint32_t Q>
__device__ inline uint32_t 
prec( uint_t u[Q]
    , volatile uint_t* sh_mem
) {
    sh_mem[0] = 0;
    __syncthreads();   
    #pragma unroll
    for (int i = Q-1; i >= 0; i--) {
        if (u[i] != 0) {
            atomicMax((uint_t*)sh_mem, Q * threadIdx.x + i + 1);
            break;
        }
    }
    __syncthreads();
    return sh_mem[0];
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
            break;
        }
    }
    __syncthreads();   
    return sh_mem[0];
}

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

template<class uint_t, uint32_t Q>
__device__ inline void 
set( uint_t u[Q]
   , uint32_t d
   , uint32_t idx
) {
    if (threadIdx.x == idx / Q) {
        u[idx % Q] = d;
    }
}

template<class uint_t, uint32_t Q>
__device__ inline void 
zeroAndSet( uint_t u[Q]
          , uint32_t d
          , uint32_t idx
) {
    for (uint32_t i = 0; i < Q; ++i) {
        u[i] = 0;
    }
    set<uint_t, Q>(u, d, idx);
}

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
}

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
        }
        else {
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


template<typename Base, uint32_t Q>
__device__ inline void 
quo( uint32_t bpow
   , uint32_t d
   , typename Base::uint_t RReg[Q]
) {
    uint64_t r = 1;
    #pragma unroll
    for (int i = bpow - 1; i >= 0; i--)
    {
        r <<= Base::bits; 
        if (r >= d) {
            if (threadIdx.x == i / Q) {
                RReg[i % Q] = r / d;
            }
            r %= d;
        }
    }
}

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
    __syncthreads();   
    bool res = reduceBlock<LessThan>(RReg[Q-1], sh_mem) & 0b01;
    __syncthreads();
    return res;
}

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