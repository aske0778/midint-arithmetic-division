/**
 * Coalesced copy from global to shared to register memory
 */
template<class uint_t, uint32_t M, uint32_t Q>
__device__ inline void 
cpyGlb2Sh2Reg( uint_t* AGlb
             , volatile uint_t* ASh
             , uint_t AReg[Q]
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
 * Coalesced copy from register to shared to global memory
 */
template<class uint_t, uint32_t M, uint32_t Q>
__device__ inline void 
cpyReg2Sh2Glb( uint_t* AGlb
             , volatile uint_t* ASh
             , uint_t AReg[Q]
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
 * Copy from register to shared memory
 */
template<class uint_t, uint32_t Q>
__device__ inline void 
cpyReg2Shm ( uint_t Rrg[Q]
           , volatile uint_t* shmem 
           , uint_t M
) { 
    if (threadIdx.x < (M+Q-1)/Q) {
        #pragma unroll
        for(int i=0; i<Q; i++) {
            shmem[Q*threadIdx.x + i] = Rrg[i];
        }
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
    #pragma unroll
    for(int i=0; i<Q; i++) {
        Rrg[i] = shmem[Q*threadIdx.x + i];
    }
}

/**
 * Copy from shared to register memory
 */
template<class uint_t, uint32_t Q>
__device__ inline void 
cpyShm2Reg ( volatile uint_t* shmem
           , uint_t Rrg[Q]
           , uint_t M
) { 
    if (threadIdx.x < (M+Q-1)/Q) {
        #pragma unroll
        for(int i=0; i<Q; i++) {
            int idx = Q*threadIdx.x + i;
            if (idx < M) {
                Rrg[i] = shmem[idx];
            }
        }
    }
}

/**
 * Copy from one register to another
 */
template<class uint_t, uint32_t Q>
__device__ inline void 
cpyReg2Reg( uint_t AReg[Q]
          , uint_t BReg[Q]
) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        BReg[i] = AReg[i];
    }
}
