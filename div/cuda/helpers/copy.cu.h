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

template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpGlb2Reg ( uint32_t ipb, volatile S* shmem, S* ass, S Arg[Q] ) {
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

template<class S, uint32_t IPB, uint32_t M, uint32_t Q>
__device__ inline
void cpReg2Glb ( uint32_t ipb, volatile S* shmem , S Rrg[Q], S* rss ) { 
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