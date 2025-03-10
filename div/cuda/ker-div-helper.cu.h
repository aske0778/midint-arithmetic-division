#ifndef KERNEL_HELPER
#define KERNEL_HELPER

#include "../../cuda/helper.h"
#include <stdint.h>

#if 0

/**
 * Helper function for copying from global to shared memory
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const T& ne
                  , T* Ass
                  , volatile T* Ash
) {
    #pragma unroll
    for(uint32_t i=0; i<Q; i++) {
        uint32_t loc_ind = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        T elm = ne;
        if(glb_ind < M) { elm = Ass[glb_ind]; }
        Ash[loc_ind] = elm;
    }
    __syncthreads();
}

/**
 * Helper function for copying from global to shared to register memory
 * Is padded
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
copyFromGlb2Shr2RegMem( const uint32_t glb_offs
                      , const T& ne
                      , T* Ass
                      , volatile T* Ash
                      , T* Arg[Q]
) {
    #pragma unroll
    for(uint32_t i=0; i<Q; i++) {
        uint32_t idx = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + idx;
        T elm = ne;
        if (idx < M) {
            elm = Ass[glb_ind];
        }
        Ash[loc_ind] = elm;
    }
    __syncthreads();

    #pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
        Arg[i] = Ash[Q * threadIdx.x + i];
    }
}

/**
 * 
 */
template<class S, uint32_t M, uint32_t Q>
__device__ inline
void copyFromReg2Shr2GlbMem ( S* Ass
                            , volatile S* Ash
                            , S Arg[Q]
) { 
    // 1. write from regs to shared memory
    for(int i=0; i<Q; i++) {
        Ash[Q*threadIdx.x + i] = Arg[i];
    }
    __syncthreads();
    // 2. write from shmem to global
    for(int i=0; i<Q; i++) {
        uint32_t loc_pos = i*(M/Q) + threadIdx.x;
        //if(loc_pos < IPB*M) 
        {
            Ass[loc_pos] = Ash[loc_pos];
        }
    }
}


/**
 * Helper function for copying from shared to global memory
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , T* Ass
                  , volatile T* Ash
) {
    #pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
        uint32_t loc_ind = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < M) {
            T elm = const_cast<const T&>(Ash[loc_ind]);
            Ass[glb_ind] = elm;
        }
    }
    __syncthreads();
}

#else

/**
 * Helper kernel for copying from global to shared memory
 */
template <class T, uint32_t Q>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const uint32_t N
                  , const T &ne
                  , T *d_inp
                  , volatile T *shmem_inp
) {
#pragma unroll
    for (uint32_t i = 0; i < Q; i++)
    {
        uint32_t loc_ind = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        T elm = ne;
        if (glb_ind < N)
        {
            elm = d_inp[glb_ind];
        }
        shmem_inp[loc_ind] = elm;
    }
    __syncthreads();
}

/**
 * Helper kernel for copying from shared to global memory
 */
template <class T, uint32_t Q>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T *d_out
                  , volatile T *shmem_inp
) {
#pragma unroll
    for (uint32_t i = 0; i < Q; i++)
    {
        uint32_t loc_ind = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < N)
        {
            T elm = const_cast<const T &>(shmem_inp[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads();
}

#endif






/**
 * Returns 1 if u is zero or 0 otherwise
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline bool
ez4Reg( 
        T u[Q]
) {
    
    bool retval = true;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M && u[idx] != 0) {
            retval = false;
        }
    }
    __syncthreads();
    return retval;
}

/**
 * Sets retval in shared memory to 1 if u is zero or 0 otherwise
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
ez4Sh( volatile T* u,
       uint32_t* retval
) {
    
    *retval = 1;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M && u[idx] != 0) {
            *retval = 0;
        }
    }
    __syncthreads();
}

/**
 * Sets retval to 1 if u and v are equal or 0 otherwise
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
eq4Sh( volatile T* u
     , volatile T* v
     , uint32_t* retval
) {
    
    *retval = 1;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M && u[idx] != v[idx]) {
            *retval = 0;
        }
    }
    __syncthreads();
}

/**
 * Compares two bigints in register memory using shmem buffer
 * Shared memory is preserved
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline bool
eq4Reg( T u[Q]
      , T v[Q]
      , bool* shmem
) {
    bool retval;
    bool tmp = shmem[0];

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M && u[idx] != v[idx]) {
            shmem[0] = false;
        }
    }
    __syncthreads();

    retval = shmem[0];
    shmem[0] = tmp;
    return retval;
}

/**
 * Sets the first element to d and zeros the rest.
 * @remark Used for bigints in register memory.
 * 
 * @param u Register memory representation of bigint
 * @param d Value to set first element of biging
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
set4Reg( T* u[Q]
       , const T d
) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M) {
            u[idx] = 0;
        }
    }
    if (threadIdx.x == 0) {
        u[0] = d;
    }
    __syncthreads();
}

/**
 * Sets the first element to d and zeros the rest
 * @remark Used for bigints in shared memory.
 * 
 * @param u Bigint in shared memory
 * @param d Value to set first element of biging
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
set4Shm( volatile T* u
       , const T d
) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M) {
            u[idx] = 0;
        }
    }
    u[0] = d;
    __syncthreads();
}

/**
 * Zeros all elements of u
 * @remark Used for bigints in register memory.
 * 
 * @param u Bigint in register memory
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
zero4Reg( T u[Q]
) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M) {
            u[idx] = 0;
        }
    }
    __syncthreads();
}

/**
 * Zeros all elements of u
 * @remark Used for bigints in shared memory.
 * 
 * @param u Bigint in shared memory
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
zero4Shm( volatile T* u
) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M) {
            u[idx] = 0;
        }
    }
    __syncthreads();
}

#if 0

/**
 * Calculates the precisions of u
 * 
 * Uses two indexes of shared memory to store
 * intermediate results of reduce over max.
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline void
prec( volatile T* u
    , volatile T* buf
) {

    if (threadIdx.x == 0) {
        buf[0] = 0;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M && u[idx] != 0 && idx > buf[0]) {
            buf[0] = idx;
            // printf("%u\n", buf[0]);
            printf("%u\n", idx);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // printf("%u\n", buf[0]);
        buf[0] += 1;
    }
    __syncthreads();
}


#else

/**
 * Returns the precision of u to shared memory
 * 
 * @param u Bigint in register memory
 * @param p Buffer in shared memory (is preserved)
 */
template <class T, uint32_t M, uint32_t Q>
__device__ inline T
prec4Reg( T u[Q]
        , T* p
) {

    int highest_idx = -1;
    int old = p[0];
    p[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M && u[idx] != 0) {
            highest_idx = max(highest_idx, idx);
        }
    }
    atomicMax(p, highest_idx + 1);
    highest_idx = p[0];
    p[0] = old;
    return highest_idx;
}

/**
 * Returns the precision of u to shared memory
 */
template <class T, uint32_t M, uint32_t Q>
__device__ inline void
prec4Shm( volatile T* u
        , T* p
) {

    int highest_idx = -1;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < M && u[idx] != 0) {
            highest_idx = max(highest_idx, idx);
        }
    }
    atomicMax(p, highest_idx + 1);
}


/**
 * @brief bigint_t division
 * @note Uses long division algorithm
 * https://en.wikipedia.org/wiki/Division_algorithm#Long_division
 *
 * @param n numerator
 * @param d denominator
 * @param q quotient
 * @param r remainder
 * @param m Total size of bigint_ts
 */
 template<class T, class T2, uint32_t Q>
 __device__ inline void
 Blockwise_quo (volatile T* n, 
      T  d,
      volatile T *q,
      T  m,
      volatile T* buf ) {
    
    if (d == 0){
        printf("Division by zero\n");
        return;
    }
    
    uint64_t r = 0;


    for (int i = 0 ; i < Q; i++ ){
        int idx = i * blockDim.x + threadIdx.x;
        
        if (idx < m) {
            if (idx > 0) {
                r = n[idx - 1];
                r = (r << 32) + n[idx];
            } else {
                r = n[0];
            } 
            if (r >= d) {
                // need minus to because we utilize that r at each step in sequential 
                // is a 64 bit composed of the previus 32 bit as most significant, and current n as least 
                buf[idx-2] = r / d; 
                r = r % d;
            } 
        }
        __syncthreads();

    }
}

template<class T, class T2, uint32_t Q>
__device__ inline void 
blockwise_lt (
    volatile T *u,
    volatile T *v, 
    const uint32_t m,
    int64_t* retval,
    volatile T2 *buf) {
    

    int step_size = 1;
    int number_of_threads = blockDim.x;



    #pragma unroll
    for (int i = 0 ; i < Q; i++ ){
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m) {
            buf[idx] = u[idx]- v[idx];
        }

    }

    #pragma unroll
    for (int i = 0 ; i < Q; i++ ){
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m) {
            if (buf[idx + step_size] != 0 ) {
                buf[idx] = buf[idx + step_size];
        }
        step_size <<= 1; 
        }
    }
}


#endif






#endif // KERNEL_HELPER