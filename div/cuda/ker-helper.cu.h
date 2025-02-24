#ifndef KERNEL_HELPER
#define KERNEL_HELPER

#include "../../cuda/helper.h"

#if 0

/**
 * Helper function for copying from global to shared memory
 */
template<class T, uint32_t Q>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const uint32_t N
                  , const T& ne
                  , T* d_inp
                  , volatile T* shmem_inp
) {
    #pragma unroll
    for(uint32_t i=0; i<Q; i++) {
        uint32_t loc_ind = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        T elm = ne;
        if(glb_ind < N) { elm = d_inp[glb_ind]; }
        shmem_inp[loc_ind] = elm;
    }
    __syncthreads();
}

/**
 * Helper function for copying from shared to global memory
 */
template<class T, uint32_t Q>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T* d_out
                  , volatile T* shmem_inp
) {
    #pragma unroll
    for (uint32_t i = 0; i < Q; i++) {
        uint32_t loc_ind = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < N) {
            T elm = const_cast<const T&>(shmem_inp[loc_ind]);
            d_out[glb_ind] = elm;
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
copyFromGlb2ShrMem(const uint32_t glb_offs, const uint32_t N, const T &ne, T *d_inp, volatile T *shmem_inp)
{
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
copyFromShr2GlbMem(const uint32_t glb_offs, const uint32_t N, T *d_out, volatile T *shmem_inp)
{
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
template<class T, uint32_t Q>
__device__ inline bool
ez2Reg( volatile T* u,
        const uint32_t m ) {
    
    bool retval = true;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && u[idx] != 0) {
            retval = false;
        }
    }
    __syncthreads();
    return retval;
}

/**
 * Sets retval in shared memory to 1 if u is zero or 0 otherwise
 */
template<class T, uint32_t Q>
__device__ inline void
ez2Sh( volatile T* u,
       uint32_t* retval,
       const uint32_t m ) {
    
    *retval = 1;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && u[idx] != 0) {
            *retval = 0;
        }
    }
    __syncthreads();
}

/**
 * Sets retval to 1 if u and v are equal or 0 otherwise
 */
template<class T, uint32_t Q>
__device__ inline void
eq( volatile T* u,
    volatile T* v,
    uint32_t* retval,
    const uint32_t m ) {
    
    *retval = 1;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && u[idx] != v[idx]) {
            *retval = 0;
        }
    }
    __syncthreads();
}

/**
 * Sets the first element to d and zeros the rest.
 * @remark Used for bigints in register memory.
 * 
 * @param u Register memory representation of bigint
 * @param d Value to set first element of biging
 */
template<class T, uint32_t Q>
__device__ inline void
set4Reg( T* u[Q],
         const T d,
         const T m ) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < m) {
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
template<class T, uint32_t Q>
__device__ inline void
set4Shm( volatile T* u,
         const T d,
         const T m ) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m) {
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
template<class T, uint32_t Q>
__device__ inline void
zero4Reg( T u[Q],
          const T m ) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < m) {
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
template<class T, uint32_t Q>
__device__ inline void
zero4Shm( volatile T* u,
          const T m ) {
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m) {
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
template<class T, uint32_t Q>
__device__ inline void
prec( volatile T* u,
      volatile T* buf,
      const uint32_t m ) {

    if (threadIdx.x == 0) {
        buf[0] = 0;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && u[idx] != 0 && idx > buf[0]) {
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
template <class T, uint32_t Q>
__device__ inline T
prec4Reg( T u[Q],
          T* p,
          const uint32_t m ) {

    int highest_idx = -1;
    int old = p[0];
    p[0] = 0;
    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < m && u[idx] != 0) {
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
template <class T, uint32_t Q>
__device__ inline void
prec4Shm( volatile T* u,
          T* p,
          const uint32_t m ) {

    int highest_idx = -1;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && u[idx] != 0) {
            highest_idx = max(highest_idx, idx);
        }
    }
    atomicMax(p, highest_idx + 1);
}



#endif






#endif // KERNEL_HELPER