#ifndef KERNEL_HELPER
#define KERNEL_HELPER

#include "../../cuda/helper.h"


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

#if 0
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
 * @brief Sets the first element to d and zeros the rest
 */
template<class T, uint32_t Q>
__device__ inline void
set( volatile T* u,
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
 */
template<class T, uint32_t Q>
__device__ inline void
zero( volatile T* u,
      const T d,
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

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        if (idx < m && u[idx != 0] && idx > buf[0]) {
            buf[0] = idx;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        buf[0] = u[buf[0]] + 1;
    }
}







#endif // KERNEL_HELPER