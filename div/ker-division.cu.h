#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

#include "../cuda/helper.h"
// #include "../cuda/ker-fft-help.cu.h"
// #include "../cuda/ker-helpers.cu.h"


/**
 * Helper kernel for copying from global to shared memory
 */
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromGlb2ShrMem( const uint32_t glb_offs
                  , const uint32_t N
                  , const T& ne
                  , T* d_inp
                  , volatile T* shmem_inp
) {
    #pragma unroll
    for(uint32_t i=0; i<CHUNK; i++) {
        uint32_t loc_ind = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        T elm = ne;
        if(glb_ind < N) { elm = d_inp[glb_ind]; }
        shmem_inp[loc_ind] = elm;
    }
    __syncthreads();
}

/**
 * Helper kernel for copying from shared to global memory
 */
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T* d_out
                  , volatile T* shmem_inp
) {
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        uint32_t loc_ind = blockDim.x * i + threadIdx.x;
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < N) {
            T elm = const_cast<const T&>(shmem_inp[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads();
}



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
 * @brief 
 * @todo Contains bug
 * 
 * @tparam Q 
 * @param n 
 * @param u 
 * @param res 
 * @param m 
 * @return __device__ 
 */
template<uint32_t Q>
__device__ inline void
BlockwiseShift( const int n,
                const uint32_t* u,
                uint32_t* res,
                const uint32_t m ) {

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        
        if (idx >= m) {
           continue; 
        }

        int offset = idx - n;
        if (n >= 0) {   // Right shift
            res[idx] = (offset >= 0) ? u[offset] : 0;
        } else {        // Left shift
            res[idx] = (offset < m) ? u[offset] : 0;
        }
    }
}


/**
 * @brief 
 * @todo Contains bug
 * 
 * @tparam Q 
 * @param u 
 * @param d 
 * @param v 
 * @param buf 
 * @param m 
 * @return __device__ 
 */
template<int Q>
__device__ inline void
BlockwiseMultD( uint32_t* u,
                uint32_t  d,
                uint32_t* v,
                volatile uint64_t* buf,
                const uint32_t m ) {
                    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx] = ((uint64_t)u[idx]) * (uint64_t)d;
    }
    __syncthreads();

    // #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx + 1] += (buf[idx] >> 32);
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        v[idx] = (uint32_t)buf[idx];
    }
    __syncthreads();
}






#endif // KERNEL_DIVISION