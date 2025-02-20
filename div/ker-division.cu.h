#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

#include "../cuda/helper.h"
// #include "../cuda/ker-fft-help.cu.h"
// #include "../cuda/ker-helpers.cu.h"


/**
 * Helper kernel for copying from global to shared memory
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
 * Helper kernel for copying from shared to global memory
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
template<class T, uint32_t Q>
__device__ inline void
shift( const int n,
       volatile T* u,
       volatile T* res,
       const uint32_t m ) {

    if (n >= 0) {   // Left shift
        #pragma unroll
        for (int i = 0; i < Q; i++) {
            int idx = i * blockDim.x + threadIdx.x;
            if (idx < m) {
                int offset = idx - n;
                res[idx] = (offset < m) ? u[offset] : 0;
            }
            __syncthreads();
        }
    } else {        // Right shift
        #pragma unroll
        for (int i = Q; i >= 0; i--) {
            int idx = i * blockDim.x + threadIdx.x;
            if (idx < m) {
                int offset = idx - n;
                res[idx] = (offset >= 0) ? u[offset] : 0;
            }
            __syncthreads();
        }
    }
    __syncthreads();
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
template<class T, class T2, uint32_t Q>
__device__ inline void
multd( volatile T* u,
       const uint32_t d,
       volatile T* v,
       volatile T2* buf,
       const uint32_t m ) {
                    
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx] = ((T2)u[idx]) * (T2)d;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        buf[idx + 1] += (buf[idx] >> 32);
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        v[idx] = (T)buf[idx];
    }
    __syncthreads();
}






#endif // KERNEL_DIVISION