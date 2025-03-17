#ifndef KERNEL_BPOW
#define KERNEL_BPOW

#include "ker-div-helper.cu.h"


/**
 * Initializes a bpow in register memory
 * Shared memory is preserved - uses reverse indexing
 * 
 * @returns a pointer to the start of the register array
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline uint32_t
bpow4Reg( T u[Q]
        , uint32_t bpow
) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        if (idx < M) {
            if (idx == bpow) {
                u[idx] == 1;
            } else {
                u[idx] == 0;
            }

        }
    }
    return &u;
}


/**
 * @brief u < B
 * Compares bigint with bpow in register memory using shmem buffer
 * Shared memory is preserved - uses reverse indexing
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline bool
lt4Reg2Bpow( T u[Q]
           , uint32_t bpow
           , volatile bool* shmem
) {
    bool retval;
    short tmp = ((short*)shmem)[0];
    shmem[1] = false;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        // Do reverse indexing
        int idx = M - (Q * threadIdx.x + i) - 1;

        if (0 <= idx && shmem[1]) {
            if (idx < bpow) {
                shmem[0] = true;
                shmem[1] = false;
            } else if (idx > bpow) {
                shmem[0] = false;
                shmem[1] = false;
            } else if (u[idx] < 1) {
                shmem[0] = true;
                shmem[1] = false;
            } else if (u[idx] > 1) {
                shmem[0] = false;
                shmem[1] = false;
            }
        }
        shmem[0] = false;
    }
    __syncthreads();

    retval = shmem[0];
    shmem[0] = tmp;
    return retval;
}


/**
 * @brief B < u
 * Compares bigint with bpow in register memory using shmem buffer
 * Shared memory is preserved - uses reverse indexing
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline bool
lt4Bpow2Reg( uint32_t bpow
           , T u[Q]
           , volatile bool* shmem
) {
    bool retval;
    short tmp = ((short*)shmem)[0];
    shmem[1] = false;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        // Do reverse indexing
        int idx = M - (Q * threadIdx.x + i) - 1;

        if (0 <= idx && shmem[1]) {
            if (idx < bpow) {
                shmem[0] = false;
                shmem[1] = false;
            } else if (idx > bpow) {
                shmem[0] = true;
                shmem[1] = false;
            } else if (u[idx] < 1) {
                shmem[0] = false;
                shmem[1] = false;
            } else if (u[idx] > 1) {
                shmem[0] = true;
                shmem[1] = false;
            }
        }
        shmem[0] = false;
    }
    __syncthreads();

    retval = shmem[0];
    shmem[0] = tmp;
    return retval;
}

/**
 * @brief B < 2u
 * Compares 2 * bigint with bpow in register memory using shmem buffer
 * Shared memory is preserved - uses reverse indexing
 * 
 * @see lt4Bpow2Reg
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline bool
lt4Bpow2RegMul2( uint32_t bpow
               , T u[Q]
               , volatile bool* shmem
               , int (*pred)(int) 
) {
    bool retval;
    short tmp = ((short*)shmem)[0];
    shmem[1] = false;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        // Do reverse indexing
        int idx = M - (Q * threadIdx.x + i) - 1;

        if (0 <= idx && shmem[1]) {
            if (idx < bpow) {
                shmem[0] = false;
                shmem[1] = false;
            } else if (idx > bpow) {
                shmem[0] = true;
                shmem[1] = false;
            } else if (2* u[idx] < 1) {
                shmem[0] = false;
                shmem[1] = false;
            } else if (2 * u[idx] > 1) {
                shmem[0] = true;
                shmem[1] = false;
            }
        }
        shmem[0] = false;
    }
    __syncthreads();

    retval = shmem[0];
    shmem[0] = tmp;
    return retval;
}


/**
 * @brief u == B
 * Compares bigint with bpow in register memory using shmem buffer
 * Shared memory is preserved - uses reverse indexing
 */
template<class T, uint32_t M, uint32_t Q>
__device__ inline bool
eq4Reg2Bpow( T u[Q]
           , uint32_t bpow
           , volatile bool* shmem
) {
    bool retval;
    int tmp = shmem[0];
    shmem[0] = true; 

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;

        if (idx < M) {
            if (idx == bpow && u[idx] != 1
             || idx != bpow && u[idx] != 0) {
                shmem[0] = false;
            }
        }
    }
    __syncthreads();

    retval = shmem[0];
    shmem[0] = tmp;
    return retval;
}









#endif // KERNEL_BPOW