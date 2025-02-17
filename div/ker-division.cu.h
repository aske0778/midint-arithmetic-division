#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

#include "../cuda/helper.h"
// #include "../cuda/ker-fft-help.cu.h"
// #include "../cuda/ker-helpers.cu.h"



template<int Q>
__device__ inline void
BlockwiseShift( const int n,
           const uint32_t* u,
           uint32_t* r,
           const uint32_t m ) {
    int idx = (blockIdx.y * blockDim.x + threadIdx.x) * Q;
    int offset = idx - n;
    
    #pragma unroll
    for (int i = idx; i < m; i++) {
        if (n >= 0) {   // Right shift
            r[i] = (offset >= 0) ? u[offset] : 0;
        } else {        // Left shift
            r[i] = (offset < m) ? u[offset] : 0;
        }
    }
}


// template<int Q>
// __device__ inline void
// BlockwiseShift( const int n,
//            const uint32_t* u,
//            uint32_t* r,
//            const uint32_t m ) {
//     int idx = blockIdx.y * blockDim.x + threadIdx.x;
//     int offset = idx - n;
    
//     if (idx < m) {
//         if (n >= 0) {   // Right shift
//             r[idx] = (offset >= 0) ? u[offset] : 0;
//         } else {        // Left shift
//             r[idx] = (offset < m) ? u[offset] : 0;
//         }
//     }
// }







#endif // KERNEL_DIVISION