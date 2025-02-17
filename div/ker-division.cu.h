#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

#include "../cuda/ker-fft-help.cu.h"
#include "../cuda/ker-helpers.cu.h"



__device__ inline void
BlockwiseShift( const int n,
           const uint32_t* u,
           uint32_t* r,
           const uint32_t m ) {
    int idx = blockIdx.y * blockDim.x + threadIdx.x;
    int offset = idx - n;
    
    if (idx < m) {
        if (n >= 0) {   // Right shift
            r[idx] = (offset >= 0) ? u[offset] : 0;
        } else {        // Left shift
            r[idx] = (offset < m) ? u[offset] : 0;
        }
    }
}


// __device__ inline void
// BlockwiseShift( const int n,
//            const digit_t* u,
//            digit_t* r,
//            const prec_t m ) {
//     const unsigned int idx = threadIdx.x;
    
//     if (n >= 0)
//     { // Right shift
//         #pragma unroll
//         for (int i = m - 1; i >= 0; i--)
//         {
//             int offset = i - n;
//             r[i] = (offset >= 0) ? u[offset] : 0;
//         }
//     }
//     else
//     { // Left shift
//         #pragma unroll
//         for (int i = 0; i < m; i++)
//         {
//             int offset = i - n;
//             r[i] = (offset < m) ? u[offset] : 0;
//         }
//     }
// }




#endif // KERNEL_DIVISION