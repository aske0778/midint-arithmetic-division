#ifndef KERNEL_DIVISION
#define KERNEL_DIVISION

// #include "../../cuda/ker-fft-help.cu.h"
// #include "../../cuda/ker-helpers.cu.h"
// #include "../../cuda/helper.h"

template <uint32_t Q>
__device__ inline void
shift(const int n,
      const uint32_t *u,
      uint32_t *res,
      const uint32_t m)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

#pragma unroll
    for (int i = 0; i < Q; i++)
    {
        int idx = tid * (i + 1);

        if (idx >= m)
        {
            return;
        }

        int offset = idx - n;
        if (n >= 0)
        { // Right shift
            res[idx] = (offset >= 0) ? u[offset] : 0;
        }
        else
        { // Left shift
            res[idx] = (offset < m) ? u[offset] : 0;
        }
    }
}

__global__ void div_shinv(const uint32_t *u, const uint32_t *v, uint32_t *res, const uint32_t m)
{
    shift<2>(2, u, res, m);
    __syncthreads();
}

#endif // KERNEL_DIVISION