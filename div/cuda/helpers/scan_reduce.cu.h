template<class OP>
__device__ inline int scanIncWarp(int u, uint32_t lane) {
    #pragma unroll
    for (int i = 1; i < WARP; i *= 2) {
        int elm = __shfl_up_sync(0xFFFFFFFF, u, (lane >= i) ? i : 0);
        u = OP::apply(elm, u);
    }
    return u;
}

template<class OP>
__device__ inline int reduceBlock(uint32_t u, volatile uint32_t* sh_mem) {
    int idx = threadIdx.x;
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    int res = scanIncWarp<OP>(u, lane);

    if (lane == (WARP-1) || idx == blockDim.x - 1) { sh_mem[warpid] = res; } 
    __syncthreads();

    if (warpid == 0) {
        res = scanIncWarp<OP>(sh_mem[threadIdx.x], lane);
        if (threadIdx.x == ((blockDim.x + WARP - 1) / WARP) - 1) {
            sh_mem[0] = res;
        }
    }
    
    __syncthreads();
    return sh_mem[0];
} 

template<class OP>
__device__ inline typename OP::RedElTp
scanIncWarp( volatile typename OP::RedElTp* ptr, const unsigned int idx ) {
    const unsigned int lane = idx & (WARP-1);
    #pragma unroll
    for(uint32_t i=0; i<lgWARP; i++) {
        const uint32_t p = (1<<i);
        if( lane >= p ) ptr[idx] = OP::apply(ptr[idx-p], ptr[idx]);
        __syncwarp();
    }
    return OP::remVolatile(ptr[idx]);
}

template<class OP>
__device__ inline typename OP::RedElTp
scanIncBlock(volatile typename OP::RedElTp* ptr, const unsigned int idx) {
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    // 1. perform scan at warp level
    typename OP::RedElTp res = scanIncWarp<OP>(ptr,idx);
    __syncthreads();   
    if(blockDim.x <= 32) { // block < WARP optimization
        return res;
    }
    
    __syncthreads();

    // 2. place the end-of-warp results in the first warp. 
    if (lane == (WARP-1)) { ptr[warpid] = res; } 
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }
    __syncthreads();
    // 5. publish to shared memory
    ptr[idx] = res;
    __syncthreads();
    
    return res;
}


//Improved scan inclusive version but uses too many registers
//
// template<class OP>
// __device__ inline int scanIncBlock(uint32_t u, volatile uint32_t* sh_mem) {
//     __syncthreads();
//     int idx = threadIdx.x;
//     const unsigned int lane   = idx & (WARP-1);
//     const unsigned int warpid = idx >> lgWARP;

//     int res = scanIncWarp<OP>(u, lane);

//     if (lane == (WARP-1)) { sh_mem[warpid] = res; } 
//     __syncthreads();

//     if (warpid == 0) {
//         scanIncWarp<OP>(sh_mem[threadIdx.x], lane);
//     }
//     __syncthreads();

//     if (warpid > 0) {
//         res = OP::apply(sh_mem[warpid-1], res);
//     }
//     __syncthreads();

//     sh_mem[idx] = res;
//     __syncthreads();
    
//     return res;
// }