#define WARP   (32)
#define lgWARP  (5)

template<uint32_t Q>
__device__ inline void cpyGlb2Sh2Reg(uint32_t* AGlb, uint32_t* ASh, uint32_t AReg[Q]) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        ASh[idx] = AGlb[idx];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        AReg[i] = ASh[Q * threadIdx.x + i];
    }
}

template<uint32_t Q>
__device__ inline void cpyReg2Sh2Glb(uint32_t AReg[Q], uint32_t* ASh, uint32_t* AGlb) {
    #pragma unroll
    for (int i=0; i < Q; i++) {
        ASh[Q * threadIdx.x + i] = AReg[i];
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = i * blockDim.x + threadIdx.x;
        AGlb[idx] = ASh[idx];
    }
}

template<uint32_t Q>
__device__ inline uint32_t prec(uint32_t u[Q], uint32_t* sh_mem) {
    sh_mem[0] = -1;
    
    #pragma unroll
    for (int i = Q-1; i >= 0; i--) {
        if (u[i] != 0) {
            atomicMax(sh_mem, Q * threadIdx.x + i);
            break;
        }
    }
    __syncthreads();    
    return sh_mem[0] + 1;
}

template<uint32_t Q>
__device__ inline bool eq(uint32_t u[Q], uint32_t bpow, uint32_t* sh_mem) {
    sh_mem[0] = true;
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != (bpow == (i * blockDim.x + threadIdx.x))) {
            sh_mem[0] = false;
            break;
        }
    }
    __syncthreads();    
    return sh_mem[0];
}

template<uint32_t Q>
__device__ inline void set(uint32_t u[Q], uint32_t d, uint32_t idx) {
    if (threadIdx.x == idx / Q) {
        u[idx % Q] = d;
    }
}

template<uint32_t M, uint32_t Q>
__device__ inline void shift(int n, uint32_t u[Q], uint32_t* sh_mem, uint32_t RReg[Q]) {
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        int idx = Q * threadIdx.x + i;
        int offset = idx - n;

        if (offset >= 0 && offset < M) {
            sh_mem[offset] = u[i];
        }
        else {
            sh_mem[M-idx-1] = 0;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        RReg[i] = sh_mem[Q * threadIdx.x + i];
    }
}

template<uint32_t Q>
__device__ inline void quo(uint32_t bpow, uint32_t d, uint32_t RReg[Q])
{
    uint64_t r = 1;
    for (int i = bpow - 1; i >= 0; i--)
    {
        r <<= 32; 
        if (r >= d) {
            if (threadIdx.x == i / Q) {
                RReg[i % Q] = r / d;
            }
            r %= d;
        }
    }
}

__device__ inline uint8_t lessThanWarp(uint8_t u, uint32_t lane) {
    #pragma unroll
    for (int i = 1; i < WARP; i *= 2) {
        uint8_t elm = __shfl_up_sync(0xFFFFFFFF, u, (lane >= i) ? i : 0);
        u |= (u & 0b10) && (elm & 0b01);
        u = (u & ~0b10) | (u & elm & 0b10);
    }
    return u;
}


template<uint32_t Q>
__device__ inline bool lt(uint32_t u[Q], uint32_t v[Q], uint32_t* sh_mem) {
    uint8_t RReg[Q] = {0};
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] < v[i]) {
            RReg[i] |= 0b01;
        }
        else if (u[i] == v[i]) {
            RReg[i] |= 0b10;
        }
        if (i != 0) {
            RReg[i] |= (RReg[i] & 0b10) && (RReg[i-1] & 0b01);
            RReg[i] = (RReg[i] & ~0b10) | (RReg[i] & RReg[i-1] & 0b10);
        }
    }

    uint16_t idx = threadIdx.x;
    const unsigned int lane   = idx & (WARP-1);
    const unsigned int warpid = idx >> lgWARP;

    uint8_t res = lessThanWarp(RReg[Q-1], lane);

    if (lane == (WARP-1) || idx == blockDim.x - 1) { sh_mem[warpid] = res; } 
    __syncthreads();

    if (warpid == 0) {
        res = lessThanWarp(sh_mem[threadIdx.x], lane);
        if (threadIdx.x == ((blockDim.x + WARP - 1) / WARP) - 1) {
            sh_mem[0] = res;
        }
    }
     __syncthreads();

    return sh_mem[0] & 0b01;
}

template<uint32_t Q>
__device__ inline void add1(uint32_t u[Q], uint32_t* sh_mem) {
    sh_mem[0] = UINT32_MAX;

    #pragma unroll
    for (int i = 0; i < Q; i++) {
        if (u[i] != UINT32_MAX) {
            atomicMin(sh_mem, Q * threadIdx.x + i);
            break;
        }
    }
    __syncthreads();

    uint32_t min_index = sh_mem[0];
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        idx = Q * threadIdx.x + i;
        if (idx < min_index) {
            u[i] == 0;
        }
        else if (idx == min_index) {
            u[i] += 1;
        }
    }
}

