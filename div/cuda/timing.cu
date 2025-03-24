#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helper.h"

// function calling division kernel
// and does the timing of the kernel
template<uint32_t M, uint32_t Q>
void gpuDiv (int num_instances){
    
    uint32_t total_work = M * num_instances;

    uint32_t uPrec = (M / 2) - 1;
    uint32_t vPrec = (uPrec) - 3;
    uint32_t* u = randBigInt(uPrec, M, num_instances);
    uint32_t* v = randBigInt(vPrec, M, num_instances);
    uint32_t* quo = (uint32_t*)calloc(total_work, sizeof(uint32_t));
    uint32_t* rem = (uint32_t*)calloc(total_work, sizeof(uint32_t));

    uint32_t *d_u, *d_v, *d_quo, *d_rem;
    cudaMalloc((void **)&d_u, total_work * sizeof(uint32_t));
    cudaMalloc((void **)&d_v, total_work * sizeof(uint32_t));
    cudaMalloc((void **)&d_quo, total_work * sizeof(uint32_t));
    cudaMalloc((void **)&d_rem, total_work * sizeof(uint32_t));

    cudaMemcpy(d_u, u, total_work * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, total_work * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // dry run to load kernel into hardware 
    // what is the point of adding tuborg out of nowhere? bajer is best?
    {
        divShinv<M, Q><<<num_instances, M/Q, 2 * M * sizeof(uint32_t)>>>(d_u, d_v, d_quo, d_rem, num_instances);
        cudaDeviceSynchronize();
        gpuAssert(cudaPeekAtLastError());
    }

    // time to time. pray for performance 

    {
        uint64_t time_elappsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start,NULL);

        // why 25 runs? follow the masters example 
        for (int i = 0; i < 25; i++){
            divShinv<M, Q><<<num_instances, M/Q, 2 * M * sizeof(uint32_t)>>>(d_u, d_v, d_quo, d_rem, num_instances);
        }

        cudaDeviceSynchronize();

        gettimeofday(&t_end,NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);

        time_elappsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / 25;

        gpuAssert(cudaPeekAtLastError());

        // prop uint8 is enough with our kernel.. cachow 
        double runtime_microsecs = time_elappsed;

        printf(" division of %d-bit Big-Int (Base uint_32): device ran %d problem instances \
in %lu micro-seconds \n", M*32, num_instances, runtime_microsecs);


    }

    cudaFree(d_u); cudaFree(d_v); cudaFree(d_quo); cudaFree(d_rem);

    free(u); free(v); free(quo); free(rem);
    
}



int main() {
  //  srand(time(NULL));
    const uint32_t num_instances = 131072;
    const uint32_t Q = 8;
    const uint32_t M = 1024;

    gpuDiv<M,Q>(num_instances);

    return 0;
}