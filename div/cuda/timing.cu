#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "helpers/helper.h"
#include "ker-division.cu.h"



int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

// function calling division kernel
// and does the timing of the kernel
template<class Base, uint32_t M, uint32_t Q>
void gpuDiv (int num_instances){
    
    uint32_t total_work = M * num_instances;

    uint32_t uPrec = (M) - 1;
    uint32_t vPrec = (uPrec) - 3;
    uint32_t* u = randBigInt<uint32_t>(uPrec, M, num_instances);
    uint32_t* v = randBigInt<uint32_t>(vPrec, M, num_instances);
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
        divShinv<Base, M, Q><<<num_instances, M/Q, 2 * M * sizeof(uint32_t)>>>(d_u, d_v, d_quo, d_rem, num_instances);
        cudaDeviceSynchronize();
        gpuAssert(cudaPeekAtLastError());
    }

    // time to time. pray for performance 

    {
        uint64_t time_elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start,NULL);

        // why 25 runs? follow the masters example 
        for (int i = 0; i < 25; i++){
            divShinv<Base, M, Q><<<num_instances, M/Q, 2 * M * sizeof(uint32_t)>>>(d_u, d_v, d_quo, d_rem, num_instances);
        }

        cudaDeviceSynchronize();

        gettimeofday(&t_end,NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);

        time_elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / 50;

        gpuAssert(cudaPeekAtLastError());

        // prop uint8 is enough with our kernel.. cachow 
        double runtime_microsecs = time_elapsed; 
        double bytes_accesses = 3.0 * num_instances * M * sizeof(uint32_t);  
        double gigabytes = bytes_accesses / (runtime_microsecs * 1000);

        printf( "Division on %d-bit Big-Numbers (base u%d) runs %d instances in: \
%lu microsecs, GB/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , M*32, 32, num_instances, time_elapsed, gigabytes, num_instances / runtime_microsecs
              );


    }

    cudaFree(d_u); cudaFree(d_v); cudaFree(d_quo); cudaFree(d_rem);

    free(u); free(v); free(quo); free(rem);
    
}



int main() {
  //  srand(time(NULL));
    const uint32_t num_instances = 32768;
    const uint32_t Q = 8;
    const uint32_t M = 4096;

    gpuDiv<U32bits,M,Q>(num_instances);

    return 0;
}