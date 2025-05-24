#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "../ker-division.cu.h"
#include "../helpers/helper.h"

#define GPU_RUNS_DIV   3
#define GPU_RUNS_MUL   3

int main()
{
    using Base = U32bits;
    using uint_t = Base::uint_t;
    const uint64_t num_instances = 10000;
    const uint32_t M = 4096;
    const uint32_t Q = 4;
    uint64_t mem_size = num_instances * M * sizeof(uint_t);
    uint64_t mul_elapsed;
    uint64_t div_elapsed;

    uint_t uPrec = M-Q;
    uint_t vPrec = 3;
    uint_t* u = randBigInt<uint_t>(uPrec, M, num_instances);
    uint_t* v = randBigInt<uint_t>(vPrec, M, num_instances);

    uint_t *d_u, *d_v, *d_quo, *d_rem, *d_mul;
    cudaMalloc((void **)&d_u, mem_size);
    cudaMalloc((void **)&d_v, mem_size);
    cudaMalloc((void **)&d_quo, mem_size);
    cudaMalloc((void **)&d_rem, mem_size);
    cudaMalloc((void **)&d_mul, mem_size);
    cudaMemcpy(d_u, u, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, mem_size, cudaMemcpyHostToDevice);

    {   // dry run mul
        bmulKerQ<Base, 1, M, Q><<<num_instances, M/Q>>>(num_instances, d_u, d_v, d_mul);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }
    {   // timing mul
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 

        for(int i=0; i<GPU_RUNS_MUL; i++) {
            bmulKerQ<Base, 1, M, Q><<<num_instances, M/Q>>>(num_instances, d_u, d_v, d_mul);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        mul_elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_MUL;

        gpuAssert( cudaPeekAtLastError() );
        printf( "Multiplcation took %lu microsecs \n", mul_elapsed);
    }
    cudaFuncSetAttribute(divShinvKer<Base,M,Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    {   // dry run div
        divShinvKer<Base, M, Q><<<num_instances, M/Q, 2 * M * sizeof(uint_t)>>>(d_u, d_v, d_quo, d_rem);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }
    {   // timing div
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<GPU_RUNS_DIV; i++) {
            divShinvKer<Base, M, Q><<<num_instances, M/Q, 2 * M * sizeof(uint_t)>>>(d_u, d_v, d_quo, d_rem);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        div_elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_DIV;

        gpuAssert( cudaPeekAtLastError() );
        printf( "Division took %lu microsecs \n", div_elapsed);
        printf( "Division was %u times slower than multiplication \n", div_elapsed/mul_elapsed);
    }

    uint_t *quo, *rem, *gmp_quo, *gmp_rem;
    {   // validation div
        quo = (uint_t*)calloc(M*num_instances, sizeof(uint_t));
        rem = (uint_t*)calloc(M*num_instances, sizeof(uint_t));
        gmp_quo = (uint_t*)calloc(M*num_instances, sizeof(uint_t));
        gmp_rem = (uint_t*)calloc(M*num_instances, sizeof(uint_t));
        cudaMemcpy(quo, d_quo, mem_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(rem, d_rem, mem_size, cudaMemcpyDeviceToHost);

        gmpDiv<uint_t, M>(num_instances, u, v, gmp_quo, gmp_rem);
        validateExact(gmp_quo, quo, gmp_rem, rem, M*num_instances);
    }
    
    free(u); free(v); free(quo); free(rem); free(gmp_quo); free(gmp_rem);
    return 0;
}