#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../ker-division.cu.h"
// #include "../sequential/helper.h"
// #include "../sequential/div.h"
#include "../../cuda/helper.h"


__global__ void CallShift(
    const int n,
    const uint32_t* u,
    uint32_t* r,
    const uint32_t m) {
        BlockwiseShift(n, u, r, m);
    }


// template<class uint_t>  // m is the size of the big word in Base::uint_t units
// void testBasicOps ( int n, int runs ) {    
//     uint_t* d_as;
//     size_t mem_size_nums = n * sizeof(uint_t);
    
//     cudaMalloc((void**) &d_as, mem_size_nums);
    
//     const size_t B = 256;
    
//     // timing instrumentation shifting
//     {
//         CallShift<<<1, B>>>(n, d_as);
//         cudaDeviceSynchronize();
//         gpuAssert( cudaPeekAtLastError() );
    
    
//         uint64_t elapsed;
//         struct timeval t_start, t_end, t_diff;
//         gettimeofday(&t_start, NULL); 
        
//         for(int i=0; i<runs; i++) {
//             additionKer<uint_t><<< (n+B-1)/B, B >>>(n, d_as);
//         }
        
//         cudaDeviceSynchronize();

//         gettimeofday(&t_end, NULL);
//         timeval_subtract(&t_diff, &t_end, &t_start);
//         elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / runs;

//         gpuAssert( cudaPeekAtLastError() );

//         printf( "Base Addition runs in: %.2f us\n", (double)elapsed );        
//     }
//     cudaFree(d_as);
// }



int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }

    uint32_t m = 10;
    int size = m * sizeof(uint32_t);

    uint32_t u[10] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t* v = (uint32_t*)malloc(size);
    uint32_t* v_D;
    cudaMalloc(&v_D, size);

    cudaMemcpy(v_D, u, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    CallShift<<<1, threadsPerBlock>>>(6, v_D, v_D, m);
    cudaDeviceSynchronize();

    gpuAssert( cudaPeekAtLastError() );
    cudaMemcpy(v, v_D, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++) {
        if (v[i] != u[i]) {
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, v[i], u[i]);

            // free(u);
            free(v);
            cudaFree(v_D);
            return 1;
        }
    }
    // free(u);
    free(v);
    cudaFree(v_D);
    return 0;
}








