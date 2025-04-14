#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helpers/helper.h"


int main() {
    using Base = U32bits;
    using uint_t = Base::uint_t;

  //  srand(time(NULL));
    bool stop = false;
    const uint32_t num_instances = 1;
    const uint32_t Q = 4;
    const uint32_t M = 4096;
    const uint32_t total_work = M * num_instances;
    const uint32_t total_work_size = total_work * sizeof(uint_t);

    for (int i = 0; i < 100 && !stop; i++) {
        printf("\rIteration: %u", i);
        uint_t uPrec = (rand() % (M-3)) + 1;
        uint_t vPrec = (rand() % uPrec) + 3;
        uint_t* u = randBigInt<uint_t>(uPrec, M, num_instances);
        uint_t* v = randBigInt<uint_t>(vPrec, M, num_instances);
        uint_t* quo = (uint_t*)calloc(total_work, sizeof(uint_t));
        uint_t* rem = (uint_t*)calloc(total_work, sizeof(uint_t));

        uint_t *d_u, *d_v, *d_quo, *d_rem;
        cudaMalloc((void **)&d_u, total_work_size);
        cudaMalloc((void **)&d_v, total_work_size);
        cudaMalloc((void **)&d_quo, total_work_size);
        cudaMalloc((void **)&d_rem, total_work_size);

        cudaMemcpy(d_u, u, total_work_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, v, total_work_size, cudaMemcpyHostToDevice);

        cudaFuncSetAttribute(divShinv<Base, M,Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98000);

        divShinv<Base, M, Q><<<num_instances, M/Q, 2 * M * sizeof(uint_t)>>>(d_u, d_v, d_quo, d_rem);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );

        cudaMemcpy(quo, d_quo, total_work_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(rem, d_rem, total_work_size, cudaMemcpyDeviceToHost);

        uint_t* quo_gmp = (uint_t*)calloc(total_work, sizeof(uint_t));
        uint_t* rem_gmp = (uint_t*)calloc(total_work, sizeof(uint_t));
        //uint32_t** u_gmp = (uint32_t**)calloc(num_instances, sizeof(uint32_t));
        //uint32_t** v_gmp = (uint32_t**)calloc(num_instances, sizeof(uint32_t));

        //for (int i = 0; i < num_instances; i++) {
        //    //v_gmp[i] = (uint32_t*)calloc(M, sizeof(uint32_t));
        //    //u_gmp[i] = (uint32_t*)calloc(M, sizeof(uint32_t));
        //    quo_gmp[i] = (uint32_t*)calloc(M, sizeof(uint32_t));
        //    rem_gmp[i] = (uint32_t*)calloc(M, sizeof(uint32_t));
        //    //for (int j = 0; j < M; j++) {
        //    //    u_gmp[i][j] = u[j + (M * i)];
        //    //    v_gmp[i][j] = v[j + (M * i)];
        //    //}
        //}
        for (int i = 0; i < num_instances; i++){
            div_gmp(&u[i*M], &v[i*M], &quo_gmp[i*M], &rem_gmp[i*M], M);
        }
        //for (int j = 0; j < num_instances; j++){
            for (int i = 0; i < total_work; i++) {
                if (quo[i] != quo_gmp[i] || rem[i] != rem_gmp[i]) {
                    stop = true;
                    printf("\nInputs:\n");
                    // prnt("  u", u, M);
                    // prnt("  v", v, M);
                    //printSlice(u, 'u', i, M);
                    //printSlice(v, 'v', i, M);
                    printf("u = %u, at %d \n",u[i],i);
                    printf("v = %u, at %d \n",v[i],i);
                    printf("Output:\n");
                    printf("quo = %u, at %d \n",quo[i],i);
                    printf("rem = %u, at %d \n",rem[i],i);
                    //printSlice(quo, 'q', i, M);
                    //printSlice(rem, 'r', i, M);
                    // prnt("  q", quo, M);
                    // prnt("  r", rem, M);
                    printf("GMP:\n");
                    printf("quo_gmp = %u, at %d \n",quo_gmp[i],i);
                    printf("rem_gmp = %u, at %d \n",rem_gmp[i],i);
                    //printSlice(quo_gmp, 'q', i, M);
                    //printSlice(rem_gmp, 'r', i, M);
                    // prnt("  q", quo_gmp, M);
                    // prnt("  r", rem_gmp, M);
                    return 1;
                }
            }
            //for (int j = 0; j < 2; j++){
            //    printf("instance : %d \n", j);
            //    printf("digits: \n ");
            //    for (int i = 0; i < M; i++) {
            //            //printf("u:       (%u, @%d), \n",u[i],i);
            //            //printf("v:       (%u, @%d), \n",v[i],i);
            //            printf("quo:     (%u, @%d), \n",quo[i + (j * M)],i + (j * M));
            //            printf("rem:     (%u, @%d), \n",rem[i + (j * M)],i + (j * M));
            //            printf("quo_gmp: (%u, @%d), \n",quo_gmp[i + (j * M)],i + (j * M));
            //            printf("rem_gmp: (%u, @%d), \n",rem_gmp[i + (j * M)],i + (j * M));
            //    }
            //}
            
        //}
        free(u); free(v); free(quo); free(rem); free(quo_gmp); free(rem_gmp);
    }
    printf("\nDone\n");
    return 0;
}