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

template<class T>
void randomInit(T* data, uint64_t size) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = rand();
}

// template<typename uint_t>
// uint64_t numAd32OpsOfMultInst(uint32_t m0) {
//     uint32_t m = m0*sizeof(uint_t) / 4;
//     uint32_t lgm = 0, mm = m;
//     for( ; mm > 1; mm >>= 1) lgm++;
//     //printf("Log %d is %d\n", m, lgm);
//     return 300 * m * lgm;
// }

/**
 * Number of giga-u32-bit unit operations.
 */
template<typename uint_t>
uint64_t numAd32OpsOfDivInst(uint32_t m0) {
    uint32_t m = m0*sizeof(uint_t) / 4;
    uint32_t lgm = 0, mm = m;
    for( ; mm > 1; mm >>= 1) lgm++;
    uint64_t fft_cost = 300 * m * lgm;
    return 3*fft_cost;
}

/**
 * Initialize the `data` array, which has `size` elements:
 * frac% of them are NaNs and (1-frac)% are random values.
 */
void randomMask(char* data, uint64_t size, float frac) {
    for (uint64_t i = 0; i < size; i++) {
        float r = rand() / (float)RAND_MAX;
        data[i] = (r >= frac) ? 1 : 0;
    }
}

/**
 * Validates asb(A - B) < ERR
 */
template<class T>
bool validate(T* A, T* B, const uint64_t sizeAB, const T ERR){
    for(uint64_t i = 0; i < sizeAB; i++) {
        T curr_err = fabs( (A[i] - B[i]) / max(A[i], B[i]) ); 
        if (curr_err >= ERR) {
            printf("INVALID RESULT at flat index %llu: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}

/**
 * Validates exactly A == B
 */
template<class T>
bool validateExact(T* A, T* B, uint64_t sizeAB){
    for(uint64_t i = 0; i < sizeAB; i++) {
        if ( A[i] != B[i] ) {
            printf("INVALID RESULT at flat index %lu: %u vs %u\n", i, A[i], B[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}

template<class T, uint32_t m>
void printInstance(uint32_t q, T* as) {
    printf(" [ %lu", as[q*m]);
    for(int i=1; i<m; i++) {
        printf(", %lu", as[q*m+i]);
    }
    printf("] \n");
}


/**
 * Creates `num_instances` big integers:
 * A big integer consists of `m` u32 words,
 * from which the first `nz` are nonzeroes,
 * and the rest are zeros.
 */
template<int m, int nz>
void ourMkRandom(uint32_t num_instances, uint32_t* as) {
    uint32_t* it_as = as;

    for(int i=0; i<num_instances; i++, it_as += m) {
        for(int k = 0; k < m; k++) {
            uint32_t v = 0;
            if(k < nz) {
                uint32_t low  = rand()*2;
                uint32_t high = rand()*2;
                v = (high << 16) + low;
            }
            it_as[k] = v;
        }        
    }
}


using namespace std;

#define GPU_RUNS_DIV    25
#define GPU_RUNS_MUL    25
#define ERR         0.000005

#define WITH_VALIDATION 0


template<int m, int nz>
void mkRandArrays ( int num_instances
                  , uint64_t** h_as
                  , uint64_t** h_bs
                  , uint64_t** h_rs_gmp
                  , uint64_t** h_rs_our
                  ) {

    *h_as     = (uint64_t*) malloc( num_instances * m * sizeof(uint32_t) );
    *h_bs     = (uint64_t*) malloc( num_instances * m * sizeof(uint32_t) );
    *h_rs_gmp = (uint64_t*) malloc( num_instances * m * sizeof(uint32_t) );
    *h_rs_our = (uint64_t*) malloc( num_instances * m * sizeof(uint32_t) );
        
    ourMkRandom<m, nz>(num_instances, (uint32_t*)*h_as);
    ourMkRandom<m, nz>(num_instances, (uint32_t*)*h_bs);
}

/****************************/
/***** Single Division ******/
/****************************/

template<class Base, uint32_t m>  // m is the size of the big word in Base::uint_t units
void gpuDiv ( uint32_t num_instances
            , typename Base::uint_t* u
            , typename Base::uint_t* v
            , typename Base::uint_t* h_rs
            ) 
{
    using uint_t = typename Base::uint_t;
    //using carry_t= typename Base::carry_t;
    
    uint_t* d_as;
    uint_t* d_bs;
    uint_t* d_rs;
    uint32_t mem_size_nums = num_instances * m * sizeof(uint_t);

    
    // 1. allocate device memory
    cudaMalloc((void**)&d_as, mem_size_nums);
    cudaMalloc((void**)&d_bs, mem_size_nums);
    cudaMalloc((void**)&d_rs, mem_size_nums);
 
    // 2. copy host memory to device
    cudaMemcpy(d_as, u, mem_size_nums, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs, v, mem_size_nums, cudaMemcpyHostToDevice);


    // 3. kernel dimensions
    const uint32_t q = 8; // use 8 for A4500 
    
#if 1
    const uint32_t Bprf = 256;
    const uint32_t m_lft = LIFT_LEN(m, q);
    const uint32_t ipb = ((m_lft / q) >= Bprf) ? 1 : 
                           (Bprf + (m_lft / q) - 1) / (m_lft / q);
#else
    const uint32_t m_lft = m;
    const uint32_t ipb = (128 + m/q - 1) / (m/q);
#endif
    assert(m_lft % q == 0 && m_lft >= q);
    
    // { // maximize the amount of shared memory for the kernel
    //     cudaFuncSetAttribute(divShinv<Base, m, q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * mem_size_nums);  // 131072 out of range
    // }    

    dim3 block( ipb * (m_lft/q), 1, 1);
    dim3 grid ( (num_instances + ipb - 1)/ipb, 1, 1);
    
    // 4. dry run
    {
        quoShinv<m, q><<<num_instances, m/q, 2 * m * sizeof(uint32_t)>>>(d_as, d_bs, d_rs, num_instances);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }
    
    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));
    
    // 5. timing instrumentation
    {
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<GPU_RUNS_DIV; i++) {
            quoShinv<m,q><<< num_instances, m/q,  2 * m * sizeof(uint32_t)>>>(d_as, d_bs, d_rs, num_instances);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_DIV;

        gpuAssert( cudaPeekAtLastError() );

        double runtime_microsecs = elapsed; 
        double num_u32_ops = num_instances * numAd32OpsOfDivInst<uint_t>(m);
        double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

        printf( "Division on %d-bit Big-Numbers (base u%d) runs %d instances in: \
%lu microsecs, Gu32ops/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , m*x*32, Base::bits, num_instances, elapsed, gigaopsu32, num_instances / runtime_microsecs
              );
    }
    
    cudaMemcpy(h_rs, d_rs, mem_size_nums, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_as);
    cudaFree(d_bs);
    cudaFree(d_rs);
}



template<class Base, int m>
void testDivision( int num_instances
                 , typename Base::uint_t* res_gmp
                 , typename Base::uint_t* res_our
                 , uint32_t with_validation
) {
    using uint_t = typename Base::uint_t;
    
    uint_t uPrec = m;
    uint_t vPrec = uPrec - (m/4);

    uint_t* u = randBigInt(uPrec, m, num_instances);
    uint_t* v = randBigInt(vPrec, m, num_instances);

    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));

    if(with_validation)
        gmpQuo<m>(num_instances, u, v, res_gmp);

    gpuDiv<Base, m/x>(num_instances, u, v, res_our);

    if(with_validation)  
        validateExact(res_gmp, res_our, num_instances*m);
}


/////////////////////////////////////////////////////////
// Main program that runs test suits
/////////////////////////////////////////////////////////
 
template<typename Base>
void runDivisions(uint64_t total_work) {

    using uint_t = typename Base::uint_t;
    uint_t *res_gmp, *res_our;

    res_our = (uint_t*)calloc(total_work, sizeof(uint_t));
    res_gmp = (uint_t*)calloc(total_work, sizeof(uint_t));

    
#if 1
    // testDivision<Base, 8192>( total_work/8192, res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base, 4096>( total_work/4096, res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base, 2048>( total_work/2048, res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base, 1024>( total_work/1024, res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base,  512>( total_work/512,  res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base,  256>( total_work/256,  res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base,  128>( total_work/128,  res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base,   64>( total_work/64,   res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base,   32>( total_work/32,   res_gmp, res_our, WITH_VALIDATION );
    testDivision<Base,   16>( total_work/16,   res_gmp, res_our, WITH_VALIDATION );
#endif
    free(res_gmp);
    free(res_our);
}

 
int main (int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s <batch-size>\n", argv[0]);
        exit(1);
    }
        
    const int total_work = atoi(argv[1]);

    cudaSetDevice(1);
    runDivisions<U32bits>(total_work);
}
