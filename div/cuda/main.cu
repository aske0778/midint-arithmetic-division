#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "helpers/helper.h"
#include "ker-division.cu.h"

using namespace std;

#define GPU_RUNS_MUL    25
#define GPU_RUNS_DIV    15
#define GPU_RUNS_GCD    15
#define ERR         0.000005

#define WITH_VALIDATION 1

#define Q 4


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

/**
 * Validates asb(A - B) < ERR
 */
template<class T>
bool validate(T* A, T* B, const uint64_t sizeAB, const T err){
    for(uint64_t i = 0; i < sizeAB; i++) {
        T curr_err = fabs( (A[i] - B[i]) / max(A[i], B[i]) ); 
        if (curr_err >= err) {
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


template<class Base, int m> // m is the size of the big word in Base::uint_t
void gpuMultiply( int num_instances
                , typename Base::uint_t* h_as
                , typename Base::uint_t* h_bs
                , typename Base::uint_t* h_rs
                ) 
{
    using uint_t = typename Base::uint_t;
    uint_t* d_as;
    uint_t* d_bs;
    uint_t* d_rs;
    size_t mem_size_nums = num_instances * m * sizeof(uint_t);
    
    // 1. allocate device memory
    cudaMalloc((void**) &d_as, mem_size_nums);
    cudaMalloc((void**) &d_bs, mem_size_nums);
    cudaMalloc((void**) &d_rs, mem_size_nums);
 
    // 2. copy host memory to device
    cudaMemcpy(d_as, h_as, mem_size_nums, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs, h_bs, mem_size_nums, cudaMemcpyHostToDevice);
    
    // 3. kernel dimensions; q must be 4; seq-factor = 2*q
    const uint32_t q    = 4;     // 4
    const uint32_t Bprf = 256; //256;
    //const uint32_t Bmax = 1024;
    const uint32_t m_lft = LIFT_LEN(m, q);
    const uint32_t ipb = ((m_lft / q) >= Bprf) ? 1 : 
                           (Bprf + (m_lft / q) - 1) / (m_lft / q);    

    assert( (q % 2 == 0) && (m_lft % q == 0) && (m_lft >= q ) );

    dim3 block( ipb*m_lft/q, 1, 1 );
    dim3 grid ( (num_instances+ipb-1)/ipb, 1, 1);  // BUG: it might not fit exactly!
   
    { // 4. dry run
        //bmulKer<Base,ipb,m><<< grid, block, ipb*2*m*sizeof(uint_t) >>>(d_as, d_bs, d_rs);
        //bmulKer<Base,ipb,m><<< grid, block >>>(d_as, d_bs, d_rs);
        bmulKerQ<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
        cudaDeviceSynchronize();
        gpuAssert( cudaPeekAtLastError() );
    }
    
    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));
    
    { // 5. timing instrumentation for One Multiplication
        uint64_t elapsed;
        struct timeval t_start, t_end, t_diff;
        gettimeofday(&t_start, NULL); 
        
        for(int i=0; i<GPU_RUNS_MUL; i++) {
            //bmulKer<Base,ipb,m><<< grid, block >>>(d_as, d_bs, d_rs);
            bmulKerQ<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_MUL;

        gpuAssert( cudaPeekAtLastError() );

        double runtime_microsecs = elapsed;
        //double num_u32_ops = 4.0 * num_instances * m * m * x * x; 
        double num_u32_ops = num_instances * numAd32OpsOfDivInst<uint_t>(m);
        double gigaopsu32  = num_u32_ops / (runtime_microsecs * 1000);

        printf( "N^2 Multiplication of %d-bits Big-Numbers (in base = %d bits) runs %d instances in: \
%lu microsecs, Gu32ops/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , m*x*32, Base::bits, num_instances, elapsed, gigaopsu32, num_instances / runtime_microsecs
              );
    }
    
    cudaMemcpy(h_rs, d_rs, mem_size_nums, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    cudaFree(d_as);
    cudaFree(d_bs);
    cudaFree(d_rs);
}

/****************************/
/***** Single Division ******/
/****************************/

template<class Base, uint32_t m>  // m is the size of the big word in Base::uint_t units
void gpuDiv ( uint32_t num_instances
            , typename Base::uint_t* u
            , typename Base::uint_t* v
            , typename Base::uint_t* h_quo
            , typename Base::uint_t* h_rem
) {
    using uint_t = typename Base::uint_t;
    //using carry_t= typename Base::carry_t;
    
    uint_t* d_as;
    uint_t* d_bs;
    uint_t* d_quo;
    uint_t* d_rem;
    uint32_t mem_size_nums = num_instances * m * sizeof(uint_t);

    // 1. allocate device memory
    cudaMalloc((void**)&d_as, mem_size_nums);
    cudaMalloc((void**)&d_bs, mem_size_nums);
    cudaMalloc((void**)&d_quo, mem_size_nums);
    cudaMalloc((void**)&d_rem, mem_size_nums);
 
    // 2. copy host memory to device
    cudaMemcpy(d_as, u, mem_size_nums, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs, v, mem_size_nums, cudaMemcpyHostToDevice);

    // 3. kernel dimensions
    const uint32_t q = Q; // use 8 for A4500 

    dim3 block( m/q, 1, 1 );
    dim3 grid ( num_instances, 1, 1);
    uint32_t sh_mem = 2 * m * sizeof(uint_t);

    if (sh_mem >= 64000) { // maximize the amount of shared memory for the kernel
        cudaFuncSetAttribute(divShinvKer<Base, m, q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98000);
    }    
    
    // 4. dry run
    {
        divShinvKer<Base, m, q><<< grid, block, sh_mem >>>(d_as, d_bs, d_quo, d_rem);
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
            divShinvKer<Base, m, q><<< grid, block, sh_mem >>>(d_as, d_bs, d_quo, d_rem);
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
    
    cudaMemcpy(h_quo, d_quo, mem_size_nums, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rem, d_rem, mem_size_nums, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_as);
    cudaFree(d_bs);
    cudaFree(d_quo);
    cudaFree(d_rem);
}

template<class Base, uint32_t m>  // m is the size of the big word in Base::uint_t units
void gpuGCD ( uint32_t num_instances
            , typename Base::uint_t* u
            , typename Base::uint_t* v
            , typename Base::uint_t* r
) {
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
    const uint32_t q = Q; // use 8 for A4500 

    dim3 block( m/q, 1, 1 );
    dim3 grid ( num_instances, 1, 1);
    uint32_t sh_mem = 2 * m * sizeof(uint_t);

    if (sh_mem >= 64000) { // maximize the amount of shared memory for the kernel
        cudaFuncSetAttribute(divShinvKer<Base, m, q>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98000);
    }    
    
    // 4. dry run
    {
        gcd<Base, m, q><<< grid, block, sh_mem >>>(d_as, d_bs, d_rs);
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
        
        for(int i=0; i<GPU_RUNS_GCD; i++) {
            gcd<Base, m, q><<< grid, block, sh_mem >>>(d_as, d_bs, d_rs);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_GCD;

        gpuAssert( cudaPeekAtLastError() );

        double runtime_microsecs = elapsed; 
        double num_u32_ops = num_instances * numAd32OpsOfDivInst<uint_t>(m);
        double gigaopsu32 = num_u32_ops / (runtime_microsecs * 1000);

        printf( "GCD on %d-bit Big-Numbers (base u%d) runs %d instances in: \
%lu microsecs, Gu32ops/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , m*x*32, Base::bits, num_instances, elapsed, gigaopsu32, num_instances / runtime_microsecs
              );
    }
    
    cudaMemcpy(r, d_rs, mem_size_nums, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_as);
    cudaFree(d_bs);
    cudaFree(d_rs);
}


template<typename Base, int m>  // m is the size of the big word in u32 units
void testNsqMul(  int num_instances
                , uint64_t* h_as_64
                , uint64_t* h_bs_64
                , uint64_t* h_rs_gmp_64
                , uint64_t* h_rs_our_64
                , uint32_t  with_validation
                ) {
                
    using uint_t = typename Base::uint_t;
    
    uint_t *h_as = (uint_t*)h_as_64; 
    uint_t *h_bs = (uint_t*)h_bs_64;
    uint_t *h_rs_our = (uint_t*)h_rs_our_64;
    uint32_t *h_rs_gmp_32 = (uint32_t*)h_rs_gmp_64;

    if(with_validation)
        gmpMultiply<m>(num_instances, (uint32_t*)h_as, (uint32_t*)h_bs, h_rs_gmp_32);
        
    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));
    
    gpuMultiply<Base, m/x>(num_instances, h_as, h_bs, h_rs_our);

    if(with_validation)
        validateExact<uint32_t>(h_rs_gmp_32, (uint32_t*)h_rs_our, num_instances*m);
}

template<class Base, int m>
void testDivision( int num_instances
                 , typename Base::uint_t* gmp_quo
                 , typename Base::uint_t* gmp_rem
                 , typename Base::uint_t* our_quo
                 , typename Base::uint_t* our_rem
                 , uint32_t with_validation
) {
    using uint_t = typename Base::uint_t;
    
    
    
    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));
    
    uint_t uPrec = (m/x)-2;
    uint_t vPrec = (rand() % (uPrec/2)) +1;
    
    uint_t* u = randBigInt<uint_t>(uPrec, m/x, num_instances);
    uint_t* v = randBigInt<uint_t>(vPrec, m/x, num_instances);

    if(with_validation)
        gmpDiv<uint_t, m/x>(num_instances, u, v, gmp_quo, gmp_rem);

    gpuDiv<Base, m/x>(num_instances, u, v, our_quo, our_rem);

    if(with_validation) {
        validateExact(gmp_quo, our_quo, gmp_rem, our_rem, num_instances*(m/x));
    }
}

template<class Base, int m>
void testGCD( int num_instances
            , typename Base::uint_t* res_gmp
            , typename Base::uint_t* res_our
            , uint32_t with_validation
) {
    using uint_t = typename Base::uint_t;
    
    
    
    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));
    
    uint_t uPrec = (m/x)-Q;
    uint_t vPrec = 3;
    
    uint_t* u = randBigInt<uint_t>(uPrec, m/x, num_instances);
    uint_t* v = randBigInt<uint_t>(vPrec, m/x, num_instances);

    if(with_validation)
        gmpGCD<uint_t, m/x>(num_instances, u, v, res_gmp);

    gpuGCD<Base, m/x>(num_instances, u, v, res_our);

    if(with_validation) {
        validateExact(res_gmp, res_our, num_instances*(m/x));
    }
}


/////////////////////////////////////////////////////////
// Main program that runs test suits
/////////////////////////////////////////////////////////

template<typename Base>
void runNaiveMuls(uint64_t total_work) {
    using uint_t = typename Base::uint_t;
    uint64_t *h_as, *h_bs, *h_rs_gmp, *h_rs_our;
    mkRandArrays<32,32>( total_work/32, &h_as, &h_bs, &h_rs_gmp, &h_rs_our );

#if 1
    // testNsqMul<Base, 8192>( total_work/8192, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base, 4096>( total_work/4096, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base, 2048>( total_work/2048, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base, 1024>( total_work/1024, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );

    testNsqMul<Base,  512>( total_work/512,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,  256>( total_work/256,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,  128>( total_work/128,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,   64>( total_work/64,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,   32>( total_work/32,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testNsqMul<Base,   16>( total_work/16,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
#endif
    free(h_as);
    free(h_bs);
    free(h_rs_gmp);
    free(h_rs_our);    
}
 

template<typename Base>
void runDivisions(uint64_t total_work) {

    using uint_t = typename Base::uint_t;
    uint_t *gmp_quo, *gmp_rem, *our_quo, *our_rem;

    gmp_quo = (uint_t*)calloc(total_work, sizeof(uint_t));
    gmp_rem = (uint_t*)calloc(total_work, sizeof(uint_t));
    our_quo = (uint_t*)calloc(total_work, sizeof(uint_t));
    our_rem = (uint_t*)calloc(total_work, sizeof(uint_t));

    testDivision<Base, 8192>( total_work/8192, gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base, 4096>( total_work/4096, gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base, 2048>( total_work/2048, gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base, 1024>( total_work/1024, gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base,  512>( total_work/512,  gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base,  256>( total_work/256,  gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base,  128>( total_work/128,  gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base,   64>( total_work/64,   gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base,   32>( total_work/32,   gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    testDivision<Base,   16>( total_work/16,   gmp_quo, gmp_rem, our_quo, our_rem, WITH_VALIDATION );
    
    free(gmp_quo);
    free(gmp_rem);
    free(our_quo);
    free(our_rem);
}

template<typename Base>
void runGCDs(uint64_t total_work) {

    using uint_t = typename Base::uint_t;
    uint_t *gmp_rs, *our_rs;

    gmp_rs = (uint_t*)calloc(total_work, sizeof(uint_t));
    our_rs = (uint_t*)calloc(total_work, sizeof(uint_t));

    // testGCD<Base, 2048>( total_work/2048, gmp_rs, our_rs, WITH_VALIDATION );
    // testGCD<Base, 4096>( total_work/4096, gmp_rs, our_rs, WITH_VALIDATION );
    // testGCD<Base, 1024>( total_work/1024, gmp_rs, our_rs, WITH_VALIDATION );
    // testGCD<Base,  512>( total_work/512,  gmp_rs, our_rs, WITH_VALIDATION );
    // testGCD<Base,  256>( total_work/256,  gmp_rs, our_rs, WITH_VALIDATION );
    // testGCD<Base,  128>( total_work/128,  gmp_rs, our_rs, WITH_VALIDATION );
    // testGCD<Base,   64>( total_work/64,   gmp_rs, our_rs, WITH_VALIDATION );
    // testGCD<Base,   32>( total_work/32,   gmp_rs, our_rs, WITH_VALIDATION );
    testGCD<Base,   16>( total_work/16,   gmp_rs, our_rs, WITH_VALIDATION );

    free(gmp_rs);
    free(our_rs);
}

 
int main (int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s <batch-size>\n", argv[0]);
        exit(1);
    }
        
    const int total_work = atoi(argv[1]);

    // cudaSetDevice(1);

    {   // 32bit integer elements
        // runNaiveMuls<U32bits>(total_work);

     //  runDivisions<U32bits>(total_work);
    }

    {   // 64bit integer elements
        // runNaiveMuls<U64bits>(total_work);

        runDivisions<U64bits>(total_work);
    }

    // {   // GCD computation
    //     runGCDs<U32bits>(total_work);
    // }
}
