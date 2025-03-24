#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "ker-division.cu.h"
#include "helper.h"

//#include "goldenSeq.h"

//#define WITH_INT_128 1


using namespace std;

#define GPU_RUNS_ADD    50
#define GPU_RUNS_MUL    25
#define ERR         0.000005

#define WITH_VALIDATION 1


template<uint32_t m>
void gmpDivOnce(uint32_t* inst_as, uint32_t* inst_bs, uint32_t* inst_rs) {
    uint32_t buff[4*m];
    mpz_t a; mpz_t b; mpz_t r;        
    mpz_init(a); mpz_init(b); mpz_init(r);

    mpz_import(a, m, GMP_ORDER, sizeof(uint32_t), 0, 0, inst_as);
    mpz_import(b, m, GMP_ORDER, sizeof(uint32_t), 0, 0, inst_bs);

    mpz_cdiv_q(r, a, b);
        
    size_t countp = 0;
    mpz_export (buff, &countp, GMP_ORDER, sizeof(uint32_t), 0, 0, r);
        
    for(int j=0; j<m; j++) {
        inst_rs[j] = buff[j];
    }      
    for(int j=countp; j < m; j++) {
        inst_rs[j] = 0;
    }
}


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
void gpuDiv ( int num_instances
            , typename Base::uint_t* h_as
            , typename Base::uint_t* h_bs
            , typename Base::uint_t* h_rs
            ) 
{
    using uint_t = typename Base::uint_t;
    //using carry_t= typename Base::carry_t;
    
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
    
    dim3 block( ipb * (m_lft/q), 1, 1);
    dim3 grid ( (num_instances + ipb - 1)/ipb, 1, 1);
    
    // 4. dry run
    {
        divShinv<m, q><<<1, m/q>>>((uint32_t*)d_as,(uint32_t*) d_bs, (uint32_t*) d_rs, (uint32_t*) d_rs);
        // divShinv<m,q><<< grid, block >>>(d_as, d_bs, d_rs, d_rs);
        // baddKer<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
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
        
        for(int i=0; i<GPU_RUNS_ADD; i++) {
            divShinv<m,q><<< grid, block >>>((uint32_t*) d_as, (uint32_t*) d_bs, (uint32_t*) d_rs, (uint32_t*) d_rs);
            // baddKer<Base,ipb,m,q><<< grid, block >>>(num_instances, d_as, d_bs, d_rs);
        }
        
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS_ADD;

        gpuAssert( cudaPeekAtLastError() );

        double runtime_microsecs = elapsed; 
        double bytes_accesses = 3.0 * num_instances * m * sizeof(uint_t);  
        double gigabytes = bytes_accesses / (runtime_microsecs * 1000);

        printf( "Division on %d-bit Big-Numbers (base u%d) runs %d instances in: \
%lu microsecs, GB/sec: %.2f, Mil-Instances/sec: %.2f\n"
              , m*x*32, Base::bits, num_instances, elapsed, gigabytes, num_instances / runtime_microsecs
              );
    }
    
    cudaMemcpy(h_rs, d_rs, mem_size_nums, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_as);
    cudaFree(d_bs);
    cudaFree(d_rs);
}


template<int m>
void gmpDiv(int num_instances, uint32_t* as, uint32_t* bs, uint32_t* rs) {
    uint32_t* it_as = as;
    uint32_t* it_bs = bs;
    uint32_t* it_rs = rs;
        
    for(int i=0; i<num_instances; i++) {
        gmpDivOnce<m>(it_as, it_bs, it_rs);
        it_as += m; it_bs += m; it_rs += m;
    }
}

template<class Base, int m>
void testDivision( int num_instances
                 , uint64_t* h_as_64
                 , uint64_t* h_bs_64
                 , uint64_t* h_rs_gmp_64
                 , uint64_t* h_rs_our_64
                 , uint32_t with_validation
) {
    using uint_t = typename Base::uint_t;
    
    uint_t *h_as = (uint_t*) h_as_64;
    uint_t *h_bs = (uint_t*) h_bs_64;
    uint_t *h_rs_our = (uint_t*) h_rs_our_64;
    uint32_t *h_rs_gmp_32 = (uint32_t*) h_rs_gmp_64;

    const uint32_t x = Base::bits/32;
    assert( (Base::bits >= 32) && (Base::bits % 32 == 0));

    if(with_validation)
        gmpDiv<m>(num_instances, (uint32_t*)h_as, (uint32_t*)h_bs, h_rs_gmp_32);

    gpuDiv<Base, m/x>(num_instances, h_as, h_bs, h_rs_our);

#if 0
    uint32_t querry_instance = 0;
    printf("as[%d]: ", querry_instance);
    printInstance<m>(querry_instance, h_as);
    printf("bs[%d]: ", querry_instance);
    printInstance<m>(querry_instance, h_bs);
    printf("rs_gmp[%d]: ", querry_instance);
    printInstance<m>(querry_instance, h_rs_gmp);
    printf("rs_our[%d]: ", querry_instance);
    printInstance<m>(querry_instance, h_rs_our);
#endif

    if(with_validation)  
        validateExact(h_rs_gmp_32, (uint32_t*)h_rs_our, num_instances*m);
}



/////////////////////////////////////////////////////////
// Main program that runs test suits
/////////////////////////////////////////////////////////
 
template<typename Base>
void runDivisions(uint64_t total_work) {
    uint64_t *h_as, *h_bs, *h_rs_gmp, *h_rs_our;
    mkRandArrays<32,32>( total_work/32, &h_as, &h_bs, &h_rs_gmp, &h_rs_our );
    
#if 1
    // testDivision<Base, 4096>( total_work/4096, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    // testDivision<Base, 2048>( total_work/2048, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    // testDivision<Base, 1024>( total_work/1024, h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    // testDivision<Base,  512>( total_work/512,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    // testDivision<Base,  256>( total_work/256,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    // testDivision<Base,  128>( total_work/128,  h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    // testDivision<Base,   64>( total_work/64,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    // testDivision<Base,   32>( total_work/32,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
    testDivision<Base,   16>( total_work/16,   h_as, h_bs, h_rs_gmp, h_rs_our, WITH_VALIDATION );
#endif
    free(h_as);
    free(h_bs);
    free(h_rs_gmp);
    free(h_rs_our);
}
 
 
int main (int argc, char * argv[]) {
    if (argc != 2) {
        printf("Usage: %s <batch-size>\n", argv[0]);
        exit(1);
    }
        
    const int total_work = atoi(argv[1]);

    cudaSetDevice(1);
    runDivisions<U64bits>(total_work);
}
