#ifndef HELPER
#define HELPER

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <gmp.h>
#include <assert.h>

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

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

template<typename uint_t>
uint64_t numAd32OpsOfMultInst(uint32_t m0) {
    uint32_t m = m0*sizeof(uint_t) / 4;
    uint32_t lgm = 0, mm = m;
    for( ; mm > 1; mm >>= 1) lgm++;
    //printf("Log %d is %d\n", m, lgm);
    return 300 * m * lgm;
}

template<typename uint_t>
uint64_t numAd32OpsOfClassicalMultInst(uint32_t m0) {
    uint32_t m = m0*sizeof(uint_t) / 4;
    return m*m;
}

/**
 * Number of giga-u32-bit unit operations of division.
 */
template<typename uint_t>
uint64_t numAd32OpsOfDivInst(uint32_t m0) {
    uint32_t m = m0*sizeof(uint_t) / 4;
    return 3*m*m;
}

/**
 * Number of operations of GCD.
 */
template<typename uint_t>
uint64_t numAd32OpsOfGCDInst(uint32_t m0) {
    uint32_t m = m0*sizeof(uint_t) / 4;
    return (0.58*m*32)*3*m*m;
}

/**
 * Initialize the `data` array, which has `size` elements:
 * frac% of them are NaNs and (1-frac)% are random values.
 * 
 */
void randomMask(char* data, uint64_t size, float frac) {
    for (uint64_t i = 0; i < size; i++) {
        float r = rand() / (float)RAND_MAX;
        data[i] = (r >= frac) ? 1 : 0;
    }
}

// error for matmul: 0.02
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

/**
 * Zeros a bigint and sets the the digit at index idx to d
 */
template<class uint_t>
void setIdx( uint_t* u
           , uint_t d
           , uint32_t idx
           , uint32_t m
) {
    for (int i = 0; i < m; i++) {
        u[i] = 0;
    }
    u[idx] = d;
}

/**
 * Returns multiple bigints of precision prec where all digits are set to x
 */
template<int m>
uint32_t* ourMkSet( uint32_t num_instances
                  , uint32_t* as
                  , uint32_t d
                  , uint32_t idx
) {
    for (int j = 0; j < num_instances; j++) {
        setIdx<uint32_t>(&as[j*m], d, idx, m);
    }
    return as;
}

#define GMP_ORDER   (-1)

// creates `num_instances` of big-integers consisting of `m` u32 words
// using gmp random functionality
template<int m>
void gmpMkRandom(uint32_t num_instances, uint32_t* as, gmp_randstate_t rnd) {
    uint32_t* it_as = as;

    for(int i=0; i<num_instances; i++, it_as += m) {
        mpz_t a;        
        mpz_init(a);    
        mpz_rrandomb(a, rnd, m*32);
        
        for(int k = 0; k < m; k++) {
            it_as[k] = 0;
        }
        
        //mpz_import (a, m, 1, sizeof(uint32_t), 0, 0, it_as);
        size_t countp = 0;
        mpz_export (it_as, &countp, GMP_ORDER, sizeof(uint32_t), 0, 0, a);
        
        if(countp != m) {
            printf("Possible error at iteration i=%d, countp=%ld instead of %d!\n", i, countp, m);
        }
        
        //if(i==0) { std::cout << a->toString() << std::endl; }
        if(i==0) gmp_printf ("%s is an mpz %Zd\n", "here", a);
    }
}

template<uint32_t m>
void gmpAddMulOnce(bool is_add, uint32_t* inst_as, uint32_t* inst_bs, uint32_t* inst_rs) {
    uint32_t buff[4*m];
    mpz_t a; mpz_t b; mpz_t r;        
    mpz_init(a); mpz_init(b); mpz_init(r);

    mpz_import(a, m, GMP_ORDER, sizeof(uint32_t), 0, 0, inst_as);
    mpz_import(b, m, GMP_ORDER, sizeof(uint32_t), 0, 0, inst_bs);

    if(is_add) {
        mpz_add(r, a, b);
    } else {
        mpz_mul(r, a, b);
    }
        
    size_t countp = 0;
    mpz_export (buff, &countp, GMP_ORDER, sizeof(uint32_t), 0, 0, r);
        
    for(int j=0; j<m; j++) {
        inst_rs[j] = buff[j];
    }      
    for(int j=countp; j < m; j++) {
        inst_rs[j] = 0;
    }
}

template<class uint_t, uint32_t m>
void gmpQuoOnce( uint_t* inst_as
               , uint_t* inst_bs
               , uint_t* inst_rs
) {
    uint_t buff[4*m];
    mpz_t a; mpz_t b; mpz_t r;        
    mpz_init(a); mpz_init(b); mpz_init(r);

    mpz_import(a, m, GMP_ORDER, sizeof(uint_t), 0, 0, inst_as);
    mpz_import(b, m, GMP_ORDER, sizeof(uint_t), 0, 0, inst_bs);

    mpz_fdiv_q(r, a, b);
        
    size_t countp = 0;
    mpz_export (buff, &countp, GMP_ORDER, sizeof(uint_t), 0, 0, r);
        
    for(int j=0; j<m; j++) {
        inst_rs[j] = buff[j];
    }      
    for(int j=countp; j < m; j++) {
        inst_rs[j] = 0;
    }
}

template<class uint_t, uint32_t m>
void gmpDivOnce( uint_t* inst_as
               , uint_t* inst_bs
               , uint_t* inst_qs
               , uint_t* inst_rs
) {
    uint_t buff[4*m];
    uint_t buff2[4*m];
    mpz_t a; mpz_t b; mpz_t q; mpz_t r;        
    mpz_init(a); mpz_init(b); mpz_init(r);

    mpz_import(a, m, GMP_ORDER, sizeof(uint_t), 0, 0, inst_as);
    mpz_import(b, m, GMP_ORDER, sizeof(uint_t), 0, 0, inst_bs);

    __gmpz_fdiv_qr(r, q, a, b);
        
    size_t countp = 0;
    mpz_export (buff, &countp, GMP_ORDER, sizeof(uint_t), 0, 0, q);
    mpz_export (buff2, &countp, GMP_ORDER, sizeof(uint_t), 0, 0, r);

        
    for(int j=0; j<m; j++) {
        inst_qs[j] = buff[j];
        inst_rs[j] = buff2[j];
    }      
    for(int j=countp; j < m; j++) {
        inst_rs[j] = 0;
    }
}

/**
 * A wrapper for the GMP div operation
 */
template<class uint_t, uint32_t m>
void gmpGCDOnce( uint_t* inst_as
               , uint_t* inst_bs
               , uint_t* inst_rs
) {
    uint_t buff[4*m];
    mpz_t a; mpz_t b; mpz_t r;   
    mpz_init(a); mpz_init(b); mpz_init(r);

    mpz_import(a, m, GMP_ORDER, sizeof(uint_t), 0, 0, inst_as);
    mpz_import(b, m, GMP_ORDER, sizeof(uint_t), 0, 0, inst_bs);

    mpz_gcd(r, a, b);
        
    size_t count = 0;
    mpz_export (buff, &count, GMP_ORDER, sizeof(uint_t), 0, 0, r);
        
    for(int j=0; j<m; j++) {
        inst_rs[j] = buff[j];
    }      
    for(int j=count; j < m; j++) {
        inst_rs[j] = 0;
    }
}

/****************************/
/***  support routines    ***/
/****************************/
void cuda_check(cudaError_t status, const char *action=NULL, const char *file=NULL, int32_t line=0) {
  // check for cuda errors

  if(status!=cudaSuccess) {
    printf("CUDA error occurred: %s\n", cudaGetErrorString(status));
    if(action!=NULL)
      printf("While running %s   (file %s, line %d)\n", action, file, line);
    exit(1);
  }
}

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)

#endif // HELPER
