#include <gmp.h>

#define GMP_ORDER   (-1)

/**
 * Print the last cuda device error
 */
int gpuAssert( cudaError_t code )
{
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

/**
 * Prints a bigint to stdout
 */
template<class uint_t>
void prnt( const char *str
         , uint_t *u
         , uint32_t m
) {
    printf("%s: [", str);
    for (int i = 0; i < m; i++)
    {
        printf("%u", u[i]);
        // printf("%" PRIu64 ", ", u[i]);
        if (i < m - 1)
            printf(", ");
    }
    printf("]\n");
}

template<class uint_t>
void prnt64( const char *str
         , uint_t *u
         , uint32_t m
) {
    printf("%s: [", str);
    for (int i = 0; i < m; i++)
    {
        printf("%llu", u[i]);
        // printf("%" PRIu64 ", ", u[i]);
        if (i < m - 1)
            printf(", ");
    }
    printf("]\n");
}

/**
 * Prints 3 indices of a bigint around
 * the index i to stdout
 */
template<class uint_t>
void printSlice( uint_t* u
               , char name
               , int i
               , uint32_t m
) {
    int min = i-3 < 0 ? 0 : i-3;
    int max = i+3 > m ? m : i+3;

    printf("%c[%u-%u]: [", name, min, max);
    for (int i = min; i < max; i++) {
        printf("%u, ", u[i]);
    }
    printf("]\n");
}

/**
 * Zeros a bigint and sets the least significant index to d
 */
template<class uint_t>
void set( uint_t* u
        , uint_t d
        , uint32_t m
) {
    for (int i = 0; i < m; i++) {
        u[i] = 0;
    }
    u[0] = d;
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
 * A wrapper for the GMP implementation of division
 */
template<class uint_t>
void div_gmp( uint_t* u
            , uint_t* v
            , uint_t* q
            , uint_t* r
            , uint32_t m
) {
    set<uint_t>(q, 0, m);
    set<uint_t>(r, 0, m);

    mpz_t a;
    mpz_init(a);
    mpz_import(a, m, -1, sizeof(uint_t), 0, 0, u);
    mpz_t b;
    mpz_init(b);
    mpz_import(b, m, -1, sizeof(uint_t), 0, 0, v);
    mpz_t c;
    mpz_init(c);
    mpz_t d;
    mpz_init(d);

    mpz_div(c, a, b);
    mpz_mul(d, b, c);
    mpz_sub(b, a, d);

    mpz_export(q, NULL, -1, sizeof(uint_t), 0, 0, c);
    mpz_export(r, NULL, -1, sizeof(uint_t), 0, 0, b);
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
    mpz_clear(d);
}


/**
 * A wrapper for the GMP implementation of multiplication
 */
 template<class uint_t>
void mul_gmp( uint_t* u
    , uint_t* v
    , uint_t* r
    , uint32_t m
) {
    set<uint_t>(r, 0, m);

    mpz_t a;
    mpz_init(a);
    mpz_import(a, m, -1, sizeof(uint_t), 0, 0, u);
    mpz_t b;
    mpz_init(b);
    mpz_import(b, m, -1, sizeof(uint_t), 0, 0, v);
    mpz_t c;
    mpz_init(c);


    mpz_mul(c, a, b);

    mpz_export(r, NULL, -1, sizeof(uint_t), 0, 0, c);
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
}

/**
 * Returns a randomly generated bigint of precision prec
 */
template<class uint_t>
uint_t* randBigInt( uint32_t prec, uint32_t m )
{
    uint_t* u = (uint_t*)calloc(m, sizeof(uint_t));

    for (int i = 0; i < prec; i++)
    {

        uint_t r = (uint_t)rand();
        u[i] = r + (r == 0);
      //  u[i] = (uint_t)rand() ;
      //  u[i] = ((uint_t)rand() % 5) + 1;
        // if (rand() % 2 == 0) {
        //     u[i] = 1;
        // } else {
        //     u[i] = (uint_t)rand();
        // }
    }
    return u;
}

/**
 * Returns multiple randomly generated bigint of precision prec
 */
template<class uint_t>
uint_t* randBigInt( uint32_t prec
                  , uint32_t m
                  , uint32_t num_instances
) {
    uint_t* u = (uint_t*)calloc(m*num_instances, sizeof(uint_t));
    for (int j = 0; j < num_instances; j++){
        for (int i = 0; i < prec; i++)
        {
            u[j*m + i] = (uint_t)rand();
        }
    }
    return u;
}

template<class uint_t>
uint_t* randBigIntPrecs( uint32_t maxPrec
                  , uint32_t m
                  , uint32_t num_instances
) {
    uint_t* u = (uint_t*)calloc(m*num_instances, sizeof(uint_t));
    for (int j = 0; j < num_instances; j++){
        for (int i = 0; i < (rand() % (maxPrec/2)) + 2; i++)
        {
            u[j*m + i] = (uint_t)rand();
        }
    }
    return u;
}

/**
 * Returns multiple bigints of precision prec where all digits are set to x
 */
template<class uint_t>
uint_t* setBigInt(  uint_t d
                  , uint32_t idx
                  , uint32_t m
                  , uint32_t num_instances
) {
    uint_t* u = (uint_t*)calloc(m*num_instances, sizeof(uint_t));
    for (int j = 0; j < num_instances; j++) {
        setIdx<uint_t>(&u[j*m], d, idx, m);
    }
    return u;
}

template<typename uint_t>
uint64_t numAd32OpsOfMultInst(uint32_t m0) {
    uint32_t m = m0*sizeof(uint_t) / 4;
    return m*m;
}

/**
 * Number of operations of division.
 */
template<typename uint_t>
uint64_t numAd32OpsOfDivInst(uint32_t m0) {
    uint32_t m = m0*sizeof(uint_t) / 4;
    return 7*m*m;
}

/**
 * Number of operations of GCD.
 */
template<typename uint_t>
uint64_t numAd32OpsOfGCDInst(uint32_t m0) {
    uint32_t m = m0*sizeof(uint_t) / 4;
    return 6*7*m*m;
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

/**
 * A wrapper for the GMP quo operation
 */
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

/**
 * A wrapper for the GMP div operation
 */
template<class uint_t, uint32_t m>
void gmpDivOnce( uint_t* inst_as
               , uint_t* inst_bs
               , uint_t* inst_quo
               , uint_t* inst_rem
) {
    uint_t buffq[4*m];
    uint_t buffr[4*m];
    mpz_t a; mpz_t b; mpz_t q; mpz_t r;   
    mpz_init(a); mpz_init(b); mpz_init(q); mpz_init(r);

    mpz_import(a, m, GMP_ORDER, sizeof(uint_t), 0, 0, inst_as);
    mpz_import(b, m, GMP_ORDER, sizeof(uint_t), 0, 0, inst_bs);

    mpz_fdiv_qr(q, r, a, b);
        
    size_t countq = 0;
    size_t countr = 0;
    mpz_export (buffq, &countq, GMP_ORDER, sizeof(uint_t), 0, 0, q);
    mpz_export (buffr, &countr, GMP_ORDER, sizeof(uint_t), 0, 0, r);
        
    for(int j=0; j<m; j++) {
        inst_quo[j] = buffq[j];
        inst_rem[j] = buffr[j];
    }      
    for(int j=countq; j < m; j++) {
        inst_quo[j] = 0;
    }
    for(int j=countr; j < m; j++) {
        inst_rem[j] = 0;
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

template<int m>
void gmpMultiply(int num_instances, uint32_t* as, uint32_t* bs, uint32_t* rs) {
    uint32_t* it_as = as;
    uint32_t* it_bs = bs;
    uint32_t* it_rs = rs;
        
    for(int i=0; i<num_instances; i++) {
        gmpAddMulOnce<m>(false, it_as, it_bs, it_rs);
        it_as += m; it_bs += m; it_rs += m;
    }
}

/**
 * A wrapper to call GMP quo for a number of instances
 */
template<class uint_t, int m>
void gmpQuo( int num_instances
           , uint_t* as
           , uint_t* bs
           , uint_t* rs
) {
    uint_t* it_as = as;
    uint_t* it_bs = bs;
    uint_t* it_rs = rs;
        
    for(int i=0; i<num_instances; i++) {
        gmpQuoOnce<uint_t, m>(it_as, it_bs, it_rs);
        it_as += m; it_bs += m; it_rs += m;
    }
}

/**
 * A wrapper to call GMP div for a number of instances
 */
template<class uint_t, int m>
void gmpDiv( int num_instances
           , uint_t* as
           , uint_t* bs
           , uint_t* quo
           , uint_t* rem
) {
    uint_t* it_as = as;
    uint_t* it_bs = bs;
    uint_t* it_quo = quo;
    uint_t* it_rem = rem;
        
    for(int i=0; i<num_instances; i++) {
        gmpDivOnce<uint_t, m>(it_as, it_bs, it_quo, it_rem);
        it_as += m; it_bs += m; it_quo += m; it_rem += m;
    }
}

/**
 * A wrapper to call GMP div for a number of instances
 */
template<class uint_t, int m>
void gmpGCD( int num_instances
           , uint_t* as
           , uint_t* bs
           , uint_t* rs
) {
    uint_t* it_as = as;
    uint_t* it_bs = bs;
    uint_t* it_rs = rs;
        
    for(int i=0; i<num_instances; i++) {
        gmpGCDOnce<uint_t, m>(it_as, it_bs, it_rs);
        it_as += m; it_bs += m; it_rs += m;
    }
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

/**
 * Validates exactly A == B for quo and rem
 */
template<class T>
bool validateExact(T* Q1, T* Q2, T* R1, T* R2, uint64_t sizeAB){
    for(uint64_t i = 0; i < sizeAB; i++) {
        if ( Q1[i] != Q2[i] ) {
            printf("INVALID RESULT at quotient index %lu: %u vs %u\n", i, Q1[i], Q2[i]);
            return false;
        }
        if ( R1[i] != R2[i] ) {
            printf("INVALID RESULT at remainder index %lu: %u vs %u\n", i, R1[i], R2[i]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}