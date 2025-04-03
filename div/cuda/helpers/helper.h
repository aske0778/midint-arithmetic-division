#include <gmp.h>

void prnt(const char *str, uint32_t *u, uint32_t m)
{
    printf("%s: [", str);
    for (int i = 0; i < m; i++)
    {
        printf("%u", u[i]);
        if (i < m - 1)
            printf(", ");
    }
    printf("]\n");
}

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

template<class uint_t>
void set( uint_t* u
        , uint32_t d
        , uint32_t m
) {
    for (int i = 0; i < m; i++)
    {
        u[i] = 0;
    }
    u[0] = d;
}

template<class uint_t>
void div_gmp( uint_t* u
            , uint_t* v
            , uint_t* q
            , uint_t* r
            , uint32_t m
) {
    set(q, 0, m);
    set(r, 0, m);

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


template<class uint_t>
uint_t* randBigInt( uint32_t prec
                  , uint32_t m
                  , uint32_t num_instances
) {
    uint_t* u = (uint_t*)calloc(m*num_instances, sizeof(uint_t));
    for (int j = 0; j < num_instances; j++){
        for (int i = 0; i < prec; i++)
        {
            u[i + (j * m)] = (uint_t)rand();
        }
    }
    return u;
}

#define GMP_ORDER   (-1)


template<uint32_t m>
void gmpQuoOnce(uint32_t* inst_as, uint32_t* inst_bs, uint32_t* inst_rs) {
    uint32_t buff[4*m];
    mpz_t a; mpz_t b; mpz_t r;        
    mpz_init(a); mpz_init(b); mpz_init(r);

    mpz_import(a, m, GMP_ORDER, sizeof(uint32_t), 0, 0, inst_as);
    mpz_import(b, m, GMP_ORDER, sizeof(uint32_t), 0, 0, inst_bs);

    mpz_fdiv_q(r, a, b);
        
    size_t countp = 0;
    mpz_export (buff, &countp, GMP_ORDER, sizeof(uint32_t), 0, 0, r);
        
    for(int j=0; j<m; j++) {
        inst_rs[j] = buff[j];
    }      
    for(int j=countp; j < m; j++) {
        inst_rs[j] = 0;
    }
}

template<uint32_t m>
void gmpDivOnce(uint32_t* inst_as, uint32_t* inst_bs, uint32_t* inst_quo, uint32_t* inst_rem) {
    uint32_t buffq[4*m];
    uint32_t buffr[4*m];
    mpz_t a; mpz_t b; mpz_t q; mpz_t r;   
    mpz_init(a); mpz_init(b); mpz_init(q); mpz_init(r);

    mpz_import(a, m, GMP_ORDER, sizeof(uint32_t), 0, 0, inst_as);
    mpz_import(b, m, GMP_ORDER, sizeof(uint32_t), 0, 0, inst_bs);

    mpz_fdiv_qr(q, r, a, b);
        
    size_t countq = 0;
    size_t countr = 0;
    mpz_export (buffq, &countq, GMP_ORDER, sizeof(uint32_t), 0, 0, q);
    mpz_export (buffr, &countr, GMP_ORDER, sizeof(uint32_t), 0, 0, r);
        
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

template<int m>
void gmpQuo(int num_instances, uint32_t* as, uint32_t* bs, uint32_t* rs) {
    uint32_t* it_as = as;
    uint32_t* it_bs = bs;
    uint32_t* it_rs = rs;
        
    for(int i=0; i<num_instances; i++) {
        gmpQuoOnce<m>(it_as, it_bs, it_rs);
        it_as += m; it_bs += m; it_rs += m;
    }
}

template<int m>
void gmpDiv(int num_instances, uint32_t* as, uint32_t* bs, uint32_t* quo, uint32_t* rem) {
    uint32_t* it_as = as;
    uint32_t* it_bs = bs;
    uint32_t* it_quo = quo;
    uint32_t* it_rem = rem;
        
    for(int i=0; i<num_instances; i++) {
        gmpDivOnce<m>(it_as, it_bs, it_quo, it_rem);
        it_as += m; it_bs += m; it_quo += m; it_rem += m;
    }
}

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

uint32_t* randBigInt(uint32_t prec, uint32_t m)
{
    uint32_t* u = (uint32_t*)calloc(m, sizeof(uint32_t));

    for (int i = 0; i < prec; i++)
    {
        u[i] = (uint32_t)rand();
    }
    return u;
}
