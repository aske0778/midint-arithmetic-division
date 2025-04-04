
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

void set(uint32_t* u, uint32_t d, uint32_t m)
{
    for (int i = 0; i < m; i++)
    {
        u[i] = 0;
    }
    u[0] = d;
}

void div_gmp(uint32_t* u, uint32_t* v, uint32_t* q, uint32_t* r, uint32_t m)
{
    set(q, 0, m);
    set(r, 0, m);

    mpz_t a;
    mpz_init(a);
    mpz_import(a, m, -1, sizeof(uint32_t), 0, 0, u);
    mpz_t b;
    mpz_init(b);
    mpz_import(b, m, -1, sizeof(uint32_t), 0, 0, v);
    mpz_t c;
    mpz_init(c);
    mpz_t d;
    mpz_init(d);

    mpz_div(c, a, b);
    mpz_mul(d, b, c);
    mpz_sub(b, a, d);

    mpz_export(q, NULL, -1, sizeof(uint32_t), 0, 0, c);
    mpz_export(r, NULL, -1, sizeof(uint32_t), 0, 0, b);
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
    mpz_clear(d);
}

uint32_t* randBigInt(uint32_t prec, uint32_t m, uint32_t num_instances)
{
    uint32_t* u = (uint32_t*)calloc(m*num_instances, sizeof(uint32_t));
    for (int j = 0; j < num_instances; j++){
        for (int i = 0; i < prec; i++)
        {
            u[i + (j * m)] = (uint32_t)rand();
        }
    }
    return u;
}
