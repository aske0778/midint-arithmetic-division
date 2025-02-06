#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>

typedef uint32_t digit_t;
typedef digit_t *bigint_t;
typedef uint32_t prec_t;
typedef int bool;

/**
 * @brief Returns the minimum of two integers
 */
int min(int a, int b)
{
    return (a < b) ? a : b;
}

/**
 * @brief Returns the maximum of two integers
 */
int max(int a, int b)
{
    return (a > b) ? a : b;
}

/**
 * @brief Checks if a bigint_t is equal to zero
 */
bool ez(bigint_t u, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        if (u[i] != 0)
        {
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Returns a < b for two bigint_ts
 */
bool lt(bigint_t a, bigint_t b, prec_t m) {
    for (int i = m-1; i >= 0; i--) {
        if (a[i] < b[i]) {
            return 1;
        }
        else if (a[i] > b[i])
        {
            return 0;
        }
    }
    return 0;
}

/**
 * @brief Checks if two bigint_ts are equal
 */
bool eq(bigint_t u, bigint_t v, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        if (u[i] != v[i])
        {
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Initializes a bigint_t with m digits
 *
 * @param m The number of digits
 * @return bigint_t The initialized bigint_t
 */
bigint_t init(prec_t m)
{
    bigint_t retval = (bigint_t)malloc(m * sizeof(digit_t));
    for (int i = 0; i < m; i++)
    {
        retval[i] = 0;
    }
    return retval;
}

/**
 * @brief Computes the base og bigint_t to the power of n
 *
 * @param n the power to raise the base to
 * @param m the total number of digits in the bigint_t
 */
bigint_t bpow(int n, prec_t m)
{
    bigint_t B = init(m);
    B[0] = 1;
    shift(n, B, B, m);
    return B;
}

/**
 * @brief Sets all digits of a bigint_t to zero except the first digit
 */
void set(bigint_t u, digit_t d, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        u[i] = 0;
    }
    u[0] = d;
}

/**
 * @brief Zeroes all digits of a bigint_t
 */
void zero(bigint_t u, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        u[i] = 0;
    }
}

/**
 * @brief The precision of the bigint_t
 */
prec_t prec(bigint_t u, prec_t m)
{
    prec_t acc = 0;
    for (int i = 0; i < m; i++)
    {
        if (u[i] != 0)
        {
            acc = i + 1;
        }
    }
    return acc + 1;
}

/**
 * @brief Shifts a bigint_t to the left or right depending on sign of n
 *
 * @param n The sign and number of shifts
 * @param u The input bigint_t
 * @param r bigint_t where result is stored
 * @param m The number of digits in u
 */
void shift(int n, bigint_t u, bigint_t r, prec_t m)
{
    if (n >= 0)
    { // Right shift
        for (int i = m - 1; i >= 0; i--)
        {
            int offset = i - n;
            r[i] = (offset >= 0) ? u[offset] : 0;
        }
    }
    else
    { // Left shift
        for (int i = 0; i < m; i++)
        {
            int offset = i - n;
            r[i] = (offset < m) ? u[offset] : 0;
        }
    }
}

/**
 * @brief Uses gmp to add two bigint_ts
 *
 * @param u First bigint_t
 * @param v Second bigint_t
 * @param w bigint_t where result is stored
 * @param m The number of digits in u and v
 */
void add_gmp(bigint_t u, bigint_t v, bigint_t w, prec_t m)
{
    mpz_t a;
    mpz_t b;
    mpz_t r;
    mpz_init(a);
    mpz_init(b);
    mpz_init(r);

    mpz_import(a, m, -1, sizeof(digit_t), 0, 0, u);
    mpz_import(b, m, -1, sizeof(digit_t), 0, 0, v);
    mpz_add(r, a, b);

    set(w, 0, m);
    mpz_export(w, NULL, -1, sizeof(digit_t), 0, 0, r);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(r);
}

/**
 * @brief Uses gmp to subtract two bigint_ts
 *
 * @param u First bigint_t
 * @param v Second bigint_t
 * @param w bigint_t where result is stored
 * @param m The number of digits in u and v
 */
void sub_gmp(bigint_t u, bigint_t v, bigint_t w, prec_t m)
{
    mpz_t a;
    mpz_t b;
    mpz_t r;
    mpz_init(a);
    mpz_init(b);
    mpz_init(r);

    mpz_import(a, m, -1, sizeof(digit_t), 0, 0, u);
    mpz_import(b, m, -1, sizeof(digit_t), 0, 0, v);
    mpz_sub(r, a, b);

    set(w, 0, m);
    mpz_export(w, NULL, -1, sizeof(digit_t), 0, 0, r);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(r);
}

/**
 * @brief Uses gmp to multiply two bigint_ts
 *
 * @param u First bigint_t
 * @param v Second bigint_t
 * @param w bigint_t where result is stored
 * @param m The number of digits in u and v
 */
void mult_gmp(bigint_t u, bigint_t v, bigint_t w, prec_t m) {
    mpz_t a;
    mpz_t b;
    mpz_t r;
    mpz_init(a);
    mpz_init(b);
    mpz_init(r);

    mpz_import(a, m, -1, sizeof(digit_t), 0, 0, u);
    mpz_import(b, m, -1, sizeof(digit_t), 0, 0, v);
    mpz_mul(r, a, b);

    set(w, 0, m);
    mpz_export(w, NULL, -1, sizeof(digit_t), 0, 0, r);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(r);
}

void div_gmp(bigint_t u, bigint_t v, bigint_t q, bigint_t r, prec_t m) {
    set(q, 0, m);
    set(r, 0, m);

    mpz_t a; mpz_init(a); mpz_import(a, m, -1, sizeof(digit_t), 0, 0, u);
    mpz_t b; mpz_init(b); mpz_import(b, m, -1, sizeof(digit_t), 0, 0, v);
    mpz_t c; mpz_init(c);
    mpz_t d; mpz_init(d);

    mpz_div(c, a, b);
    mpz_mul(d, b, c);
    mpz_sub(b, a, d);

    mpz_export(q, NULL, -1, sizeof(digit_t), 0, 0, c);
    mpz_export(r, NULL, -1, sizeof(digit_t), 0, 0, b);
    mpz_clear(a); mpz_clear(b); mpz_clear(c); mpz_clear(d);
}

/**
 * @brief bigint_t division
 * @note Uses long division algorithm
 * https://en.wikipedia.org/wiki/Division_algorithm#Long_division
 * 
 * @param n numerator
 * @param d denominator
 * @param q quotient
 * @param r remainder
 * @param m Total size of bigint_ts
 */
void div(bigint_t n, digit_t d, bigint_t q, prec_t m) {
    // if (d == 0) {
    //     printf("Division by zero\n");
    //     return;
    // }
    // zero(q, m);
    // digit_t r = 0;
    // for (int i = m-1; i >= 0; i--) {
    //     // shift(-1, r, r, m);
    //     r = r << 32;
    //     r += n[i];
    //     if (r >= d) {
    //         r - d;
    //         sub(r, d, r, m);
    //         q[i] = 1;
    //     }
    // }
}

/**
 * @brief Calculates (a * b) rem B^d
 * @todo Implement this
 */
void multmod(bigint_t a, bigint_t b, int d, bigint_t r, prec_t m)
{
    zero(r, m);
    mult_gmp(a, b, r, d);
}

/**
 * @brief Calculates B^h-v*w and returns the result in B
 *
 * @param v
 * @param w
 * @param h
 * @param l
 * @param B Returns value in this bigint_t
 * @param m
 */
void powdiff(bigint_t v, bigint_t w, int h, int l, bigint_t B, prec_t m)
{
    int L = prec(v, m) + prec(w, m) - l + 1;
    if (ez(v, m) || ez(w, m) || L >= h)
    {
        bigint_t Bh = init(m);
        shift(h, B, Bh, m);
        mult_gmp(v, w, B, m);
        sub_gmp(Bh, B, B, m);
        free(Bh);
    }
    else
    {
        bigint_t P = init(m);
        multmod(v, w, L, P, m);
        if (ez(P, m))
        {
            zero(B, m);
        }
        else if (P[L - 1] == 0)
        {
            zero(B, m);
            sub_gmp(B, P, B, m);
        }
        else
        {
            bigint_t Bl = bpow(L, m);
            sub_gmp(Bl, P, B, m);
            free(Bl);
        }
        free(P);
    }
}

/**
 * @brief Refines accuracy of shifted inverse
 * @note Naive implementation
 */
void refine1(bigint_t v, int h, int k, bigint_t w, int l, prec_t m)
{
    int g = 1;
    h = h + g;
    shift(k - k - l, w, w, m);
    while (h - k > l)
    {
        step(h, v, w, 0, l, 0, m);
        l = min(2 * l - 1, h - k);
    }
    shift(-g, w, w, m);
}

/**
 * @brief Refines accuracy of shifted inverse
 * @note Only accurate intermediate digits are computed
 */
void refine2(bigint_t v, int h, int k, bigint_t w, int l, prec_t m)
{
    int g = 2;
    shift(g, w, w, m);
    while (h - k > l)
    {
        int n = min(h - k + 1 - l, l);
        step(k + l + n + g, v, w, n, l, g, m);
        shift(-1, w, w, m);
        l = l + n - 1;
    }
    shift(-g, w, w, m);
}

/**
 * @brief Refines accuracy of shifted inverse
 * @note Optimized refined version. Use this.
 *
 * @param v Input divisor
 * @param h The precision of the input divisor
 * @param k
 * @param w Approximation of quotient
 * @param l Current number of digits in w
 * @param m The number of digits in v
 */
void refine3(bigint_t v, int h, int k, bigint_t w, int l, prec_t m)
{
    int s;
    int g = 2;
    bigint_t v0 = init(m);
    shift(g, w, w, m);
    while (h - k > l)
    {
        int n = min(h - k + 1 - l, l);
        s = max(0, k - 2 * l + 1 - g);
        shift(-s, v, v0, m);
        step(k + l + n - s + g, v0, w, n, l, g, m);
        shift(-1, w, w, m);
        l = l + n - 1;
    }
    shift(-g, w, w, m);
}

/**
 * @brief
 *
 * @param h precision - 1
 * @param v input divisor
 * @param w return bigint_t
 * @param n number of digits needed (renamed from m)
 * @param l number of correct leading digits in w
 * @param g
 * @param m total number of digits in v
 */
void step(int h, bigint_t v, bigint_t w, prec_t n, int l, int g, prec_t m)
{
    bigint_t tmp = init(m);

    powdiff(v, w, h - m, l - g, tmp, m);
    mult_gmp(w, tmp, tmp, m);
    shift(2 * m - h, tmp, tmp, m);
    shift(n, w, w, m);
    add_gmp(w, tmp, w, m);
    free(tmp);
}

/**
 * @brief Writes the shiftet inverse of a bigint_t to w
 *
 * @param v the input bigint_t
 * @param w the output shiftet inverse
 * @param h precision - 1
 * @param k
 * @param m the total number of digits in v
 */
void shinv(bigint_t v, int h, int k, bigint_t w, prec_t m) {

    bigint_t B = bpow(1, m);
    bigint_t Bh = bpow(h, m);
    bigint_t Bk = bpow(k, m);


    if (lt(v, B, m))  { }
    if (lt(v, Bh, m)) { }
    if (eq(v, Bk, m)) { }

    if (lt(v, Bh, m)) { }

    int l = min(k, 2);
    // TODO: Implement this line
    bigint_t V = init(m);
    bigint_t B2l = bpow(2*l, m);
    sub_gmp(B2l, V, w, m);
    quo(w, V, w, m);
    add_gmp(w, 1, w, m);

    if (h - k <= l) {
        shift(h-k-l, w, w, m);
        return;
    }
    refine(v, h, k, w, l, m);

    free(B);
    free(Bh);
    free(Bk);
    free(V);
    free(B2l);
}

/**
 * @brief Divides two bigint_ts and returns the quotient and remainder
 * 
 * @param n numerator
 * @param d denominator
 * @param q quotient is returned here
 * @param r remainder is returned here
 * @param m Total number of digits in n and d
 */
void div(bigint_t n, bigint_t d, bigint_t q, bigint_t r, prec_t m) {
    int h = prec(n, m);
    int k = prec(d, m) - 1;

    // TODO: perform k == 1 check

    // Calculate quotient
    shinv(d, q, h, k, m);
    mult_gmp(n, q, q, m);
    shift(-h, q, q, m);

    // Calculate remainder
    mult_gmp(d, q, r, m);
    sub_gmp(n, r, r, m);

    // TODO: adjust approximation with regards to lambda = {0, 1}
}



int main() {
    
}





