#ifndef SEQ_DIV
#define SEQ_DIV

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>
#include "helper.h"

/**
 * @brief Shifts a bigint_t to the left or right depending on sign of n
 *
 * @param n The sign and number of shifts
 * @param u The input bigint_t
 * @param r bigint_t where result is stored
 * @param m The number of digits in u
 */
void shift_(int n, bigint_t u, bigint_t r, prec_t m)
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
 * @brief Computes the base og bigint_t to the power of n
 *
 * @param n the power to raise the base to
 * @param m the total number of digits in the bigint_t
 */
bigint_t bpow_(int n, prec_t m)
{
    bigint_t a = init_(m);
    a[0] = 1;
    shift_(n, a, a, m);
    return a;
}

/**
 * @brief Uses gmp to add two bigint_ts
 *
 * @param u First bigint_t
 * @param v Second bigint_t
 * @param w bigint_t where result is stored
 * @param m The number of digits in u and v
 */
void add_gmp_(bigint_t u, bigint_t v, bigint_t w, prec_t m)
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

    set_(w, 0, m);
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
void sub_gmp_(bigint_t u, bigint_t v, bigint_t w, prec_t m)
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

    set_(w, 0, m);
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
void mult_gmp_(bigint_t u, bigint_t v, bigint_t w, prec_t m)
{
    mpz_t a;
    mpz_t b;
    mpz_t r;
    mpz_init(a);
    mpz_init(b);
    mpz_init(r);

    mpz_import(a, m, -1, sizeof(digit_t), 0, 0, u);
    mpz_import(b, m, -1, sizeof(digit_t), 0, 0, v);
    mpz_mul(r, a, b);

    set_(w, 0, m);
    mpz_export(w, NULL, -1, sizeof(digit_t), 0, 0, r);

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(r);
}

void div_gmp_(bigint_t u, bigint_t v, bigint_t q, bigint_t r, prec_t m)
{
    set_(q, 0, m);
    set_(r, 0, m);

    mpz_t a;
    mpz_init(a);
    mpz_import(a, m, -1, sizeof(digit_t), 0, 0, u);
    mpz_t b;
    mpz_init(b);
    mpz_import(b, m, -1, sizeof(digit_t), 0, 0, v);
    mpz_t c;
    mpz_init(c);
    mpz_t d;
    mpz_init(d);

    mpz_div(c, a, b);
    mpz_mul(d, b, c);
    mpz_sub(b, a, d);

    mpz_export(q, NULL, -1, sizeof(digit_t), 0, 0, c);
    mpz_export(r, NULL, -1, sizeof(digit_t), 0, 0, b);
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
    mpz_clear(d);
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
void quo_(bigint_t n, digit_t d, bigint_t q, prec_t m)
{
    if (d == 0)
    {
        printf("Division by zero\n");
        return;
    }
    uint64_t r = 0;
    for (int i = m - 1; i >= 0; i--)
    {
        r = (r << 32) + n[i];
        if (r >= d)
        {
            q[i] = r / d;
            r = r % d;
        }
    }
}

/**
 * @brief Multiplication of bigint_t with digit_t
 *
 * @param a Input bigint_t
 * @param b Input digit_t
 * @param r Result bigint_t
 * @param m Size of bigint_t
 */
void multd_(bigint_t a, digit_t b, bigint_t r, prec_t m)
{
    uint64_t buf[m];

    for (int i = 0; i < m; i++)
    {
        buf[i] = ((uint64_t)a[i]) * (uint64_t)b;
    }

    for (int i = 1; i < m - 1; i++)
    {
        buf[i + 1] += buf[i] >> 32;
    }

    for (int i = 0; i < m; i++)
    {
        r[i] = (digit_t)buf[i];
    }
}

/**
 * @brief Calculates (a * b) rem B^d
 * @todo Implement this
 */
void multmod_(bigint_t a, bigint_t b, int d, bigint_t r, prec_t m)
{
    zero_(r, m);
    mult_gmp_(a, b, r, d);
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
bool powdiff_(bigint_t v, bigint_t w, int h, int l, bigint_t B, prec_t m)
{
    prec_t L = prec_(v, m) + prec_(w, m) - l + 1;
    prec_t sign = 1;
    if (ez_(v, m) || ez_(w, m) || L >= h)
    {
        bigint_t Bh = bpow_(h, m);
        mult_gmp_(v, w, B, m);

        if (lt_(B, Bh, m))
            sub_gmp_(Bh, B, B, m);
        else
        {
            sub_gmp_(B, Bh, B, m); // else case nogensinde aktuelt?
            sign = 0;
        }
        free(Bh);
    }
    else
    {
        multmod_(v, w, L, B, m);
        if (ez_(B, m))
        {
            zero_(B, m);
        }
        else if (B[L - 1] == 0)
        {
            sign = 0;
        }
        else
        {

            bigint_t Bl = bpow_(L, m);
            sub_gmp_(Bl, B, B, m);
            if (lt_(B, Bl, m))
                sub_gmp_(Bl, B, B, m);
            else
            {
                sub_gmp_(B, Bl, B, m);
                sign = 0;
            }
            free(Bl);
        }
    }
    return sign;
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
void step_(int h, bigint_t v, bigint_t w, prec_t n, int l, int g, prec_t m)
{
    bigint_t tmp = init_(m);

    prec_t sign = powdiff_(v, w, h - n, l - g, tmp, m);

    mult_gmp_(w, tmp, tmp, m);
    // prnt("w", w, m);
    shift_(2 * n - h, tmp, tmp, m);
    shift_(n, w, w, m);
    // prnt("w", w, m);
    if (sign)
        add_gmp_(w, tmp, w, m);
    else
        sub_gmp_(w, tmp, w, m);

    free(tmp);
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
void refine3_(bigint_t v, int h, int k, bigint_t w, int l, prec_t m)
{
    int s;
    int g = 2; // på grund af 4x padding
    bigint_t v0 = init_(m);
    shift_(g, w, w, m);
    while (h - k > l)
    {
        int n = min_(h - k + 1 - l, l);

        s = max_(0, k - 2 * l + 1 - g);

        shift_(-s, v, v0, m);

        step_(k + l + n - s + g, v0, w, n, l, g, m);
        shift_(-1, w, w, m);
        l = l + n - 1;
    }

    shift_(-g, w, w, m);
    // prnt("res1", w, m);
    free(v0);
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
void shinv_(bigint_t v, int h, bigint_t w, prec_t m)
{
    int k = prec_(v, m) - 1;
    bool rp = 0;

    bigint_t B = bpow_(1, m);
    bigint_t Bh = bpow_(h, m);
    bigint_t Bk = bpow_(k, m);
    bigint_t v2 = init_(m);
    // prnt("Bh:", Bh, m);
    multd_(v, 2, v2, m);

    if (lt_(v, B, m))
    {
        quo_(Bh, v[0], w, m);
        rp = 1;
    }
    if (lt_(Bh, v, m))
    {
        zero_(w, m);
        rp = 1;
    }
    if (lt_(Bh, v2, m))
    {
        zero_(w, m);
        rp = 1;
    }
    if (eq_(v, Bk, m))
    {
        zero_(w, m);
        w[0] = 1;
        shift_(h - k, w, w, m);
        rp = 1;
    }
    free(B);
    free(Bh);
    free(Bk);
    free(v2);

    if (rp)
    {
        return;
    }

    int l = min_(k, 2);
    __uint128_t V = 0;
    for (int i = 0; i <= l; i++)
    {
        V += ((__uint128_t)v[k - l + i]) << (32 * i);
    }

    __uint128_t b2l = (__uint128_t)1 << 32 * 2 * l;
    __uint128_t tmp = (b2l - V) / V + 1;

    w[0] = (digit_t)(tmp);
    w[1] = (digit_t)(tmp >> 32);
    w[2] = (digit_t)(tmp >> 64);
    w[3] = (digit_t)(tmp >> 96);

    if (h - k <= l)
    {
        shift_(h - k - l, w, w, m);
    }
    else
    {
        refine3_(v, h, k, w, l, m);
    }
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
void div_shinv_(bigint_t u, bigint_t v, bigint_t q, bigint_t r, prec_t m)
{
    int h = prec_(u, m);
    printf("%i", h);
    // Calculate quotient
    shinv_(v, h, q, m);
    mult_gmp_(u, q, q, m);
    prnt("q:", q, m);
    shift_(-h, q, q, m);

    // Calculate remainder
    mult_gmp_(v, q, r, m);
    sub_gmp_(u, r, r, m);

    if (!lt_(r, v, m))
    {
        bigint_t a = bpow_(0, m);
        add_gmp_(q, a, q, m);
        sub_gmp_(r, v, r, m);
    }
}

#endif // SEQ_DIV