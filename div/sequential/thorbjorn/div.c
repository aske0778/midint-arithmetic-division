#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>

/* This file assumes big integers as of type `uint32_t*`.

   It implements quotient (division) of big integers of exact precision.

   The implementation is based on the paper ``Efficient Generic Quotients
   Using Exact Arithmetic'' by Stephen M. Watt:
   https://arxiv.org/pdf/2304.01753.pdf

   The implementation does not aim to be highly efficient, but moreso as a
   baseline prototype to improve upon, e.g. by efficient parallel GPGPU
   implementations in CUDA and Futhark.

   Functions are parameterized over the exact precision `m`.
*/

/** TYPES **/

typedef uint32_t digit_t;
typedef uint64_t bigDigit_t;
// typedef uint16_t digit_t;
// typedef uint32_t bigDigit_t;
typedef digit_t *bigint_t;
const int32_t bits = 32;
//const int32_t  bits = 16;
typedef int prec_t;
typedef int bool;

/** HELPERS INT **/

int max(int a, int b)
{
    return (a > b) ? a : b;
}

int min(int a, int b)
{
    return (a < b) ? a : b;
}

/** HELPERS BIG INT **/

// allocate and initialize a big-int
bigint_t init(prec_t m)
{
    bigint_t retval = (bigint_t)malloc(m * sizeof(digit_t));
    for (int i = 0; i < m; i++)
    {
        retval[i] = 0;
    }
    return retval;
}

// checks whether big-int `u` is less than big-int `v`
bool lt(bigint_t u, bigint_t v, prec_t m)
{
    bool retval = 0;
    for (int i = 0; i < m; i++)
    {
        retval = (u[i] < v[i]) || ((u[i] == v[i]) && retval);
    }
    return retval;
}

// checks whether big-int `u` is equal to big-int `v`
bool eq(bigint_t u, bigint_t v, prec_t m)
{
    bool retval = 1;
    for (int i = 0; i < m; i++)
    {
        retval &= u[i] == v[i];
    }
    return retval;
}

// checks whether big-int `u` is equal to zero
bool ez(bigint_t u, prec_t m)
{
    bool retval = 1;
    for (int i = 0; i < m; i++)
    {
        retval &= u[i] == 0;
    }
    return retval;
}

// copy digits to big-int `u` from big-int `v`
void cpy(bigint_t u, bigint_t v, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        u[i] = v[i];
    }
}

// sets big-int `u` to a single digit `d` and zeroes out the rest
void set(bigint_t u, digit_t d, prec_t m)
{
    for (int i = 1; i < m; i++)
    {
        u[i] = 0;
    }
    u[0] = d;
}

// prints a string `s` followed big-int `u` to stdout
void prnt(char *s, bigint_t u, prec_t m)
{
    printf("%s: [", s);
    for (int i = 0; i < m; i++)
    {
        printf("%u,", u[i]);
    }
    printf("]\n");
}

/** PRIMARY **/

// computes how many digits are "used" (i.e. size without front zeroes)
prec_t findk(bigint_t u, prec_t m)
{
    prec_t k = 0;
    for (int i = 0; i < m; i++)
    {
        k = (u[i] != 0) ? i : k;
    }
    return k;
}

// computes how many digits are "used" + 1 (i.e. size without front zeroes)
prec_t prec(bigint_t u, prec_t m)
{
    return findk(u, m) + 1;
}

// shifts a big-int `u` by `n` to the right or left depending on the sign of `n`
// and writes result to big-int `v` (note that `u` and `v` can be the same)
void shift(int n, bigint_t u, bigint_t v, prec_t m)
{
    if (n >= 0)
    { // right shift
        for (int i = m - 1; i >= 0; i--)
        {
            int off = i - n;
            v[i] = (off >= 0) ? u[off] : 0;
        }
    }
    else
    { // left shift
        for (int i = 0; i < m; i++)
        {
            int off = i - n;
            v[i] = (off < m) ? u[off] : 0;
        }
    }
}

// computes and returns the base (uint32_t) of big-ints to the power of `n`
// (syntactic sugar for an initialization to `1` followed by a `n` right-shift)
bigint_t bpow(int n, prec_t m)
{
    bigint_t b = init(m);
    b[0] = 1;
    shift(n, b, b, m);
    return b;
}

// adds the two big-ints `u` and `v`, and write the result to `w`
void add(bigint_t u, bigint_t v, bigint_t w, prec_t m)
{
    /* This is supposed to be badd, but GMP is used for simplicity
       since this C implementation is not the focus of this thesis.
    */
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

// subtract the big-int `v` from the big-int `u`, and write the result to `w`
void sub(bigint_t u, bigint_t v, bigint_t w, prec_t m)
{
    /* This is supposed to be badd, but GMP is used for simplicity
       since this C implementation is not the focus of this thesis.
    */
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

// divides a big-int `u` by one digit `d` and write the result to `w`
void quod(bigint_t u, digit_t d, bigint_t w, prec_t m)
{
    /* The Long Division algorithm is used here
       https://en.wikipedia.org/wiki/Division_algorithm#Long_division.
    */
    bigDigit_t d0 = (bigDigit_t)d;
    bigDigit_t r = 0;
    for (int i = m - 1; i >= 0; i--)
    {
        r = (r << bits) + (bigDigit_t)u[i];
        if (r >= d0)
        {
            bigDigit_t q = r / d0;
            r -= q * d0;
            w[i] = (digit_t)q;
        }
        else
        {
            w[i] = 0;
        }
    }
}

// multiplies the two big-ints `u` and `v`, and write the result to `w`
void mult(bigint_t u, bigint_t v, bigint_t w, prec_t m)
{
    /* This is supposed to be convmult or FTT, but GMP is used for simplicity
       since this C implementation is not the focus of this thesis.
    */
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

// multiplies a big-int `u` with one digit `d` and write the result to `w`
void multd(bigint_t u, digit_t d, bigint_t w, prec_t m)
{
    bigDigit_t buff[m];
    for (int i = 0; i < m; i++)
    {
        buff[i] = ((bigDigit_t)u[i]) * ((bigDigit_t)d);
    }
    for (int i = 0; i < m - 1; i++)
    {
        // this cannot overflow since `buff[i]` and `buff[i+1]` is at most
        // `((2^32)-1) * ((2^32)-1) = 18446744065119617025`,
        // so `buff[i] >> 32` becomes `(2^32)-2` and `buff[i+1]` becomes
        // `((2^32)-2) + 18446744065119617025 = 18446744069414584319`,
        // which is less than `((2^64)-1) = 18446744073709551616`
        buff[i + 1] += buff[i] >> bits;
    }
    for (int i = 0; i < m; i++)
    {
        w[i] = (digit_t)buff[i];
    }
}

// multmod of big-ints `u` and `v` with exponent `d` and write the result to `w`
void multmod(bigint_t u, bigint_t v, int d, bigint_t w, prec_t m)
{
    /* MultMod computes `(u * v) rem (B^d)`. Expanding the definitions we get;
       `(u * v) rem (B^d) = (u * v) - (floor((u * v) / (B^d)) * (B^d))`.

       Since `B^d` is a big-int with exactly one `1`-digit at index `d`, we know
       that for some big-int `x`, `x rem (B^d)` is the same as zeroing out the
       upper `m-d` digits of `x` (i.e. the first `d-1` digits of `x`).

       However, `u * v` is still a nasty computation, since it requires doubling
       `m`. E.g. if multiplication by convolution is used, which is complexity
       O(n^2), doubling `m` results in quadruple runtime.

       Instead, we can use the fact that the upper `m-d` digits does not
       contribute to the lower `d` digits after the multiplication, and we get:
       `(u * v) rem (B^d) = ((u rem (B^d)) * (v rem (B^d))) rem (B^d)`.

       This can be optimized further to simply shorten the size-parameter of the
       multiplication from `m` to `d` and storing the result directly in `w`.
    */
    set(w, 0, m);
    mult(u, v, w, d);
}

// powerdiff function as described in the paper, writes result to big-int `P`
// (note, instead of incorporating signs in the big-ints, we return a bool)
bool powerdiff(bigint_t v, bigint_t w, int h, int l, bigint_t P, prec_t m)
{
    int L = prec(v, m) + prec(w, m) - l + 1;
    bigint_t bh = bpow(h, m);
    bool retval = 0;          // +
    if (ez(v, m) || ez(w, m)) // multiplication by 0 so `P = B^h`
        cpy(P, bh, m);
    else if (L >= h)
    {
        mult(v, w, P, m);
        if (lt(P, bh, m))
            sub(bh, P, P, m);
        else
        {
            sub(P, bh, P, m);
            retval = 1; // -
        }
    }
    else
    {
        multmod(v, w, L, P, m);
        if (!ez(P, m))
        {
            if (P[L - 1] == 0)
                retval = 1; // -
            else
            {
                bigint_t bL = bpow(L, m);
                if (lt(P, bL, m))
                    sub(bL, P, P, m);
                else
                {
                    sub(P, bL, P, m);
                    retval = 1; // -
                }
                free(bL);
            }
        }
    }
    free(bh);
    return retval;
}

// step function as described in the paper s.t. it writes result to big-int `w`
void step(int h, bigint_t v, bigint_t w, int n, int l, int g, prec_t m)
{
    // make a shifted copy of w
    bigint_t P = init(m);
    bigint_t w0 = init(m);
    cpy(w0, w, m);
    shift(n, w0, w0, m);

    // multiply, powerdiff and shift
    bool sign = powerdiff(v, w, h - n, l - g, P, m);
    mult(w, P, w, m);
    shift(2 * n - h, w, w, m);

    // add/subtract the two terms
    if (sign)
        sub(w, w0, w, m);
    else
        add(w, w0, w, m);

    // cleanup
    free(w0);
    free(P);
}

// DOES NOT VALIDATE, USE REFINE2 OR REFINE3 INSTEAD
// refine1 function as described in the paper s.t. it writes result to `w`
void refine1(bigint_t v, int h, int k, bigint_t w, int l, prec_t m)
{
    int g = 1;
    h += g;
    shift(h - k - l, w, w, m); // scale initial value to full length
    while (h - k > l)
    {
        step(h, v, w, 0, l, 0, m);
        l = min(2 * l - 1, h - k); // number of correct digits
    }
    shift(-g, w, w, m);
}

// refine2 function as described in the paper s.t. it writes result to `w`
void refine2(bigint_t v, int h, int k, bigint_t w, int l, prec_t m)
{
    int g = m / 4; // `m` guard digits; NOTE `m/4` because we use 4times buffer
    shift(g, w, w, m);
    while (h - k > l)
    {
        int n = min(h - k + 1 - l, l); // how much to grow
        step(k + l + n + g, v, w, n, l, g, m);
        shift(-1, w, w, m);
        l += n - 1;
    }
    shift(-g, w, w, m);
}

// refine3 function as described in the paper s.t. it writes result to `w`
void refine3(bigint_t v, int h, int k, bigint_t w, int l, prec_t m)
{
    int g = m / 4; // `m` guard digits; NOTE `m/4` because we use 4times buffer
    shift(g, w, w, m);
    bigint_t v0 = init(m);

    while (h - k > l)
    {
        int n = min(h - k + 1 - l, l);
        int s = max(0, k - (2 * l) + 1 - g); // how to scale v
        shift(-s, v, v0, m);
        step(k + l + n - s + g, v0, w, n, l, g, m);
        shift(-1, w, w, m);
        l += n - 1;
    }

    shift(-g, w, w, m);
    free(v0);
}

// whole-shifted inverse of big-int `v` by `h`; result is written to big-int `w`
void shinv(bigint_t v, int h, int k, bigint_t w, prec_t m)
{
    /* In the paper, we group digits if the base is too small. However,
       that is not an issue here since we use a sufficiently large fixed base.
    */

    // 1. handle special cases to guarantee `B < v <= B^h / 2`
    {
        // return predicate
        bool rp = 0;

        // compute base-powers
        bigint_t b = bpow(1, m);
        bigint_t bh = bpow(h, m);
        bigint_t bk = bpow(k, m);
        bigint_t v2 = init(m);
        multd(v, 2, v2, m);

        if (lt(v, b, m))
        {
            quod(bh, v[0], w, m);
            rp = 1;
        }
        else if (lt(bh, v, m))
        {
            set(w, 0, m);
            rp = 1;
        }
        else if (lt(bh, v2, m))
        {
            set(w, 1, m);
            rp = 1;
        }
        else if (eq(v, bk, m))
        {
            set(w, 0, m);
            w[0] = 1;
            shift(h - k, w, w, m);
            rp = 1;
        }

        // cleanup and exit if a special case is met
        free(b);
        free(bh);
        free(bk);
        free(v2);
        if (rp)
        {
            return;
        }
    }

    // 2. form initial approximation
    /* We handle the case where `k == 1` outside of this function (by shifting
       both `u` and `v` by one, and we handle case where `k == 0` in step 1..
    */
    int l = 2;
    {
        /* The method for finding the initial approximation described in the
           paper focuses on a generalized base. However, we can specialize it
           to make it significantly computational efficient:

           Since `k = 0` is handled as a special case above, we have either
           `l = 1` or `l = 2`. Thus, `V` is either two or three digits and
           `b^(2*l)` is either three or five digits (i.e. `[0,0,1]` or
           `[0,0,0,0,1]`). Hence, subtracting `V` from `b^2(2*l)` results in a
           term of either 2 or 4 digits.

           In turn, we can represent everything as unsigned 128-bit integers,
           which C supports, meaning we can efficiently divide.
        */
       // prnt("v",v,m);
        __uint128_t V = 0;
        V += ((__uint128_t)v[k - 2]);
        V += ((__uint128_t)v[k - 1]) << bits;
        V += ((__uint128_t)v[k]) << bits * 2;
        // printf("V:%u \n", v[k - 2]);
        // printf("V:%u \n", v[k-1]);
        // printf("V:%u \n", v[k]);

        // uint64_t V_low  = (uint64_t)(((__uint128_t)v[k - 2]));
        // uint64_t V_high = (uint64_t)(((__uint128_t)v[k - 2]) >> 64);

        // printf("Full V (uint128): high = %llu, low = %llu\n", (unsigned long long)V_high, (unsigned long long)V_low);

        // V_low  = (uint64_t)(((__uint128_t)v[k - 1]) << bits);
        // V_high = (uint64_t)((((__uint128_t)v[k - 1]) << bits) >> 64);

        // printf("Full V (uint128): high = %llu, low = %llu\n", (unsigned long long)V_high, (unsigned long long)V_low);

        // V_low  = (uint64_t)(((__uint128_t)v[k]) << bits * 2);
        // V_high = (uint64_t)((((__uint128_t)v[k]) << bits * 2) >> 64);

        // printf("Full V (uint128): high = %llu, low = %llu\n", (unsigned long long)V_high, (unsigned long long)V_low);

        // compute `(B^4 - V) / (V+1)`
        __uint128_t r = (((__uint128_t)0) - V) / (V) + 1;

        w[0] = (digit_t)(r);
        w[1] = (digit_t)(r >> bits);
        w[2] = (digit_t)(r >> bits * 2);
        w[3] = (digit_t)(r >> bits * 3);
    }
   // prnt("w",w,m);

    // 3. either return (if sufficient) or refine initial approximation
    if (h - k <= l)
    {
      //  printf("HERE");
        shift(h - k - l, w, w, m);
       // prnt("w",w,m);
    }
    else
    {
        refine3(v, h, k, w, l, m);
    }
}

// divides big integer `u` by `v` using Theorem 1 (see paper), and writes the
// quotient to big integer `q` and remainder to `r`
void div_shinv(bigint_t u, bigint_t v, bigint_t q, bigint_t r, prec_t m)
{
    // 1. compute assumptions
    int h = findk(u, m) + 1;
    int k = findk(v, m);

    // 2. requires padding of at least one since if `h=m` then `B^h > (B^m)-1`,
    //    but we use `3*m` because of the multiplications ('3m' is excesive)
    prec_t p = m * 4;

    // 3. allocate and initialize some big integers
    bigint_t a = init(p);
    cpy(a, u, m); // `a = u`
    bigint_t b = init(p);
    cpy(b, v, m);         // `b = v`
    bigint_t c = init(p); // `c = 0`

    // 4. if `k == 1`, we shift `u` and `v` by one to avoid the infinite loop
    //    for `l = 1` inside the `shinv` algorithm
   // prnt("v",v,m);
    if (k == 1)
    {
        h++;                   // `h = h + 1`
        k++;                   // `k = k + 1`
        shift(1, a, a, m + 1); // `a = a << 1`
        shift(1, b, b, m + 1); // `b = b << 1`
    }

    // 5. compute the quotient
    shinv(b, h, k, c, p); // `c = shinv_h b`
   // prnt("c",c,p);

    mult(a, c, b, p);     // `b = a * c`
    shift(-h, b, b, p);   // `b = shift_(-h) b`
    cpy(q, b, m);         // `q = b`

    // 6. compute the remainder and check whether 𝛿 = {0,1}
    mult(v, q, a, m); // `a = v * q`
    sub(u, a, r, m);  // `r = u - a`
    if (!lt(r, v, m))
    {                    // `if r >= v`
        set(a, 1, m);    // `a = 1`
        add(q, a, q, m); // `w = w + a`
        sub(r, v, r, m); // `r = r - v`
    }

    // 7. cleanup
    free(a);
    free(b);
    free(c);
}

// divides `u` by `v` using GMP and write quotient to `q` and remainder to `r`
void div_gmp(bigint_t u, bigint_t v, bigint_t q, bigint_t r, prec_t m)
{
    set(q, 0, m);
    set(r, 0, m);

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

    mpz_div(c, a, b); // c = a / b
    mpz_mul(d, b, c); // d = b * c
    mpz_sub(b, a, d); // b = a - d

    mpz_export(q, NULL, -1, sizeof(digit_t), 0, 0, c);
    mpz_export(r, NULL, -1, sizeof(digit_t), 0, 0, b);
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
    mpz_clear(d);
}

void randBigInt(bigint_t u, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        u[i] = (digit_t)rand();
    }
}

bigint_t init_arr(prec_t m, digit_t values[])
{
    bigint_t retval = (bigint_t)malloc(m * sizeof(digit_t));
    for (int i = 0; i < m; i++)
    {
        retval[i] = values[i];
    }
    return retval;
}
