#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>




typedef uint32_t digit_t;
typedef digit_t *bigint_t;
typedef uint32_t prec_t;



/**
 * @brief Checks if a bigint_t is equal to zero
 */
bool ez(bigint u, prec_t m) {
    for (int i = 0; i < m; i++) {
        if (u[i] != 0) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Checks if two bigint_ts are equal
 */
bool eq(bigint_t u, bigint_t v, prec_t m) {
    for (int i = 0; i < m; i++) {
        if (u[i] != v[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Initializes a bigint_t with m digits
 * 
 * @param m The number of digits
 * @return bigint_t The initialized bigint_t
 */
bigint_t init(prec_t m) {
    bigint_t retval = (bigint_t) malloc(m * sizeof(digit_t));
    for (int i=0; i < m; i++) {
        retval[i] = 0;
    }
    return retval;
}

/**
 * @brief The precision of the bigint_t
 */
prec_t prec(bigint_t u, prec_t m) {
    prec_t acc = 0;
    for (int i = 0; i < m; i++) {
        if (u[i] != 0) {
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
 * @param v bigint_t where result is stored
 * @param m The number of digits in u
 */
void shift(int n, bigint_t u, bigint_t v, prec_t m) {
    if (n >= 0) {   // Right shift
        for (int i = m - 1; i >= 0; i--) {
            int offset = i - n;
            v[i] = (offset >= 0) ? u[offset] : 0;
        }
    } else {        // Left shift
        for (int i = 0; i < m; i++) {
            int offset = i - n;
            v[i] = (offset < m) ? u[offset] : 0;
        }
    }
}


/**
 * @brief Uses gmp to multiply two bigint_ts
 * 
 * @param u First bigint_t
 * @param v Second bigint_t
 * @param w bigint_t where result is stored
 * @param m The number of digits in u and v
 */
void mult(bigint_t u, bigint_t v, bigint_t w, prec_t m) {
    mpz_t a; mpz_t b; mpz_t r;
    mpz_init(a); mpz_init(b); mpz_init(r);

    mpz_import(a, m, -1, sizeof(digit_t), 0, 0, u);
    mpz_import(b, m, -1, sizeof(digit_t), 0, 0, v);
    mpz_mul(r, a, b);

    set(w, 0, m);
    mpz_export(w, NULL, -1, sizeof(digit_t), 0, 0, r);

    mpz_clear(a); mpz_clear(b); mpz_clear(r);
}


// void powdiff(bigint_t v, bigint_t w, int h, int l, bigint_t B, prec_t m) {
//     int L = prec(v, m) + prec(w, m) - l + 1;
//     if (v == 0 || w == 0 || L >= h) {
//         return 
//     }

// }


