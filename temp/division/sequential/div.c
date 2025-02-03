#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>




typedef uint32_t digit_t;
typedef digit_t *bigint_t;
typedef uint32_t prec_t;


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
 * @brief Shifts a bigint_t to the left or right depending on sign of n
 * 
 * @param n The sign and number of shifts
 * @param u The input bigint_t
 * @param m The number of digits in u
 * 
 * @return bigint_t The shifted bigint_t
 */
bigint_t shift(int n, bigint_t u, prec_t m) {
    bigint_t v = init(m);
    if (n >= 0) {
        for (int i = m - 1; i >= 0; i--) {
            int offset = i - n;
            v[i] = (offset >= 0) ? u[offset] : 0;
        }
    } else {
        for (int i = 0; i < m; i++) {
            int offset = i - n;
            v[i] = (offset < m) ? u[offset] : 0;
        }
    }
    return v;
}



