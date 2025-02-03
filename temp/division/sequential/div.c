#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>




typedef uint32_t digit_t;
typedef digit_t *bigint_t;
typedef uint32_t prec_t;



bigint_t init(prec_t m) {
    bigint_t retval = (bigint_t) malloc(m * sizeof(digit_t));
    for (int i=0; i < m; i++) {
        retval[i] = 0;
    }
    return retval;
}





