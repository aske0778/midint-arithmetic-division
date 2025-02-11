#ifndef HELPER
#define HELPER

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



#endif // HELPER