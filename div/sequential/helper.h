#ifndef HELPER
#define HELPER

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>
#include <string.h>

typedef unsigned __int128 uint128_t;

//8BIT
// typedef uint8_t digit_t;
// typedef uint16_t bigDigit_t;
// typedef uint32_t uquad_t;
// typedef digit_t *bigint_t;
// typedef uint32_t prec_t;
// const int32_t  bits = 8;
// typedef int bool;


//16BIT
typedef uint16_t digit_t;
typedef uint32_t bigDigit_t;
typedef uint64_t uquad_t;
typedef digit_t *bigint_t;
typedef uint32_t prec_t;
const int32_t  bits = 16;
typedef int bool;


//32BIT
// typedef uint32_t digit_t;
// typedef uint64_t bigDigit_t;
// typedef __uint128_t uquad_t;
// typedef digit_t *bigint_t;
// typedef uint32_t prec_t;
// const int32_t bits = 32;
// typedef int bool;

//64BIT
// typedef uint64_t digit_t;
// typedef __uint128_t bigDigit_t;
// typedef __uint128_t uquad_t;
// typedef digit_t *bigint_t;
// typedef uint32_t prec_t;
// const int32_t bits = 64;
// typedef int bool;

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
bool lt(bigint_t a, bigint_t b, prec_t m)
{
    for (int i = m - 1; i >= 0; i--)
    {
        if (a[i] < b[i])
        {
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

bigint_t init_arr(prec_t m, digit_t values[])
{
    bigint_t retval = (bigint_t)malloc(m * sizeof(digit_t));
    for (int i = 0; i < m; i++)
    {
        retval[i] = values[i];
    }
    return retval;
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
    prec_t acc = 0; // should be -1?
    for (int i = 0; i < m; i++)
    {
        if (u[i] != 0)
        {
            acc = i;
        }
    }
    return acc + 1;
}

/**
 * @brief Prints a string followed by the bigint_t
 */
// void prnt(char *str, bigint_t u, prec_t m)
// {
//     printf("%s: [", str);
//     for (int i = 0; i < m; i++)
//     {
//         printf("%u, ", u[i]);
//     }
//     printf("]\n");
// }
#include <inttypes.h>
void prnt(char *str, bigint_t u, prec_t m)
{
    printf("%s: [", str);
    for (int i = 0; i < m; i++)
    {
      //  printf("%u, ", u[i]);
        printf("%" PRIu64 ", ", u[i]);
    }
    printf("]\n");
}

void randBigInt(bigint_t u, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        u[i] = (digit_t)rand();
    }
}

// copy digits to big-int `u` from big-int `v`
void cpy(bigint_t u, bigint_t v, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        u[i] = v[i];
    }
}

uint128_t divide_u256_by_u128(uint128_t high, uint128_t low, uint128_t divisor) {
    uint128_t quotient = 0;
    uint128_t rem = 0;
    
    bool overflow = 0;
    for (int i = 192; i >= 0; i--) {
        if (rem & (__uint128_t)1 << 127) {     //(__uint128_t)1 << 127  
            overflow = 1;
        }
        rem <<= 1;

        if (i == 192) {
            rem |= 1;
        } 

        quotient <<= 1;

        if (rem >= divisor || overflow) {
            rem -= divisor;
            quotient |= 1;
            overflow = 0;
        }
    }

    return quotient;
}



#endif // HELPER