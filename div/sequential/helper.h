#ifndef HELPER
#define HELPER

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <gmp.h>
#include <string.h>

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

prec_t findk(bigint_t u, prec_t m)
{
    prec_t k = 0;
    for (int i = 0; i < m; i++)
    {
        k = (u[i] != 0) ? i : k;
    }
    return k;
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
            acc = i;
        }
    }
    return acc + 1;
}

/**
 * @brief Prints a string followed by the bigint_t
 */
void prnt(char *str, bigint_t u, prec_t m)
{
    printf("%s: [", str);
    for (int i = 0; i < m; i++)
    {
        printf("%u, ", u[i]);
    }
    printf("]\n");
}

void randBigInt(bigint_t u, prec_t m)
{
    for (int i = 0; i < m; i++)
    {
        u[i] = (uint32_t)rand();
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

#endif // HELPER