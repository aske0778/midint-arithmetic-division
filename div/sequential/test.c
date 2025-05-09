#include <time.h>
#include "div.h"
//#include "thorbjorn/div.c"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }
    //srand(time(NULL));

    prec_t m = atoi(argv[1]);
    // digit_t u_arr[16] = {1681692777, 1714636915, 1957747793, 424238335, 719885386, 1649760492, 596516649, 1189641421, 0, 0, 0, 0, 0, 0, 0, 0};
    // digit_t v_arr[16] = {1025202362, 1350490027, 783368690, 1102520059, 2044897763, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    // digit_t u_arr[16] = {8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0};
    // digit_t v_arr[16] = {4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};

    // digit_t u_arr[16] = {8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0};
    // digit_t v_arr[16] = {4, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

    // digit_t u_arr[64] = {39017, 18547, 56401, 23807, 37962, 22764, 7977, 31949, 22714, 55211, 16882, 7931, 43491, 57670, 124, 25282, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // digit_t v_arr[64] = {2132, 10232, 8987, 59880, 52711, 17293, 3958, 9562, 63790, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    digit_t u_arr[32] = {4, 3, 1, 4, 2, 2, 1, 3, 1, 1, 2, 3, 2, 3, 3, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    digit_t v_arr[32] = {3, 3, 4, 2, 4, 4, 3, 4, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


    for (int i = 0; i < 1; i++) {
        printf("\rIteration: %u", i);
        bigint_t u = init_arr(m, u_arr);
        bigint_t v = init_arr(m, v_arr);
        // bigint_t u = init(m); 
        // bigint_t v = init(m);
        bigint_t q_own = init(m);
        bigint_t r_own = init(m);
        bigint_t q_gmp = init(m);
        bigint_t r_gmp = init(m);

        // digit_t uPrec = (rand() % m) + 1;
        // digit_t vPrec = (rand() % uPrec) + 3;
        // randBigInt(u, uPrec);
        // randBigInt(v, vPrec);

        
        div_shinv(u, v, q_own, r_own, m);
        div_gmp(u, v, q_gmp, r_gmp, m);

        printf("Inputs:\n");
        prnt("  u", u, m);
        prnt("  v", v, m);
        printf("Output:\n");
        prnt("  q", q_own, m);
        prnt("  r", r_own, m);
        printf("GMP:\n");
        prnt("  q", q_gmp, m);
        prnt("  r", r_gmp, m);

        for (int i = 0; i < m; i++)
        {
            if (q_own[i] != q_gmp[i] || r_own[i] != r_gmp[i])
            {
                // printf("Inputs:\n");
                // prnt("  u", u, m);
                // prnt("  v", v, m);
                // printf("Output:\n");
                // prnt("  q", q_own, m);
                // prnt("  r", r_own, m);
                // printf("GMP:\n");
                // prnt("  q", q_gmp, m);
                // prnt("  r", r_gmp, m);

                printf("[%d/%d] IS INVALID\n", i, m - 1);
                break;
            }
        }
     //   printf("IS VALID\n");
    
        free(u);
        free(v);
        free(q_own);
        free(r_own);
        free(q_gmp);
        free(r_gmp);
    }
    return 0;
}
