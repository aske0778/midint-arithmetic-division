#include <time.h>
#include "div.h"
// #include "thorbjorn/div.c"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        printf("usage number of times to devide %s <m> \n", argv[0]);
        exit(1);
    }

    prec_t m = atoi(argv[1]);
    int j = (atoi(argv[2]));

    for (int k = 0; k < j; k++){
        bigint_t u = init(m);
        bigint_t v = init(m);
        bigint_t q_own = init(m);
        bigint_t r_own = init(m);
        bigint_t q_gmp = init(m);
        bigint_t r_gmp = init(m);

        srand(time(NULL));
        randBigInt(u, m);
        randBigInt(v, m - 2);

        //v[0] = (v[0] == 0) ? (uint32_t)1: v[0]; 
        if (v[0] == 0) {
            v[0] = (uint32_t)1;
            printf("tried to devide by zero");
        }

        div_shinv(u, v, q_own, r_own, m);
        div_gmp(u, v, q_gmp, r_gmp, m);

        for (int i = 0; i < m; i++)
        {
            if (q_own[i] != q_gmp[i] || r_own[i] != r_gmp[i])
            {
                printf("Inputs:\n");
                prnt("  u", u, m);
                prnt("  v", v, m);
                printf("Output:\n");
                prnt("  q", q_own, m);
                prnt("  r", r_own, m);
                printf("GMP:\n");
                prnt("  q", q_gmp, m);
                prnt("  r", r_gmp, m);
                printf("[%d/%d] IS INVALID\n", i, m - 1);
                break;
            }
        }

        free(u);
        free(v);
        free(q_own);
        free(r_own);
        free(q_gmp);
        free(r_gmp);
    }


    return 0;
}
