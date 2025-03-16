#include <time.h>
#include "div.h"
// #include "thorbjorn/div.c"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }

    prec_t m = atoi(argv[1]);

    uint32_t u_arr[] = {724803052, 756165936, 1836347364, 1132624577, 412559579, 362594798, 351040828, 0};
    uint32_t v_arr[] = {1829237727, 310022529, 1091414751, 0, 0, 0, 0, 0};

    bigint_t u = init_arr(m, u_arr);
    bigint_t v = init_arr(m, v_arr);
    bigint_t q_own = init(m);
    bigint_t r_own = init(m);
    bigint_t q_gmp = init(m);
    bigint_t r_gmp = init(m);

    // srand(time(NULL));
    // randBigInt(u, m);
    // randBigInt(v, m - 50);

    
    div_shinv(u, v, q_own, r_own, m);
    div_gmp(u, v, q_gmp, r_gmp, m);

    // printf("Inputs:\n");
    // prnt("  u", u, m);
    // prnt("  v", v, m);
    // printf("Output:\n");
    // prnt("  q", q_own, m);
    // prnt("  r", r_own, m);
    // printf("GMP:\n");
    // prnt("  q", q_gmp, m);
    // prnt("  r", r_gmp, m);

    for (int i = 0; i < m; i++)
    {
        if (q_own[i] != q_gmp[i] || r_own[i] != r_gmp[i])
        {
            printf("[%d/%d] IS INVALID\n", i, m - 1);
            break;
        }
    }
    printf("IS VALID\n");

    free(u);
    free(v);
    free(q_own);
    free(r_own);
    free(q_gmp);
    free(r_gmp);
    return 0;
}
