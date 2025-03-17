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

    uint32_t u_arr[96] = {
    724803052, 756165936, 1836347364, 1132624577, 412559579, 362594798, 351040828, 0,
    1046741222, 337739299, 1896306640, 1343606042, 1111783898, 446340713, 1197352298,
    915256190, 1782280524, 846942590, 524688209, 700108581, 1566288819, 1371499336,
    2114937732, 726371155, 1927495994, 292218004, 882160379, 11614769, 1682085273,
    1662981776, 630668850, 246247255, 1858721860, 1548348142, 105575579, 964445884,
    2118421993, 1520223205, 452867621, 1017679567, 1857962504, 201690613, 213801961,
    822262754, 648031326, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    uint32_t v_arr[96] = {
    1829237727, 310022529, 1091414751, 0, 0, 0, 0, 0,  
    1411154259, 1737518944, 282828202, 110613202, 114723506, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

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
