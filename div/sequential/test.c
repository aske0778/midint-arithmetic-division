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

    uint32_t u[256] = {
        1681692777, 1714636915, 1957747793, 424238335, 719885386, 1649760492, 596516649,
        1189641421, 1025202362, 1350490027, 783368690, 1102520059, 2044897763, 1967513926,
        1365180540, 1540383426, 304089172, 1303455736, 35005211, 521595368, 294702567,
        1726956429, 336465782, 861021530, 278722862, 233665123, 2145174067, 468703135,
        1101513929, 1801979802, 1315634022, 635723058, 1369133069, 1125898167, 1059961393,
        2089018456, 628175011, 1656478042, 1131176229, 1653377373, 859484421, 1914544919,
        608413784, 756898537, 1734575198, 1973594324, 149798315, 2038664370, 1129566413,
        184803526, 412776091, 1424268980, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };


    uint32_t v[256] = {
        1911759956, 749241873, 137806862, 42999170, 982906996, 135497281, 511702305, 
        2084420925, 1937477084, 1827336327, 572660336, 1159126505, 805750846, 1632621729, 
        1100661313, 1433925857, 1141616124, 84353895, 939819582, 2001100545, 1998898814, 
        1548233367, 610515434, 1585990364, 1374344043, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
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
