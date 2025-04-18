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
    digit_t u_arr[32] = {0, 805251743, 1198720172, 1805613091, 144874089, 1510906527, 473903566, 13798878, 94255812, 1564003050, 99885196, 2081362124, 636453333, 363304213, 79065186, 1360478499, 604263370, 775056794, 1588695568, 1155465115, 535286141, 1389079342, 442982639, 1582482437, 4744263, 1642663198, 1153263590, 844169939, 1033206202, 0, 0, 0};
    digit_t v_arr[32] = {0, 181226513, 286791631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    for (int i = 0; i < 1; i++) {
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
