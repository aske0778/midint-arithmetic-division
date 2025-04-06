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
    srand(time(NULL));

    prec_t m = atoi(argv[1]);
    uint32_t u_arr[32] = {1681692777, 1714636915, 1957747793, 424238335, 719885386, 1649760492, 596516649, 1189641421, 1025202362, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    uint32_t v_arr[32] = {1350490027, 783368690, 1102520059, 2044897763, 1967513926, 1365180540, 1540383426, 304089172, 1303455736, 35005211, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   
    for (int i = 0; i < 1; i++) {
        bigint_t u = init_arr(m, u_arr); //init(m); 
        bigint_t v = init_arr(m, v_arr); //init(m);
        bigint_t q_own = init(m);
        bigint_t r_own = init(m);
        bigint_t q_gmp = init(m);
        bigint_t r_gmp = init(m);

        // uint32_t uPrec = (rand() % m) + 1;
        // uint32_t vPrec = (rand() % uPrec) + 3;
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
                printf("[%d/%d] IS INVALID\n", i, m - 1);
                break;
            }
        }
      //  printf("IS VALID\n");
    
        free(u);
        free(v);
        free(q_own);
        free(r_own);
        free(q_gmp);
        free(r_gmp);
    }
    return 0;
}
