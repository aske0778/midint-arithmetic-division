#include "../div.h"





int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }

    prec_t m = atoi(argv[0]);

    bigint_t u = init(m);
    bigint_t v = init(m);

    set(u, 5, m);
    set(v, 8, m);

    bigint_t q_own = init(m);
    bigint_t r_own = init(m);
    bigint_t q_gmp = init(m);
    bigint_t r_gmp = init(m); 

    div_shinv(u, v, q_own, r_own, m);
    div_gmp(u, v, q_gmp, r_gmp, m);

    for (int i = 0; i < m; i++) {
        if (q_own[i] != q_gmp[i] || r_own[i] != r_gmp[i]) {
            printf("---------------------------------------------------\n");
            printf("Inputs:\n");
            printf("  u", u, m);
            printf("  v", v, m);
            printf("Output:\n");
            printf("  q", q_own, m);
            printf("  r", r_own, m);
            printf("GMP:\n");
            printf("  q", q_gmp, m);
            printf("  r", r_gmp, m);
            printf("---------------------------------------------------\n");
            printf("[%d/%d] IS VALID\n", i, m-1);
            return 1;
        }
    }
    printf("IS VALID\n");
    return 0;
}







