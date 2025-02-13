#include "../div.h"


int testPrec(prec_t m) {
    bigint_t u = init(m);
    bigint_t v = init(m);

    set(u, 15, m);
    set(v, -1, m);
    shift(6, u, u, m);
    shift(4, v, v, m);

    prec_t p1 = prec(u, m);
    prec_t p2 = prec(v, m);

    prec_t u_correct = 7;
    prec_t v_correct = 5;

    if (p1 != u_correct) {
        prnt("  u", u, m);
        printf("INVALID: [%u/%u]\n", p1, u_correct);

        free(u);
        free(v);
        return 1;
    }
    if (p2 != v_correct) {
        prnt("  v", v, m);
        printf("INVALID: [%u/%u]\n", p2, v_correct);

        free(u);
        free(v);
        return 1;
    }
    free(u);
    free(v);
    return 0;
}







