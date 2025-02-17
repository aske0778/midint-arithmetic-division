#include "../div.h"


int testMultd(prec_t m) {
    bigint_t u = init(m);
    bigint_t v = init(m);

    set(u, 15, m);
    set(v, -1, m);
    shift(6, u, u, m);
    shift(4, v, v, m);

    digit_t d1 = 20;
    digit_t d2 = 4;

    multd(u, d1, u, m);
    multd(v, d2, v, m);

    uint32_t* u_correct = (uint32_t*)calloc(m, sizeof(uint32_t));
    u_correct[6] = 300;

    uint32_t* v_correct = (uint32_t*)calloc(m, sizeof(uint32_t));
    v_correct[4] = 4294967292;
    v_correct[5] = 3;

    for (int i = 0; i < m; i++) {
        if (u[i] != u_correct[i]) {
            printf("Input:\n");
            prnt("  u", u, m);
            printf("INVALID AT INDEX %u: [%u/%u]\n", i, u[i], u_correct[i]);

            free(u);
            free(v);
            free(u_correct);
            free(v_correct);
            return 1;
        }
        if (v[i] != v_correct[i]) {
            printf("Input:\n");
            prnt("  v", v, m);
            printf("INVALID AT INDEX %u: [%u/%u]\n", i, v[i], v_correct[i]);

            free(u);
            free(v);
            free(u_correct);
            free(v_correct);
            return 1;
        }
    }
    free(u);
    free(v);
    free(u_correct);
    free(v_correct);
    return 0;
}







