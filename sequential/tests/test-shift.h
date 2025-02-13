#include "../div.h"


int testShift(prec_t m) {
    bigint_t u = init(m);
    bigint_t v = init(m);

    set(u, 1, m);
    set(v, 1, m);

    shift(2, u, u, m);
    shift(4, u, u, m);
    shift(-3, u, u, m);
    shift(-1, v, v, m);

    uint32_t* correct = (uint32_t*)calloc(m, sizeof(uint32_t));
    correct[3] = 1;

    for (int i = 0; i < m; i++) {
        if (u[i] != correct[i]) {
            printf("Inputs:\n");
            prnt("  v", u, m);
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, u[i], correct[i]);

            free(u);
            free(v);
            free(correct);
            return 1;
        }
        if (v[i] != 0) {
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, v[i], 0);

            free(u);
            free(v);
            free(correct);
            return 1;
        }
    }
    free(u);
    free(v);
    free(correct);
    return 0;
}







