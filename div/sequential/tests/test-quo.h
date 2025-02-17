#include "../div.h"


int testQuo(prec_t m) {
    bigint_t n = init(m);
    set(n, 20, m);
    shift(4, n, n, m);
    digit_t d = 5;

    quo(n, d, n, m);

    uint32_t* correct = (uint32_t*)calloc(m, sizeof(uint32_t));
    correct[4] = 4;

    for (int i = 0; i < m; i++) {
        if (n[i] != correct[i]) {
            printf("Input:\n");
            prnt("  n", n, m);
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, n[i], correct[i]);

            free(n);
            free(correct);
            return 1;
        }
    }
    free(n);
    free(correct);
    return 0;
}







