#include "../div.h"


int testPowdiff(prec_t m) {
    bigint_t u = init(m);
    bigint_t v = init(m);
    bigint_t B = bpow(4, m);

    int h = 2;
    int l = 5;

    set(u, 7, m);
    set(v, 3, m);
    shift(6, u, u, m);
    shift(4, v, v, m);

    powdiff(u, v, h, l, B, m);

    uint32_t* B_correct = (uint32_t*)calloc(m, sizeof(uint32_t));
    B_correct[6] = 1;

    for (int i = 0; i < m; i++) {
        if (B[i] != B_correct[i]) {
            printf("Input:\n");
            prnt("  u", B, m);
            printf("INVALID AT INDEX %u: [%u/%u]\n", i, B[i], B_correct[i]);

            free(u);
            free(v);
            free(B);
            free(B_correct);
            return 1;
        }
    }
    free(u);
    free(v);
    free(B);
    free(B_correct);
    return 0;
}







