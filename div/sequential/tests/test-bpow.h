#include "../div.h"


int testBpow(prec_t m) {
    bigint_t u = bpow(5, m);
    bigint_t v = bpow(3, m);

    uint32_t* u_correct = (uint32_t*)calloc(m, sizeof(uint32_t));
    u_correct[5] = 1;

    uint32_t* v_correct = (uint32_t*)calloc(m, sizeof(uint32_t));
    v_correct[3] = 1;

    for (int i = 0; i < m; i++) {
        if (u[i] != u_correct[i]) {
            printf("Input:\n");
            prnt("  u", u, m);
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, u[i], u_correct[i]);

            free(u);
            free(v);
            free(u_correct);
            free(v_correct);
            return 1;
        }
        if (v[i] != v_correct[i]) {
            printf("Input:\n");
            prnt("  v", v, m);
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, v[i], v_correct[i]);

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







