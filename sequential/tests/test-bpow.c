#include "../div.h"


void prnt(char* str, bigint_t u, prec_t m) {
    printf("%s: [", str);
    for (int i = 0; i < m; i++) {
        printf("%d, ", u[i]);
    }
    printf("]\n");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }

    prec_t m = atoi(argv[1]);

    bigint_t u = bpow(5, m);
    bigint_t v = bpow(3, m);

    uint32_t* u_correct = (uint32_t*)malloc(m * (sizeof(uint32_t)));
    u_correct[5] = 1;

    uint32_t* v_correct = (uint32_t*)malloc(m * (sizeof(uint32_t)));
    v_correct[3] = 1;

    for (int i = 0; i < m; i++) {
        if (u[i] != u_correct[i]) {
            printf("Input:\n");
            prnt("  u", u, m);
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, u[i], u_correct[i]);
            return 1;
        }
        if (v[i] != v_correct[i]) {
            printf("Input:\n");
            prnt("  v", v, m);
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, v[i], v_correct[i]);
            return 1;
        }
    }
    printf("BPOW IS VALID\n");
    return 0;
}







