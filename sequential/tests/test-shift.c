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

    bigint_t u = init(m);
    bigint_t v = init(m);

    set(u, 1, m);
    set(v, 1, m);

    shift(2, u, u, m);
    shift(4, u, u, m);
    shift(-3, u, u, m);
    shift(-1, v, v, m);

    uint32_t* correct = (uint32_t*)malloc(m * (sizeof(uint32_t)));
    correct[3] = 1;

    for (int i = 0; i < m; i++) {
        if (u[i] != correct[i]) {
            printf("Inputs:\n");
            prnt("  v", u, m);
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, u[i], correct[i]);
            return 1;
        }
        if (v[i] != 0) {
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, v[i], 0);
            return 1;
        }
    }
    printf("SHIFT IS VALID\n");
    return 0;
}







