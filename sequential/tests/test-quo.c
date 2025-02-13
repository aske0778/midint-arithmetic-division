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

    bigint_t n = init(m);
    set(n, 20, m);
    shift(4, n, n, m);
    digit_t d = 5;

    quo(n, d, n, m);

    uint32_t* correct = (uint32_t*)malloc(m * (sizeof(uint32_t)));
    correct[4] = 4;

    for (int i = 0; i < m; i++) {
        if (n[i] != correct[i]) {
            printf("Input:\n");
            prnt("  n", n, m);
            printf("INVALID AT INDEX %d: [%d/%d]\n", i, n[i], correct[i]);
            return 1;
        }
    }
    printf("QUO IS VALID\n");
    return 0;
}







