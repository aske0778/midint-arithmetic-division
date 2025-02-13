#include "../div.h"


void prnt(char* str, bigint_t u, prec_t m) {
    printf("%s: [", str);
    for (int i = 0; i < m; i++) {
        printf("%u, ", u[i]);
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

    set(u, 15, m);
    set(v, -1, m);
    shift(6, u, u, m);
    shift(4, v, v, m);

    digit_t d1 = 20;
    digit_t d2 = 4;

    multd(u, d1, u, m);
    multd(v, d2, v, m);

    uint32_t* u_correct = (uint32_t*)malloc(m * (sizeof(uint32_t)));
    u_correct[6] = 300;

    uint32_t* v_correct = (uint32_t*)malloc(m * (sizeof(uint32_t)));
    v_correct[8] = 800;

    for (int i = 0; i < m; i++) {
        if (u[i] != u_correct[i]) {
            printf("Input:\n");
            prnt("  u", u, m);
            printf("INVALID AT INDEX %u: [%u/%u]\n", i, u[i], u_correct[i]);
            return 1;
        }
        if (v[i] != v_correct[i]) {
            printf("Input:\n");
            prnt("  v", v, m);
            printf("INVALID AT INDEX %u: [%u/%u]\n", i, v[i], v_correct[i]);
            return 1;
        }
    }
    printf("MULTD IS VALID\n");
    return 0;
}







