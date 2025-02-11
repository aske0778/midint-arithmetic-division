#include "../div.h"





int main() {
    prec_t m = 10;

    bigint_t u = init(m);
    bigint_t v = init(m);
    bigint_t q_own = init(m);
    bigint_t r_own = init(m);
    bigint_t q_gmp = init(m);
    bigint_t r_gmp = init(m); 

    div_shinv(u, v, q_own, r_own, m);
    div_gmp(u, v, q_gmp, r_gmp, m);

    bool p = 1;
    for(int i = 0; i < m; i++) {
        p = p && q_own[i] == q_gmp[i] && r_own[i] == r_gmp[i];
    }

}







