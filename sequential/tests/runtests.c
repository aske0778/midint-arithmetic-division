#include "../div.h"
#include "test-shift.h"
#include "test-quo.h"
#include "test-prec.h"
#include "test-multmod.h"
#include "test-multd.h"
#include "test-bpow.h"


int main(int argc, char* argv[]) {
    if (argc != 1) {
        printf("Usage-fixd: %s 0 <m> <space-seperated big-ints>\n", argv[0]);
        printf("Usage-rand: %s 1 <m>\n", argv[0]);
        exit(1);
    }

    // prec_t m = atoi(argv[1]);
    prec_t m = 10;

    printf("Running tests...\n");
    printf("Test shift: \t%s\n", testShift(m) ? "FAILED" : "PASSED");
    printf("Test quo: \t%s\n", testQuo(m) ? "FAILED" : "PASSED");
    printf("Test prec: \t%s\n", testPrec(m) ? "FAILED" : "PASSED");
    printf("Test multd: \t%s\n", testMultd(m) ? "FAILED" : "PASSED");
    printf("Test bpow: \t%s\n", testBpow(m) ? "FAILED" : "PASSED");
    printf("Test multmod: \t%s\n", testMultmod(m) ? "FAILED" : "PASSED");

}


