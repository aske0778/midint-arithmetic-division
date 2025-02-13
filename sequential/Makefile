UNAME_S := $(shell uname -s)
COMPILER = gcc
FLAGS = -lgmp

M = 10

ifeq ($(UNAME_S), Darwin)
	FLAGS += -L/opt/homebrew/lib -I/opt/homebrew/include
endif


default:
	$(COMPILER) test.c $(FLAGS) -o test.exe && ./test.exe $(M)


.PHONY: test
test: test-shift test-bpow test-quo test-multd

test-shift:
	$(COMPILER) tests/test-shift.c $(FLAGS) -o tests/test-shift.exe && ./tests/test-shift.exe $(M)

test-bpow:
	$(COMPILER) tests/test-bpow.c $(FLAGS) -o tests/test-bpow.exe && ./tests/test-bpow.exe $(M)

test-quo:
	$(COMPILER) tests/test-quo.c $(FLAGS) -o tests/test-quo.exe && ./tests/test-quo.exe $(M)

test-multd:
	$(COMPILER) tests/test-multd.c $(FLAGS) -o tests/test-multd.exe && ./tests/test-multd.exe $(M)





.PHONY: clean
clean:
	rm -rf *.exe tests/*.exe