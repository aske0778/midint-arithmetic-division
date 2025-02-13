UNAME_S := $(shell uname -s)
COMPILER = gcc
FLAGS = -lgmp

M = 10


ifeq ($(UNAME_S), Darwin)
	FLAGS += -L/opt/homebrew/lib -I/opt/homebrew/include
endif


default:
	$(COMPILER) test.c $(FLAGS) -o test.exe && ./test.exe $(M)

test-shift:
	$(COMPILER) test-shift.c $(FLAGS) -o test-shift.exe && ./test-shift.exe $(M)




.PHONY: clean
clean:
	rm -rf *.exe