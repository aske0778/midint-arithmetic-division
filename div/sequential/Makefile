UNAME_S := $(shell uname -s)
COMPILER = gcc
FLAGS = -lgmp

M = 256

ifeq ($(UNAME_S), Darwin)
	FLAGS += -L/opt/homebrew/lib -I/opt/homebrew/include
endif


default:
	$(COMPILER) test.c $(FLAGS) -o test.exe && ./test.exe $(M)

test: 
	$(COMPILER) tests/runtests.c $(FLAGS) -o tests/runtests.exe && ./tests/runtests.exe





.PHONY: clean
clean:
	rm -rf *.exe tests/*.exe