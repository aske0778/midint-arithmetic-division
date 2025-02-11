UNAME_S := $(shell uname -s)

COMPILER = gcc

FLAGS = -lgmp
ifeq ($(UNAME_S), Darwin)
	FLAGS += -L/opt/homebrew/lib -I/opt/homebrew/include
endif

default:
	$(COMPILER) test.c $(FLAGS) -o test.exe && ./test.exe

.PHONY: clean
clean:
	rm -rf *.exe


