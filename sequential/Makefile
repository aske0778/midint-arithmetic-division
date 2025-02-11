
COMPILER = gcc

FLAGS = -lgmp

#For Mac
#$(COMPILER) test.c -o test.exe -L/opt/homebrew/lib -I/opt/homebrew/include -lgmp && ./test.exe

default:
	$(COMPILER) test.c $(FLAGS) -o test.exe && ./test.exe

.PHONY: clean
clean:
	rm -rf *.exe


