CXX = nvcc -O3 -arch=sm_80

SRC_MAIN = main
SRC_MAIN_GCD = main_gcd
SRC_TEST = test
SRC_BENCH = bench

default: main

main:
	$(CXX) $(SRC_MAIN).cu -Xptxas -v -lgmp -o $(SRC_MAIN).exe && ./$(SRC_MAIN).exe

main_gcd:
	$(CXX) $(SRC_MAIN_GCD).cu -Xptxas -v -lgmp -o $(SRC_MAIN_GCD).exe && ./$(SRC_MAIN_GCD).exe

test:
	$(CXX) $(SRC_TEST).cu -Xptxas -v -lgmp -o $(SRC_TEST).exe && ./$(SRC_TEST).exe

bench:
	$(CXX) $(SRC_BENCH).cu -Xptxas -v -lgmp -o $(SRC_BENCH).exe && ./$(SRC_BENCH).exe

.PHONY: clean
clean:
	rm -rf *.o *.exe
