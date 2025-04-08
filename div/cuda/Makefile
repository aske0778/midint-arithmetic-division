CXX = nvcc -O3 -arch=sm_80

SRC_MAIN = main
SRC_TEST = test
SRC_BENCH = benchmarks
SRC_time = timing

BATCH_SIZE = 134217728

default: main

main:
	$(CXX) $(SRC_MAIN).cu -lgmp -o $(SRC_MAIN).exe && ./$(SRC_MAIN).exe

test:
	$(CXX) $(SRC_TEST).cu -lgmp -o $(SRC_TEST).exe && ./$(SRC_TEST).exe

test-martin:
	$(CXX) test-martin.cu -Xptxas -v -lgmp -o test-martin.exe && ./test-martin.exe

main-martin:
	$(CXX) main-martin.cu -Xptxas -v -lgmp -o main-martin.exe && ./main-martin.exe

time:
	$(CXX) $(SRC_time).cu -lgmp -o $(SRC_time).exe && ./$(SRC_time).exe

bench:
	$(CXX) $(SRC_BENCH).cu -lgmp -o $(SRC_BENCH).exe && ./$(SRC_BENCH).exe $(BATCH_SIZE)

.PHONY: clean
clean:
	rm -rf *.o *.exe
