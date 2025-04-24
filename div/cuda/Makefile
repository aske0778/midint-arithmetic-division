CXX = nvcc -O3 -arch=sm_80 -Xptxas -v 
# -diag-suppress 550 -maxrregcount 70

SRC_MAIN = main
SRC_TEST = test
SRC_BENCH = bench
SRC_time = timing

#BATCH_SIZE = 33554432
#BATCH_SIZE = 134217728
BATCH_SIZE = 4194304

default: main

main: $(SRC_MAIN).cu ker-division.cu.h
	$(CXX) $(SRC_MAIN).cu -lgmp -o $(SRC_MAIN).exe && ./$(SRC_MAIN).exe $(BATCH_SIZE)

test:
	$(CXX) $(SRC_TEST).cu -lgmp -o $(SRC_TEST).exe && ./$(SRC_TEST).exe

bench:
	$(CXX) $(SRC_BENCH).cu -lgmp -o $(SRC_BENCH).exe && ./$(SRC_BENCH).exe

time:
	$(CXX) $(SRC_time).cu -lgmp -o $(SRC_time).exe && ./$(SRC_time).exe

.PHONY: clean
clean:
	rm -rf *.o *.exe
