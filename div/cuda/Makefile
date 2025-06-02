CXX = nvcc -O3 -arch=sm_80
# CXX = nvcc -O3 -arch=sm_80 -Xptxas -v 
# -diag-suppress 550 -maxrregcount 70

SRC_MAIN = main

FLAGS = -lgmp -diag-suppress 63

# BATCH_SIZE = 8388608 #GCD
# BATCH_SIZE = 33554432
BATCH_SIZE = 134217728

default: main

main: $(SRC_MAIN).cu ker-division.cu.h
	$(CXX) $(SRC_MAIN).cu $(FLAGS) -o $(SRC_MAIN).exe && ./$(SRC_MAIN).exe $(BATCH_SIZE)

.PHONY: clean
clean:
	rm -rf *.o *.exe
