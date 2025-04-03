CXX = nvcc

SRC_MAIN = main
SRC_TEST = test

default: main

main:
	$(CXX) $(SRC_MAIN).cu -lgmp -o $(SRC_MAIN).exe && ./$(SRC_MAIN).exe

test:
	$(CXX) $(SRC_TEST).cu -Xptxas -v -lgmp -o $(SRC_TEST).exe && ./$(SRC_TEST).exe

.PHONY: clean
clean:
	rm -rf *.o *.exe
