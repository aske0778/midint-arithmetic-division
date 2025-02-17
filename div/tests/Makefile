CXX = nvcc -O3 -arch=sm_80
#-arch=sm_70
#-arch=sm_60

SRC_CGBN = blockwise-shift-test.cu
EXE_CGBN = blockwise-shift.exe

default: run

#.cu.o: $(SRC) $(HELPERS)
#	$(CXX) -c $@ $<


run:
	$(CXX) -ICGBN -DNUM_BITS=32768 -DTHD_PER_INST=32 $(SRC_CGBN) -o $(EXE_CGBN) -lgmp
	./$(EXE_CGBN)


clean:
	rm -f $(EXE_CGBN) $(EXE_OUR)

