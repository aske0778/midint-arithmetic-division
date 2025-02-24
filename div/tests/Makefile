CXX = nvcc -O3 -arch=sm_80
#-arch=sm_70
#-arch=sm_60

# SRC_CGBN = blockwise-shift-test.cu
# EXE_CGBN = blockwise-shift.exe

default: test-shift test-multd test-set

#.cu.o: $(SRC) $(HELPERS)
#	$(CXX) -c $@ $<

test-set:
	$(CXX) -ICGBN -DNUM_BITS=32768 -DTHD_PER_INST=32 blockwise-set-test.cu -o blockwise-set-test.exe -lgmp
	./blockwise-set-test.exe

test-shift:
	$(CXX) -ICGBN -DNUM_BITS=32768 -DTHD_PER_INST=32 blockwise-shift-test.cu -o blockwise-shift-test.exe -lgmp
	./blockwise-shift-test.exe

test-multd:
	$(CXX) -ICGBN -DNUM_BITS=32768 -DTHD_PER_INST=32 blockwise-multd-test.cu -o blockwise-multd-test.exe -lgmp
	./blockwise-multd-test.exe

test-prec:
	$(CXX) -ICGBN -DNUM_BITS=32768 -DTHD_PER_INST=32 blockwise-prec-test.cu -o blockwise-prec-test.exe -lgmp
	./blockwise-prec-test.exe

clean:
	rm -rf *.o *.exe

