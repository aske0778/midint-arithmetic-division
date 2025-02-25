CXX = nvcc -O3 -arch=sm_80


default: 


run:
	$(CXX) -ICGBN -DNUM_BITS=32768 -DTHD_PER_INST=32 main.cu -o main.exe -lgmp
	./main.exe





clean:
	rm -rf *.o *.exe
