CXX = cuda

default: bench-div

test:
	futhark test div.fut

bench-div:
	futhark bench div.fut --backend=$(CXX) --entry=bench_div

bench-div-single:
	futhark bench div.fut --backend=$(CXX) --entry=bench_div_single

bench-gcd:
	futhark bench div.fut --backend=$(CXX) --entry=bench_gcd

bench-gcd-single:
	futhark bench div.fut --backend=$(CXX) --entry=bench_gcd_single

.PHONY: clean
clean:
	rm -rf *.o *.c big-add sqr-mul div *.0 *.actual *.expected
