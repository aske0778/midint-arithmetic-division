CXX = cuda

default: bench-div

test:
	futhark test div.fut

bench-quo:
	futhark bench div.fut --backend=$(CXX) --entry=bench_quo

bench-quo-single:
	futhark bench div.fut --backend=$(CXX) --entry=bench_quo_single

bench-div:
	futhark bench div.fut --backend=$(CXX) --entry=bench_div

.PHONY: clean
clean:
	rm -rf *.o *.c big-add sqr-mul div *.0 *.actual *.expected
