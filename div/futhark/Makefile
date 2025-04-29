CXX = cuda

default: bench-div

test-quo:
	futhark bench div.fut --entry test_quo

test-div:
	futhark bench div.fut --entry test_div

bench-quo:
	futhark bench --backend $(CXX) div.fut --entry bench_quo

bench-div:
	futhark bench --backend $(CXX) div.fut --entry bench_div

.PHONY: clean
clean:
	rm -rf *.o *.c big-add sqr-mul div *.0 *.actual *.expected
