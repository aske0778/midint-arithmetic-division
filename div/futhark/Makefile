CXX = cuda
ALGO = div

default: bench-$(ALGO)

test-$(ALGO):
	futhark bench --backend $(CXX) div.fut --entry test_$(ALGO)

bench-div:
	futhark bench --backend $(CXX) div.fut --entry bench_$(ALGO)


.PHONY: clean
clean:
	rm -rf *.o *.c big-add sqr-mul div
