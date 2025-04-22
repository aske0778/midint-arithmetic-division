CXX = cuda

default: div

div:
	futhark bench --backend $(CXX) div

sqr-mul:
	futhark bench --backend $(CXX) sqr-mul

big-add:
	futhark bench --backend $(CXX) big-add

.PHONY: clean
clean:
	rm -rf *.o *.c big-add sqr-mul div
