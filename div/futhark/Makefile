CXX = cuda

default: div

div:
	futhark bench --backend $(CXX) div.fut

sqr-mul:
	futhark bench --backend $(CXX) sqr-mul.fut

big-add:
	futhark bench --backend $(CXX) big-add.fut

.PHONY: clean
clean:
	rm -rf *.o *.c big-add sqr-mul div
