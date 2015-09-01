NVCC=nvcc
NVCCFLAGS= -c

all: prog

prog: engine.o image.o ppm.o
	$(NVCC) ppm.o image.o engine.o -o fractal-gen

engine.o: engine.cu
	$(NVCC) $(NVCCFLAGS) engine.cu


ppm.o: ppm.cpp
	$(NVCC) $(NVCCFLAGS) ppm.cpp

image.o: image.cpp
	$(NVCC) $(NVCCFLAGS) image.cpp

clean:
	rm -rf *.o
