NVCC=nvcc
NVCCFLAGS= -c

all: prog

prog: engine.o image.o ppm.o
	$(NVCC) ppm.o image.o engine.o main.o -o fractal-gen

main.o: main.cu
	$(NVCC) $(NVCCFLAGS) main.cu

engine.o: engine.cu
	$(NVCC) $(NVCCFLAGS) engine.cu

ppm.o: ppm.cpp
	$(NVCC) $(NVCCFLAGS) ppm.cu

image.o: image.cpp
	$(NVCC) $(NVCCFLAGS) image.cu

clean:
	rm -rf *.o
