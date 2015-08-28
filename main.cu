#include <stido.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>



__global__ ifsKernel(float * pts, unsigned char * outBuf) {

	

	float outs[7];
	int idx = threadIdx.x + blockIdx.x*blockDim.x
	switch (blockIdx.x) {
		case (1): {
			for (int i = 0; i < 7; ++i)
				out[i] = sin(pts[idx+i]);
			break;
		}
		case (2): {
			outs[0] = pts[idx+1] * pts[idx+5]
					- pts[idx+2] * pts[idx+4];
			outs[1] = pts[idx+2] * pts[idx+3]
					- pts[idx] * pts[idx+5];
			outs[2] = pts[idx] * pts[idx+4]
					- pts[idx+1] * pts[idx+3];
			outs[3] = pts[idx+3];
			outs[4] = pts[idx+4];
			outs[5] = pts[idx+5];
			outs[6] = pts[idx+6];
			break;
		}
		case (3): {
			for (int i = 0; i < 7; ++i)
				out[i] = (pts[idx+i]+pts[(idx+i+1)%7])/3.0;
			break;
		}
		case (4): {
			for (int i = 0; i < 7; ++i)
				out[i] = atan(pts[idx+i]+pts[(idx+i*i)%7])/3.0;
			break;
		}
		//tranform
		//quantize
		//mix rgba
		//add to outBuf
	}
}

void * writerThread(void * arg) {
	t_data * input = (t_data *) arg;
	cudaMemCpy(input->charBuf, input->GPUcharBuf, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	for (int i = 0; i < input->size; ++i) {
		int x = input->charBuf[i*7+0];
		int y = input->charBuf[i*7+1];
		
		node_t * mynode = malloc(sizeof(node));
		node->r = input->charBuf[i*7+3];
		node->g = input->charBuf[i*7+4];
		node->b = input->charBuf[i*7+5];
		node->a = input->charBuf[i*7+6];
		node->z = input->charBuf[i*7+2];
	
		insertNode(&image, x, y, mynode);
	}
	sem_post(input->sem);

}

typedef t_data {
	image_t ;
	int size;
	sem_t sem;
	unsigned char * charBuf;
	unsigned char * GPUcharBuf;
} t_data;


void main(int argc, char ** argv) {
	
	int dimx = 500;
	int dimy = 500;
	
	const int dimSize = 7; //xyzrgba
	const int totalThreads = 1024;
	const int totalThreads = 4; 
	const int bufSize = totalThreads * dimSize;
	float buf[bufSize];
	unsigned char charBuf[bufSize];
	float * ins;
	unsigned char * outs;
	node * image = calloc(sizeof(node) * dimx * dimy);

	cudaMalloc(&in, bufSize,....);
	cudaMalloc(&outs, bufSize,....);

	pthread_t tid;
	sem_t sem;
	sem_init(&sem, 0, 1);
	t_data threadAgs;
	threadArgs.image = image;
	threadArgs.size = bufSize;
	threadArgs.charBuf = outs;
	


	srand(time(0));

	for (int i = 0; i < 100; ++i) {
		for (int i = 0; i < bufSize; ++i)
			buf[i] = -1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0)));
		cudaMemCpy(ins, buf ...);
		dim3 threadsPerBlock(totalThreads/totalBlocks);
		dim3 numBlocks(totalBlocks);

		sem_wait(&sem);
			ifsKernel<<<numBlocks,threadsPerBlock>>>(ins, outs);
			pthread_create(&tid, NULL, writerThread, (void *) &threadArgs);
			cudaMemCpy(charBuf, outs ...);
			writeToImage
			sempost();
			join()
		  }
		

	}

	


}
