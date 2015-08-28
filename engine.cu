#include "engine.cuh"

__global__ void ifs_kernel(float * pts, unsigned char * out_buf) {
	float outs[7];
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	switch (blockIdx.x) {
		case (1): {
			for (int i = 0; i < 7; ++i)
				outs[i] = sin(pts[idx+i]);
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
				outs[i] = (pts[idx+i]+pts[(idx+i+1)%7])/3.0;
			break;
		}
		case (4): {
			for (int i = 0; i < 7; ++i)
				outs[i] = atan(pts[idx+i]+pts[(idx+i*i)%7])/3.0;
			break;
		}
		//tranform
		//quantize
		//mix rgba
		//add to outBuf
	}
}

void * writer_thread(void * arg) {
	t_data * input = (t_data *) arg;
	cudaMemcpy(input->char_buf, input->gpu_char_buf, input->size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	for (int i = 0; i < input->size; ++i) {
		int x = input->char_buf[i*7+0];
		int y = input->char_buf[i*7+1];
		
		node_t * node = (node_t *) malloc(sizeof(node_t));
		node->r = input->char_buf[i*7+3];
		node->g = input->char_buf[i*7+4];
		node->b = input->char_buf[i*7+5];
		node->a = input->char_buf[i*7+6];
		node->z = input->char_buf[i*7+2];
	
		insert_node(input->image, x, y, node);
	}
	sem_post(input->sem);
	pthread_exit();
}

void main(int argc, char ** argv) {
	
	int dimx = 500;
	int dimy = 500;
	
	const int pix_size = 7; //xyzrgba
	const int total_threads = 1024;
	const int total_blocks= 4; 
	const int buf_size = total_threads * pix_size;
	float pt_buf[buf_size];
	float gpu_pt_buf[buf_size];
	unsigned char char_buf[buf_size];
	unsigned char * gpu_char_buf;

	image_t image;
	image.data = calloc(sizeof(node_t *) * dimx * dimy);
	image.xdim = xdim;
	image.ydim = ydim;
	
	cudaMalloc(&gpu_pt_buf, buf_size * sizeof(float));
	cudaMalloc(&gpu_char_buf, buf_size * sizeof(unsigned char));

	pthread_t tid;
	sem_t sem;
	sem_init(&sem, 0, 1);
	t_data thread_args;
	thread_args.image = image;
	thread_args.size = buf_size;
	thread_args.char_buf = char_buf;
	thread_args.gpu_char_buf = gpu_char_buf;

	srand(time(0));

	for (int i = 0; i < 100; ++i) {
		for (int i = 0; i < buf_size; ++i)
			pt_buf[i] = -1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0)));
		cudaMemcpy(gpu_pt_buf, pt_buf, buf_size * sizeof(float), cudaMemcpyHostToDevice);
		dim3 threads_per_block(total_threads/total_blocks);
		dim3 num_blocks(total_blocks);

		sem_wait(&sem);

		ifsKernel<<<num_blocks,threads_per_block>>>(gpu_pt_buf, gpu_char_buf);
		pthread_create(&tid, NULL, writer_thread, (void *) &thread_args);
	}
	sem_wait(&sem);

	cudaFree(gpu_pt_buf);
	cudaFree(gpu_char_buf);
	free(pt_buf);
	free(char_buf);
	unsigned char * ppm_buf = malloc(sizeof(unsigned char) * dimx * dimy * 3);
	render_image(image, ppm_buf);
	writePPM(ppm_buf, dimx, dimy, "my_fractal.ppm");
	free(ppm_buf);
	return 0;
}
