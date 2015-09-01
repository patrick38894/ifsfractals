#include "engine.h"

int main(int argc, char ** argv) {
	
	int dimx = 500;
	int dimy = 500;
	
	const int xyz = 3; //xyzrgba
	const int rgba = 4;
	const int total_threads = 1024;
	const int total_blocks= 4; 
	const int buf_size = total_threads * (xyz+rgba);
	float pt_buf[buf_size];
	float gpu_pt_buf[buf_size];
	unsigned char char_buf[total_threads * rgba];

	int * gpu_coord_buf;
	float * gpu_z_buf;
	unsigned char * gpu_char_buf;

	image_t image;
	image.data = (node_t **) calloc(sizeof(node_t *), dimx * dimy);
	image.xdim = dimx;
	image.ydim = dimy;
	
	cudaMalloc((void **) &gpu_pt_buf, buf_size * sizeof(float));
	cudaMalloc((void **) &gpu_char_buf, buf_size * sizeof(unsigned char));
	cudaMalloc((void **) &gpu_coord_buf, total_threads * 2 * sizeof(int));
	cudaMalloc((void **) &gpu_z_buf, total_threads * sizeof(float));

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

		//////////////////////////
		cudaMemcpy(gpu_pt_buf, pt_buf, buf_size * sizeof(float), cudaMemcpyHostToDevice);
		dim3 threads_per_block(total_threads/total_blocks);
		dim3 num_blocks(total_blocks);

		sem_wait(&sem);

		ifs_kernel<<<num_blocks,threads_per_block>>>(gpu_pt_buf, dimx, dimy,
				gpu_coord_buf, gpu_z_buf, gpu_char_buf);
		///////////////////////
		pthread_create(&tid, NULL, writer_thread, (void *) &thread_args);
	}
	sem_wait(&sem);

	cudaFree(gpu_pt_buf);
	cudaFree(gpu_char_buf);
	cudaFree(gpu_z_buf);
	cudaFree(gpu_coord_buf);
	unsigned char * ppm_buf = (unsigned char *) malloc(sizeof(unsigned char) * dimx * dimy * 3);
	render_image(image, ppm_buf);
	writePPM(ppm_buf, dimx, dimy, "my_fractal.ppm");
	free(ppm_buf);
	return 0;
	
	//TODO: move all functions with cuda calls into the engine.cu file and wrap them as simply as possible in a C function
}
