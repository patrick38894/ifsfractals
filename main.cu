#include "engine.h"

sem_t sem;

int main(int argc, char ** argv) {
	
	int dimx = 1920;
	int dimy = 1080;
	
	const int xyz = 3; //xyzrgba
	const int rgba = 4;
	const int total_threads = 4096;
	const int total_blocks= 32; 
	const int buf_size = total_threads * (xyz+rgba);
	const int gpu_iterations = 128;
	float pt_buf[buf_size];

	float * gpu_pt_buf;
	int * gpu_coord_buf;
	float * gpu_z_buf;
	unsigned char * gpu_char_buf;

	image_t image;
	image.data = (node_t **) calloc(dimx * dimy, sizeof(node_t *));
	image.xdim = dimx;
	image.ydim = dimy;
	image.num_nodes = 0;
	
	gpuErrchk(cudaMalloc((void **) &gpu_pt_buf, buf_size * sizeof(float)));
	gpuErrchk(cudaMalloc((void **) &gpu_char_buf, gpu_iterations * total_threads * rgba * sizeof(unsigned char)));
	gpuErrchk(cudaMalloc((void **) &gpu_coord_buf, gpu_iterations * total_threads * 2 * sizeof(int)));
	gpuErrchk(cudaMalloc((void **) &gpu_z_buf, gpu_iterations * total_threads * sizeof(float)));

	pthread_t tid;
	sem_init(&sem, 0, 1);
	t_data thread_args;
	thread_args.image = image;
	thread_args.num_elements = total_threads * gpu_iterations;
	thread_args.num_dims = xyz;
	thread_args.gpu_char_buf = gpu_char_buf;
	thread_args.gpu_z_buf = gpu_z_buf;
	thread_args.gpu_coord_buf = gpu_coord_buf;

	srand(time(0));
	
	int iterations = 512;
	printf("generating fractal\n");
	for (int i = 0; i < iterations; ++i) {
		for (int j = 0; j < buf_size; ++j)
			pt_buf[j] = -1.0 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(2.0)));

		//////////////////////////
		//printf("iteration %d\n", i );
		gpuErrchk(cudaMemcpy(gpu_pt_buf, pt_buf, buf_size * sizeof(float), cudaMemcpyHostToDevice));
		dim3 threads_per_block(total_threads/total_blocks);
		dim3 num_blocks(total_blocks);

		sem_wait(&sem);
		ifs_kernel<<<num_blocks,threads_per_block>>>(gpu_pt_buf, dimx, dimy, gpu_coord_buf, gpu_z_buf, gpu_char_buf, gpu_iterations);
		///////////////////////
		pthread_create(&tid, NULL, writer_thread, (void *) &thread_args);
		//if (i%10 == 0)
			printf("%d%% complete\n", 100*i/iterations);
	}
	sem_wait(&sem);

	gpuErrchk(cudaFree(gpu_pt_buf));
	gpuErrchk(cudaFree(gpu_char_buf));
	gpuErrchk(cudaFree(gpu_z_buf));
	gpuErrchk(cudaFree(gpu_coord_buf));
	unsigned char * ppm_buf = (unsigned char *) malloc(sizeof(unsigned char) * dimx * dimy * 3);
	render_image(image, ppm_buf);
	writePPM(ppm_buf, dimx, dimy, "my_fractal.ppm");
	free(ppm_buf);
	return 0;
	
}
