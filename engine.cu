#include "engine.h"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

__global__ void ifs_kernel(float * pts, int xdim, int ydim, int * pt_out_buf, float * zbuf, unsigned char * rgba_out_buf, int iterations) {
	float outs[7];
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	
	int initial_iterations = 20;
	for (int k = 0; k < initial_iterations + iterations; ++k) {
		switch (blockIdx.x % 4) {
			case (1): {
				for (int i = 0; i < 7; ++i)
					outs[i] = sin(pts[idx+i]) + sqrt(abs(pts[idx+((i+1)%7)]));
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
					outs[i] = (pts[idx+i]+pts[idx+((i+2)%7)])/1.2;
				break;
			}
			case (0): {
				for (int i = 0; i < 7; ++i)
					outs[i] = atan(pts[idx+i]+pts[(idx+i*i%7)])/2.0;
				break;
			}
		}
		//remap
		for (int i = 0; i < 7; ++i)
			pts[idx+i] = outs[i];


		
		//write out
		if (k >= initial_iterations) {
			int j = k - initial_iterations;
			//tranform
			//left blank for now

			//quantize
			float viewbox_size = 4.0;
			pt_out_buf[2*idx*iterations+j*2] = (int) (outs[0]/viewbox_size * (float) xdim /2.0 + ((float) xdim /2.0));
			pt_out_buf[2*idx*iterations+j*2+1] = (int) (outs[1]/viewbox_size * (float) ydim /2.0 + ((float) ydim /2.0));
			zbuf[idx*iterations+j] = outs[2];

			//rgba
			rgba_out_buf[4*idx*iterations+j*4] = (unsigned char) (127.0 + outs[3] * 127.0);
			rgba_out_buf[4*idx*iterations+j*4+1] = (unsigned char) (127.0 + outs[4] * 127.0);
			rgba_out_buf[4*idx*iterations+j*4+2] = (unsigned char) (127.0 + outs[5] * 127.0);
			rgba_out_buf[4*idx*iterations+j*4+3] = (unsigned char) (127.0 + outs[6] * 127.0);
		}
	}
}

void * writer_thread(void * arg) {


	t_data * input = (t_data *) arg;
	unsigned char * temp_char_buf = (unsigned char *) malloc(4 * input->num_elements * sizeof(unsigned char));
	float * temp_z_buf = (float *) malloc(input->num_elements * sizeof(float));
	int * temp_coord_buf = (int *) malloc(2 * input->num_elements * sizeof(int));
	gpuErrchk(cudaMemcpy(temp_char_buf, input->gpu_char_buf, 4 * input->num_elements * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_z_buf, input->gpu_z_buf, input->num_elements * sizeof(float), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(temp_coord_buf, input->gpu_coord_buf, 2 * input->num_elements * sizeof(float), cudaMemcpyDeviceToHost));
	for (int i = 0; i < input->num_elements; ++i) {
		int x = temp_coord_buf[i*2];
		int y = temp_coord_buf[i*2+1];
		//printf("inserting node at position (%d,%d)\n", x, y);
		node_t * node = (node_t *) malloc(sizeof(node_t));
		node->r = temp_char_buf[i*4];
		node->g = temp_char_buf[i*4+1];
		node->b = temp_char_buf[i*4+2];
		node->a = temp_char_buf[i*4+3];
		node->z = temp_z_buf[i];
		node->next = NULL;

		//help blurring
		node->a /= 4;
	
		insert_node(input->image, x, y, node);

		//pixel blurring
		int diameter = 7;
		int d2 = diameter /2;
		for (int j = 0; j < diameter; ++j) {
			for (int k = 0; k < diameter; ++k) {
				if (k == d2 && j == d2)
					continue;
				node_t * temp = (node_t *) malloc(sizeof(node_t));
				memcpy(temp, node, sizeof(node_t));
				int tempx = x -d2 + j;
				int tempy = y -d2 + k;
				temp->a = node->a/static_cast<unsigned char>((int)(j-d2)*(j-d2)+(k-d2)*(k-d2))/2;
				temp->r = node->r/static_cast<unsigned char>((int)(j-d2)*(j-d2)+(k-d2)*(k-d2))/2;
				temp->g = node->g/static_cast<unsigned char>((int)(j-d2)*(j-d2)+(k-d2)*(k-d2))/2;
				temp->b = node->b/static_cast<unsigned char>((int)(j-d2)*(j-d2)+(k-d2)*(k-d2))/2;
				insert_node(input->image, tempx, tempy, temp);
			}
		}
		///
	}
	free(temp_z_buf);
	free(temp_char_buf);
	free(temp_coord_buf);
	sem_post(&sem);
	pthread_exit(0);
}

