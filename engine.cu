#include "engine.h"

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

__global__ void ifs_kernel(float * pts, int xdim, int ydim, int * pt_out_buf, float * zbuf, unsigned char * rgba_out_buf) {
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
		//left blank for now
	}
		//quantize
	pt_out_buf[2*idx] = (int) (outs[0] * (float) xdim /2.0 + ((float) xdim /2.0));
	pt_out_buf[2*idx+1] = (int) (outs[1] * (float) ydim /2.0 + ((float) ydim /2.0));
	zbuf[idx] = outs[2];

	//rgba
	rgba_out_buf[4*idx] = (unsigned char) (outs[3] * 255.0);
	rgba_out_buf[4*idx+1] = (unsigned char) (outs[4] * 255.0);
	rgba_out_buf[4*idx+2] = (unsigned char) (outs[5] * 255.0);
	rgba_out_buf[4*idx+3] = (unsigned char) (outs[6] * 255.0);
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
	
		insert_node(input->image, x, y, node);
	}
	free(temp_z_buf);
	free(temp_char_buf);
	free(temp_coord_buf);
	sem_post(&sem);
	pthread_exit(0);
}

