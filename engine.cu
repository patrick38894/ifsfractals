#include "engine.h"

__global__ void ifs_kernel(float * pts, int xdim, int ydim, int * pt_out_buf, unsigned char * rgba_out_buf) {
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

		//quantize
		pt_out_buf[3*idx] = (int) (out[0] * (float) xdim * 1.5);
		pt_out_buf[3*idx+1] = (int) (out[1] * (float) ydim * 1.5);
		pt_out_buf[3*idx+2] = (int) (out[2] * (float));

		//rgba
		rgba_out_buf[4*idx] = (unsigned char) (outs[3] * 255.0);
		rgba_out_buf[4*idx+1] = (unsigned char) (outs[4] * 255.0);
		rgba_out_buf[4*idx+2] = (unsigned char) (outs[5] * 255.0);
		rgba_out_buf[4*idx+3] = (unsigned char) (outs[6] * 255.0);
	
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
	sem_post(&input->sem);
	pthread_exit(0);
}

