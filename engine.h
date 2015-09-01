#ifndef ENGINE_H
#define ENGINE_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>

#include "image.h"
#include "ppm.h"

extern sem_t sem;

typedef struct {
        image_t image;
        int num_elements;
        int num_dims;
        unsigned char * gpu_char_buf;
        int * gpu_coord_buf;
	float * gpu_z_buf;
} t_data;


__global__ void ifs_kernel(float * pts, int xdim, int ydim, int * pt_out_buf, float * zbuf, unsigned char * out_buf);
void * writer_thread(void * arg);

#endif
