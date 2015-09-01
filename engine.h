#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>

#include "image.h"
#include "ppm.h"

typedef struct {
        image_t image;
        int size;
        sem_t sem;
        unsigned char * char_buf;
        unsigned char * gpu_char_buf;
} t_data;


__global__ void ifs_kernel(float * pts, unsigned char * out_buf);
void * writer_thread(void * arg);

