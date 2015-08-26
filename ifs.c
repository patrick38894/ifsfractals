#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>



 
int writePPM(char * buf, int dimx, int dimy, const char * name)
{
  int i, j;
  FILE *fp = fopen(name, "wb"); /* b - binary mode */
  (void) fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
  for (j = 0; j < dimy; ++j)
  {
    for (i = 0; i < dimx; ++i)
    {
      static unsigned char color[3];
      color[0] = buf[j*dimx + i] * (255/8);  /* red */
      color[1] = color[0];  /* green */
      color[2] = color[0];  /* blue */
      (void) fwrite(color, 1, 3, fp);
    }
  }
  (void) fclose(fp);
  return 1;
}

typedef struct vec2 {
	double x; double y;
} vec2;

void f5(vec2 * in, vec2 ** out) {
	*out = malloc(sizeof(vec2));
	(*out)->x = in->x + 0.1 * in->y;
	(*out)->y = in->y;
}

void f0(vec2 * in, vec2 ** out) {
	*out = malloc(sizeof(vec2));
	(*out)->x = in->x;
	(*out)->y = sqrt(3*in->y);
}
 
void f4(vec2 * in, vec2 ** out) {
	*out = malloc(sizeof(vec2));
	(*out)->x = (in->x + in->y) / 2.0;
	(*out)->y = 0;
}

void f3(vec2 * in, vec2 ** out) {
	*out = malloc(sizeof(vec2));
	(*out)->x = sqrt(in->x *in->x + in->y * in->y);
	(*out)->y = atan2(in->y,in->x);
}

void f2(vec2 * in, vec2 ** out) {
	*out = malloc(sizeof(vec2));
	(*out)->x = in->y / 0.7;
	(*out)->y = sin(in->y + in->x) ;
}

void f1(vec2 * in, vec2 ** out) {
	*out = malloc(sizeof(vec2));
	(*out)->x = atan(in->x);
	(*out)->y = tan(in->y);
}

void plot(unsigned char * buf, int x, int y, vec2 * pt) {
	int xdest = (int) ((pt->x + 1) * ((double) x / 2.0));
	int ydest = (int) ((pt->y + 1) * ((double) y / 2.0));
	if (buf [ ydest * x + xdest] < 8)
		buf[ydest*x+xdest] += 1;

}

int main (int argc, char ** argv) {
	int edges[6]= {2,4,4,2,4,2};
	int graph[6][4]= {{1,2,-1,-1}, {0,2,3,4}, {0,1,4,5}, {1,4,-1,-1}, {1,2,3,5}, {2,4,-1,-1}}; //triangle
	void (*funcs[6]) (vec2 *, vec2 **) = {f0, f1, f2, f3, f4, f5};

	
	int x = 1920;
	int y = 1080;
	
	unsigned char * buf = malloc(sizeof(unsigned char) * x * y);
	for (int i = 0; i < x*y; ++i)
		buf[i] = 0;

	srand(time(NULL));
	
	for (int i = 0; i < 500000; ++i) {
		vec2 * pt = malloc(sizeof(vec2));
		vec2 * temp;
		int pos = rand() % 6;
		pt->x = -1.0 + 2 * ((double)rand()/(double)RAND_MAX);
		pt->y = -1.0 + 2 * ((double)rand()/(double)RAND_MAX);
		for (int j = 0; j < 1000; ++j) {
			int dest = rand() % edges[pos];
			(funcs[pos])(pt,&temp);
			free(pt);
			pt = temp;
			if (j > 30) {
				if ((pt->x < 1.0 && pt->x > -1.0) && (pt->y < 1.0 && pt->y > -1.0))
					plot(buf, x, y, pt);
				pos = graph[pos][dest];
			}
		}
		free(pt);
	}
	writePPM(buf, x, y, "myPicture.ppm");
	return 0;
}
