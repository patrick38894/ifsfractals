#include "ppm.h"
 
int writePPM(unsigned char * buf, int dimx, int dimy, const char * name)
{
  int i, j;
  FILE *fp = fopen(name, "wb"); /* b - binary mode */
  (void) fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
  for (j = 0; j < dimy; ++j)
  {
    for (i = 0; i < dimx; ++i)
    {
      static unsigned char color[3];
      color[0] = buf[j*dimx*3 +i];  /* red */
      color[1] = buf[j*dimx*3 +i+1];  /* green */
      color[2] = buf[j*dimx*3 +i+2];  /* blue */
      (void) fwrite(color, 1, 3, fp);
    }
  }
  (void) fclose(fp);
  return 1;
}

