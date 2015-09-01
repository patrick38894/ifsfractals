#include "image.h"

//naive implementation of rgba image rendering with zbuffer
int insert_node(image_t image, int x, int y,  node_t * to_add) {
	int idx = y * image.xdim + x;
	node_t * current = image.data[idx];
	if (!current) {
		image.data[idx] = to_add;
		return 1;
	}
	node_t * prev = current;
	while (current && (to_add->z > current->z)) {
		prev = current;
		current = current->next;
	}
	prev->next = to_add;
	to_add->next = current;
}

unsigned char over(unsigned char a, unsigned char aa, unsigned char b, unsigned char ba) {
	return (unsigned char) (((float) a * (float) aa + (float) b * (float) ba * (1.0 - (float) aa))
				/((float) aa + (float) ba * (1.0 - (float) aa)));
}

void render_stack(node_t * head, unsigned char * r, unsigned char * g, unsigned char * b, unsigned char * a) {
	if (!head) {
		*r = *g = *b = 0;
		*a = 1;
	}
	render_stack(head->next, r, g, b, a);
	*r = over(head->r, head->a, *r, *a);
	*g = over(head->g, head->a, *g, *a);
	*b = over(head->b, head->a, *b, *a);
	*a = over(head->a, head->a, *a, *a);
}

int render_image(image_t image, unsigned char * output) {
	for (int i = 0; i < image.xdim*image.ydim; ++i) {
		unsigned char r;
		unsigned char g;
		unsigned char b;
		unsigned char a;
		render_stack(image.data[i], &r, &g, &b, &a);
		output[i*3+0] = r;
		output[i*3+1] = g;
		output[i*3+2] = b;
	}
	return 1;
}

