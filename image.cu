#include "image.h"

//naive implementation of rgba image rendering with zbuffer
int insert_node(image_t image, int x, int y, node_t * to_add) {
	int idx = y * image.xdim + x;
//	printf("adding node at coordinates (%d,%d)\n)", x, y);
	if ((x < 0 || x >= image.xdim) || (y < 0 || y >= image.ydim))
		return 0;
	node_t ** head_ref = &image.data[idx];
	if (*head_ref == NULL || (*head_ref)->z > to_add->z) {
		to_add->next = *head_ref;
		*head_ref = to_add;
		return 1;
	}
	node_t * current = *head_ref;
	int alphasum = 0;
	while (current->next && (to_add->z > current->next->z)) {
		alphasum += current->a;
		current = current->next;
	}
	if (alphasum > 250) {
		free(to_add);
		return 0;
	}
	to_add->next = current->next;
	current->next = to_add; 
	return 1;
}

unsigned char over(unsigned char a, unsigned char aa, unsigned char b, unsigned char ba) {
	return (unsigned char) (((float) a * (float) aa + (float) b * (float) ba * (1.0 - (float) aa))
				/((float) aa + (float) ba * (1.0 - (float) aa)));
}

void render_stack(node_t * head, unsigned char * r, unsigned char * g, unsigned char * b, unsigned char * a) {
/*	if (!head) {
		*r = *g = *b = 0;
		*a = 1;
		return;
	}
	render_stack(head->next, r, g, b, a);
	*r = over(head->r, head->a, *r, *a);
	*g = over(head->g, head->a, *g, *a);
	*b = over(head->b, head->a, *b, *a);
	*a = over(head->a, head->a, *a, *a);
*/


	//NOTE: in order to implement a quick fix for the stack overflow problem in the recursive version, i have reversed the order of the stack (placing the view on positive side of the z axis, looking back in the negative direction)

	*r = *g = *b = 0;
	*a = 255;
	node_t * current = head;
	while (current) {
		*r = over(current->r, current->a, *r, *a);
		*g = over(current->g, current->a, *g, *a);
		*b = over(current->b, current->a, *b, *a);
		*a = over(current->a, current->a, *a, *a);
		current = current->next;
	}
}

int render_image(image_t image, unsigned char * output) {
	printf("rasterizing:\n");
	for (int i = 0; i < image.xdim*image.ydim; ++i) {
		unsigned char r;
		unsigned char g;
		unsigned char b;
		unsigned char a;
		render_stack(image.data[i], &r, &g, &b, &a);
		output[i*3+0] = r;
		output[i*3+1] = g;
		output[i*3+2] = b;
		//if (i % 1000 == 0)
		//	printf("%d%% complete\n", i*100/(image.xdim*image.ydim));
	}
	return 1;
}

