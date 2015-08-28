typedef node_t {
        unsigned char r;
        unsigned char g;
        unsigned char b;
        unsigned char a;
        unsigned char z;
        node * next;
} node_t;

typedef image_t {
	int xdim;
	int ydim;
	node_t ** data;
} image_t;

int insert_node(image_t image, int x, int y,  node_t * to_add);
unsigned char over(unsigned char a, unsigned char aa, unsigned char b, unsigned char ba) {
int render_image(image_t image, unsigned char * output) {
void render_stack(node_t * head, unsigned char * r, unsigned char * g, unsigned char * b, unsigned char * a) {


