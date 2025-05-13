#ifndef GRAPH
#define GRAPH
#include<stdio.h>
#include <jpeglib.h>

typedef struct {
	int width;
	int height;
	float *pixel;
} graph_t;

void read_jpg(char* file_name, graph_t *graph);

#endif
