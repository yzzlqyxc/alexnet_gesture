#include<stdio.h>
#include "get_jgp.h"
#include "network.h"

int main() 
{
	
	int n = 10;
	Conv_IO* conv = CreateConv(10, 10, 10); 

	char* filename = "./training/weights.txt";
	char* picname = "./training/data/Dataset/0/IMG_1118.JPG";

	printf("%s\n", filename);
	Network nt = ReadNetwork(filename);



	graph_t graph;
	char *graph_path = "./training/data/Dataset/0/IMG_1118.JPG";
	read_jpg(graph_path, &graph);


	return 0;
}
