#include<stdio.h>
#include <stdlib.h>
#include "network.h"

int main() 
{
	int n = 10;
	Conv_IO* conv = CreateConv(10, 10, 10); 

	char* filename = "./training/weights.txt";
	printf("%s\n", filename);
	Network nt = ReadNetwork(filename);

	return 0;
}
