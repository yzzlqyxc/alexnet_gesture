#include "network.h"
#include <stdio.h>
#include <stdlib.h>

Conv_IO* CreateConv(int w, int h, int d) 
{
    Conv_IO * conv = malloc(sizeof(Conv_IO));
    conv->w = w;
    conv->h = h;
    conv->d = d;

    float *** parmas;
    parmas = malloc(d * sizeof(float**));
    for(int i = 0; i < d; i ++ ) {
        parmas[i] = malloc(h*sizeof(float*));
        for(int j = 0; j < h; j ++ ) {
            parmas[i][j] = malloc(w*sizeof(float));
        }
    }

    for(int i = 0, cnt = 0; i < d; i ++ ) {
        for(int j = 0; j < h; j ++ ) {
            for(int k = 0; k < w; k ++ ) {
                parmas[i][j][k] = cnt ++;
            }
        }
    }
    conv->parmas = parmas;
    return conv;
}

Conv ReadConv(FILE* fp)
{
	Conv conv;
	int out, in, kw, kh, bias;
	fscanf(fp, "%d%d%d%d", &out, &in, &kh, &kw);

	int len = 9*out*in;
	conv.parmas = (float*)malloc(len*sizeof(float));
	for(int i = 0; i < len; i ++ )
		fscanf(fp, "%f", &conv.parmas[i]);

	fscanf(fp, "%d", &bias);
	conv.bias = (float*)malloc(bias*sizeof(float));
	for(int i = 0; i < bias; i ++ )
		fscanf(fp, "%f", &conv.bias[i]);

	conv.out = out;
	conv.in = in;
	return conv;
}

Fc ReadFc(FILE* fp)
{
	Fc fc;
	int out, in, bias;
	fscanf(fp, "%d%d", &out, &in);

	int len = out*in;
	fc.parmas = (float*)malloc(len*sizeof(float));
	for(int i = 0; i < len; i ++ )
		fscanf(fp, "%f", &fc.parmas[i]);

	fscanf(fp, "%d", &bias);
	fc.bias = (float*)malloc(bias*sizeof(float));
	for(int i = 0; i < bias; i ++ )
		fscanf(fp, "%f", &fc.bias[i]);

	return fc;
}

Network ReadNetwork(char* filename)
{
	Network nt; 
	FILE *fp = fopen(filename, "r");
	nt.con1 = ReadConv(fp);
	nt.con2 = ReadConv(fp);
	nt.con3 = ReadConv(fp);
	nt.con4 = ReadConv(fp);
	nt.fc1 = ReadFc(fp);
	nt.fc2 = ReadFc(fp);
	printf("%f\n", nt.fc2.bias[9]);

	return nt;
}
