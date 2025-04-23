#pragma once
#ifndef NETWORK
#define NETWORK

typedef struct {
    int w;
    int h;
    int d;
    float *** parmas;
} Conv_IO;

typedef  struct{
	int out;
	int in;
  float* parmas;
  float* bias;
} Conv;

typedef  struct{
	int out;	
	int in;	
	float* parmas;
  float* bias;
} Fc;

typedef struct {
	Conv con1;
	Conv con2;
	Conv con3;
	Conv con4;
	Fc fc1;
	Fc fc2;
} Network;

Conv_IO* CreateConv(int w, int h, int d);
Network ReadNetwork(char* filename);

#endif
