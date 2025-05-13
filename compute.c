#include "./network.h"

// 3*32*32 * 32*32*3*3 -> 32*32*32
void ComputeCov(Conv *conv, Conv_IO *in, Conv_IO *out) {
	// add padding
	int inw = in->w + 2, inh = in->h + 2;
	Conv_IO* in_padding = CreateConv(inw, inh, in->d);

	for(int k = 0; k < in->d; k ++ ) {
		for(int i = 0; i < inw; i ++ ) {
			for(int j = 0; j < inh; j ++ ) {
				if (i == 0 || i == inw - 1 || j == 0 || j == inh - 1) {
					in_padding->parmas[k][i][j] = 0;
				} else {
					in_padding->parmas[k][i][j] = in->parmas[in->d][i-1][j-1];
				}
			}
		}
	}


	// do compute
	out = CreateConv(in->w, in->h, conv->out);
	int w = out->w, h = out->h, d = out->d;
	for(int i = 0; i < w; i ++ ) {
		for(int j = 0; j < h; j ++ ) {
			for(int k = 0; k < d; k ++ ) {

			}
		}
	}
	

}
