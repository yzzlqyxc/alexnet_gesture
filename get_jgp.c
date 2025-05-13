#include "get_jgp.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

inline int16_t float_to_fixed(float input) {
    return (int16_t)(input * 256.0f);
}

void resample(int w, int h, float** f){
	int tw = 32, th = 32, ch = 3;

	float dx = w / (float)tw, dy = h / (float)th;
	float *pic[3];
	for(int i = 0; i < 3; i ++) pic[i] = (float*)malloc(tw*th*sizeof(float));

	for(int i = 0; i < ch; i ++ ) {
		for(int j = 0; j < tw; j ++ ) {
			for(int k = 0; k < th; k ++ ) {
				float x = dx*j, y = dy*k, res = 0;
				int fx = floor(x), fy = floor(y);
				float a = x-fx, b = y-fy;
				float lu = f[i][fy*h+fx], ld = f[i][(fy+1)*h+fx], ru = f[i][fy*h+fx+1], rd = f[i][(fy+1)*h+fx+1];
				pic[i][k*th+j] = (1-a)*(1-b)*lu+a*(1-b)*ru+(1-a)*b*ld+a*b*rd;
			}
		}
	}

}

void read_jpg(char* file_name, graph_t *graph)
{
	FILE *infile = fopen(file_name, "rb");
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	cinfo.err = jpeg_std_error(&jerr);

	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, infile);
	int rc = jpeg_read_header(&cinfo, TRUE);  //read the jpeg header
	if (rc != 1) {
		printf("JPEG file read fail!!\n");
		exit(rc);
	}

	jpeg_start_decompress(&cinfo);
	int width = cinfo.output_width, cnt = 0;
	int height = cinfo.output_height;
	int channels = cinfo.output_components;
	float *f = (float*)malloc(sizeof(float)*width*height*channels);

	graph->height = height;
	graph->width = width;


	printf("JPEG image is %d x %d with %d color channels\n", width, height, channels);
	printf("%d %d %d\n", cinfo.rec_outbuf_height, cinfo.output_scanline, cinfo.output_height);

	unsigned char *buf[1];
	buf[0] = (unsigned char*)malloc(sizeof(unsigned char)*width*cinfo.output_components);

	while (cinfo.output_scanline < cinfo.output_height) {
		jpeg_read_scanlines(&cinfo, &buf[0], 1);
		for(int i = 0; i < width; i ++ ) {
			printf("%d ", buf[0][i]);
			f[cnt++] = buf[0][i]/255.0;
		}
	}
	float *pixel[3];
	resample(width, height, pixel);

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(infile);


}
