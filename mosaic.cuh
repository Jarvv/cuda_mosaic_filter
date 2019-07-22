#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <vector_types.h>
#include <vector_functions.h>

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;

// The rgb stuct containing red, green and blue unsigned chars.
typedef struct {
	unsigned char red, blue, green;
}rgb;

// The PPM image stuct, containing the width, height and an rgb array.
typedef struct {
	int width;
	int height;
	uchar4 *rgb;
}ppm;

// Read PPM
ppm * read_file(const char *input_filename);
ppm *readPlainTextFile(FILE *f, char buffer[]);
ppm *readBinaryFile(FILE *f);

// CPU functions
void average_colour_value(ppm *image, unsigned long averageColour[]);
ppm *do_mosaic(ppm *image);

// OPENMP functions
void average_colour_value_OPENMP_I(ppm *image, unsigned long averageColour[]);
void average_colour_value_OPENMP_II(ppm *image, unsigned long averageColour[]);
ppm *do_mosaic_OPENMP_I(ppm *image);
ppm *do_mosaic_OPENMP_II(ppm *image);
ppm *do_mosaic_OPENMP_III(ppm *image);

// CUDA host functions
ppm* perform_CUDA(ppm *h_image, int iteration);
void checkCUDAError(const char *msg);
unsigned int div_round(unsigned int a, unsigned int b);