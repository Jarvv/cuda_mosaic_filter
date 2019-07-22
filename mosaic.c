#include <time.h>
#include "mosaic.cuh"
	
#define FAILURE 0
#define SUCCESS !FAILURE
#define USER_NAME "aca15jh"

// Iteration 1,2,2.5,3
#define ITERATION 2.5

// CUDA Iteration 1,2,3
#define CUDA_ITERATION 1

int process_command_line(int argc, char *argv[]);
int isPowerOfTwo(unsigned int n);
void print_help();
void savePPM(ppm *image);
void runCPU(ppm *image);
void runOpenMP(ppm *image);
void runCuda(ppm *image);

// Command line argument vars
unsigned int c = 0;
MODE execution_mode;
const char *input_filename;
const char *output_filename;
const char *output_format;

int main(int argc, char *argv[]) {
	if (process_command_line(argc, argv) == FAILURE)
		return 1;

	// Get and set the max threads for omp
	int max_threads = omp_get_max_threads();
	omp_set_num_threads(max_threads);

	ppm *image;

	//TODO: read input image file (either binary or plain text PPM) 
	image = read_file(input_filename);

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode){
		case (CPU) : {
			runCPU(image);
			break;
		}
		case (OPENMP) : {
			runOpenMP(image);
			break;
		}
		case (CUDA) : {
			runCuda(image);
			break;
		}
		case (ALL) : {
			runCPU(image);
			runOpenMP(image);
			runCuda(image);
			break;
		}
	}
	
 	return 0;
}

void print_help(){
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		   "\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		   "\t               ALL. The mode specifies which version of the simulation\n"
		   "\t               code should execute. ALL should execute each mode in\n"
		   "\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		   "\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		   "\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]){
	if (argc < 7){
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name

	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);
	// Check if power of 2
	if (isPowerOfTwo(c) == 0) {
		printf("C should be a power of 2 \n");
		return FAILURE;
	}

	//TODO: read in the mode
	const char* mode = argv[2];
	if (strcmp("CPU",mode) == 0) execution_mode = CPU;
	else if (strcmp("CUDA", mode) == 0) execution_mode = CUDA;
	else if (strcmp("OPENMP", mode) == 0) execution_mode = OPENMP;
	else if (strcmp("ALL", mode) == 0) execution_mode = ALL;
	else {
		printf("Invalid mode...\n");
		return FAILURE;
	}
		
	//TODO: read in the input image name
	input_filename = argv[4];

	//TODO: read in the output image name
	output_filename = argv[6];
	const char *file_extenstion = strrchr(output_filename, '.')+1;
	if (strcmp(file_extenstion,"ppm") == 1) {
		printf("Invalid output file extension...\n");
		return FAILURE;
	}

	//TODO: read in any optional part 3 arguments
	// If isn't either binary or plain text, default to binary
	output_format = argv[7];
	if (strcmp("PPM_BINARY", output_format) == 1 && strcmp("PPM_PLAIN_TEXT", output_format) == 1) output_format = "PPM_BINARY";

	return SUCCESS;
}

void runCPU(ppm *image){
	ppm *mosaic;
	double begin, end, seconds;

	// Begin timing
	begin = omp_get_wtime();

	// Calculate the average colour value
	static unsigned long average_colour[3] = { 0,0,0 };
	average_colour_value(image, average_colour);

	// Mosaic filter
	mosaic = do_mosaic(image);

	// Output the average colour value for the image
	fprintf(stderr, "CPU Average image colour red = %d, green = %d, blue = %d \n", average_colour[0], average_colour[1], average_colour[2]);

	// End timing
	end = omp_get_wtime();
	seconds = (end - begin);

	fprintf(stderr, "CPU mode execution time took %.4f s\n", seconds);

	savePPM(mosaic);
}

void runOpenMP(ppm *image) {
	ppm *mosaic;
	double begin, end, seconds;

	// Start timing
	begin = omp_get_wtime();

	static unsigned long average_colour[3] = { 0,0,0 };

	// Perform different iterations of the functions depending on the value
	// of iteration, default to 3 if not given.
	if (ITERATION == 1) {
		average_colour_value_OPENMP_I(image, average_colour);
		mosaic = do_mosaic_OPENMP_I(image);
	}
	else if (ITERATION == 2) {
		// Allow nesting
		//omp_set_nested(1);

		average_colour_value_OPENMP_II(image, average_colour);
		mosaic = do_mosaic_OPENMP_II(image);
	}
	else if (ITERATION == 2.5) {
#pragma omp parallel 
#pragma omp sections nowait
		{
#pragma omp section 
			{

				average_colour_value_OPENMP_II(image, average_colour);
			}
#pragma omp section
			{
				mosaic = do_mosaic_OPENMP_II(image);
			}
		}
	}
	else {
		average_colour_value_OPENMP_II(image, average_colour);
		mosaic = do_mosaic_OPENMP_III(image);
	}

	// Output the average colour value for the image
	fprintf(stderr, "OPENMP Average image colour red = %d, green = %d, blue = %d \n", average_colour[0], average_colour[1], average_colour[2]);

	// End timing
	end = omp_get_wtime();
	seconds = (end - begin);
	fprintf(stderr, "OPENMP mode execution time took %.4f s\n", seconds);

	savePPM(mosaic);
}

void runCuda(ppm *image) {
	ppm *mosaic;
	mosaic = perform_CUDA(image, CUDA_ITERATION);
	savePPM(mosaic);
}

void savePPM(ppm *image) {
	FILE *f = NULL;

	// Output format
	if (strcmp(output_format,"PPM_BINARY") == 0){
		f = fopen(output_filename, "wb");
		fprintf(f, "P6\n");
	}
	else if (strcmp(output_format, "PPM_PLAIN_TEXT") == 0) {
		f = fopen(output_filename, "w");
		fprintf(f, "P3\n");
	}

	// Random comment
	fprintf(f, "# Test Comment\n");

	// Dimensions
	fprintf(f, "%d %d\n", image->width, image->height);

	// Colour depth
	fprintf(f, "%d\n", 255);

	// Data
	unsigned char currentPixel[3];
	for (int i = 0; i < image->width * image->height; i++) {
		currentPixel[0] = image->rgb[i].x;
		currentPixel[1] = image->rgb[i].y;
		currentPixel[2] = image->rgb[i].z;
		
		if (strcmp(output_format, "PPM_BINARY") == 0) {
			fwrite(currentPixel, 3, 1, f);
		}
		else if (strcmp(output_format, "PPM_PLAIN_TEXT") == 0) {
			// When have reached the end pixel, print a new line
			if(i != 0 && (i+1) % image->width == 0) fprintf(f, "%d %d %d \n", currentPixel[0], currentPixel[1], currentPixel[2]);
			else fprintf(f, "%d %d %d \t", currentPixel[0], currentPixel[1], currentPixel[2]);
		}
		
	}

	fclose(f);

	free(image->rgb);
	free(image);
}

// A standard way of finding out if an int is a power of 2
int isPowerOfTwo(unsigned int n)
{
	if (n == 0) return 0;
	while (n != 1) {
		if (n % 2 != 0) return 0;
		n /= 2;
	}
	return 1;
}