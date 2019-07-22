#include "mosaic.cuh"

#define BUFFER_SIZE 32
extern unsigned int c;

int readLine(FILE *f, char buffer[], char read_for);
void checkForComment(FILE *f);

ppm *read_file(const char *input_filename) {
	ppm *image;
	char buffer[BUFFER_SIZE];
	FILE *f = NULL;

	// Try and open the file
	f = fopen(input_filename, "rb");
	// If the file cant be found, display this to the user.
	if (!f) {
		fprintf(stderr, "File %s not found...", input_filename);
		exit(1);
	}

	// Read the first line of the header which will indicate if the
	// file is plain text or binary.
	readLine(f, buffer, '\n');

	// If P3, file is plain texts
	if (buffer[0] == 'P' && buffer[1] == '3') {
		image = readPlainTextFile(f, buffer);
		return image;
	}
	// If P6, file is binary
	else if (buffer[0] == 'P' && buffer[1] == '6') {
		image = readBinaryFile(f);
		return image;
	}
	// If this can't be read then the file may be wrong.
	else {
		printf("Invalid file format \n");
		fclose(f);
		exit(1);
	}
}

ppm *readPlainTextFile(FILE *f, char buffer[]) {
	ppm *image;
	int colour_depth;
	int cell_size = c;

	// Allocate memory for the image
	image = (ppm *)malloc(sizeof(ppm));

	// Check for comments
	checkForComment(f);

	// Get the width and height
	readLine(f, buffer, '\n');
	sscanf(buffer, "%d", &image->width);

	// Check for comments
	checkForComment(f);

	readLine(f, buffer, '\n');
	sscanf(buffer, "%d", &image->height);
	// Check that c is not greater than the width or height
	if (image->height < cell_size || image->width < cell_size) {
		printf("C cannot be greater than the width or height of the image");
		exit(1);
	}

	// Check for comments
	checkForComment(f);

	// Get the colour depth, terminate if not 255
	readLine(f, buffer, '\n');
	sscanf(buffer, "%d", &colour_depth);
	if (colour_depth != 255) {
		printf("Invalid rgb component \n");
		exit(1);
	}

	// Allocate memory to the rbg struct of the image
	image->rgb = (uchar4*)malloc(image->width * image->height * sizeof(uchar4*));

	// For each pixel, read the three rgb values and add them to the image
	for (int i = 0; i < image->width * image->height; i++) {
		if (i != 0 && (i + 1) % image->width == 0) readLine(f, buffer, '\n');
		else readLine(f, buffer, '\t');

		sscanf(buffer, "%hhd %hhd %hhd", &image->rgb[i].x, &image->rgb[i].y, &image->rgb[i].z);
		if (colour_depth < image->rgb[i].x || colour_depth < image->rgb[i].y || colour_depth < image->rgb[i].z) {
			printf("Pixel rgb cannot be greater than the specified colour depth \n");
			exit(1);
		}
	}

	fclose(f);

	return image;
}

ppm *readBinaryFile(FILE *f) {
	ppm *image;
	int colour_depth;
	int cell_size = c;
	
	// Allocate memory for the image
	image = (ppm *)malloc(sizeof(ppm));

	// Check for comments
	checkForComment(f);

	// Get width and height
	fscanf(f, "%d %d", &image->width, &image->height);
	if (image->height < cell_size || image->width < cell_size) {
		printf("C cannot be greater than the width or height of the image");
		exit(1);
	}

	// Check for comments
	checkForComment(f);

	// Get the colour depth, terminate if not 255
	fscanf(f, "%d", &colour_depth);
	if (colour_depth != 255) {
		printf("Invalid rgb component \n");
		exit(1);
	}

	// End the plain text line
	while (fgetc(f) != '\n');

	// Allocate memory to the rbg struct of the image
	image->rgb = (uchar4*)malloc(image->width * image->height * sizeof(uchar4*));

	// For each pixel, read the three rgb values and add them to the image
	unsigned char currentPixel[3];

	// For each line in the data, read to the pixel array and assign the image
	// at that index the rgb values.
	for (int i = 0; i < image->width * image->height; i++) {
		fread(currentPixel, 3, 1, f);
		if (colour_depth < currentPixel[0] || colour_depth < currentPixel[1] || colour_depth < currentPixel[2]) {
			printf("Pixel rgb cannot be greater than the specified colour depth \n");
			exit(1);
		}
		image->rgb[i].x = currentPixel[0];
		image->rgb[i].y = currentPixel[1];
		image->rgb[i].z = currentPixel[2];
	}

	fclose(f);

	return image;
}

// Check for a comment in the current line, if the # is found, read
// till the next new line.
void checkForComment(FILE *f) {
	unsigned char ci;

	ci = getc(f);
	while (ci == '#') {
		while (getc(f) != '\n');
		ci = getc(f);
	}
	ungetc(ci, f);
}

// This code was taken from the lab class, but with the addition of the
// read_for parameter so I can read the new line or tab space sequences.
int readLine(FILE *f, char buffer[], char read_for) {
	
	int i = 0;
	char c;
	while ((c = getc(f)) != read_for) {
		if (c == EOF)
			return 0;
		buffer[i++] = c;
		if (i == BUFFER_SIZE) {
			fprintf(stderr, "Buffer size is too small for line input\n");
			exit(0);
		}
	}
	buffer[i] = '\0';

	if (strncmp(buffer, "exit", 4) == 0)
		return 0;
	else
		return 1;
}