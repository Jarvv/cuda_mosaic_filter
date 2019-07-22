#include "mosaic.cuh"

extern unsigned int c;

// My first version experimenting with OpenMP, this used a standard parallel for loop
// with atomics for the red, green and blue totals.
void average_colour_value_OPENMP_I(ppm *image, unsigned long averageColour[]) {
	int num_pixels = image->width * image->height;

	unsigned long x = 0;
	unsigned long y = 0;
	unsigned long z = 0;

	int i;
#pragma omp parallel for shared(image) private(i)
	for (i = 0; i < num_pixels; i++) {
		uchar4 pixel = image->rgb[i];

#pragma omp atomic
			x += pixel.x;
		
#pragma omp atomic
			y += pixel.y;
		
#pragma omp atomic
			z += pixel.z;
		
	}

	averageColour[0] = x / num_pixels;
	averageColour[1] = y / num_pixels;
	averageColour[2] = z / num_pixels;

}

// My final version of the average colour value for OpenMP. This uses a reduction 
// instead to summate the total rgb values.
void average_colour_value_OPENMP_II(ppm *image, unsigned long averageColour[]) {
	int num_pixels = image->width * image->height;

	unsigned long x = 0;
	unsigned long y = 0;
	unsigned long z = 0;

	int i;
#pragma omp parallel for shared(image) private(i) reduction(+: x,y,z)
	for (i = 0; i < num_pixels; i++) {
		uchar4 pixel = image->rgb[i];
		x += pixel.x;
		y += pixel.y;
		z += pixel.z;
	}

	averageColour[0] = (x / num_pixels);
	averageColour[1] = (y / num_pixels);
	averageColour[2] = (z / num_pixels);

}

// My initial openmp implementation of the mosaic loop. This version has a few improvements to the code
// structure as well 1 parallel for loop over the mosaic tiles which gives it a perfomance increase
// over the CPU version.
ppm *do_mosaic_OPENMP_I(ppm *image) {
	ppm *mosaic;

	// Allocate memory for the mosaic image
	mosaic = (ppm *)malloc(sizeof(ppm));

	// As c is a global variable, make it local so it is able to be cached.
	int cell_size = c;
	int cell_pixels = cell_size * cell_size;

	int mos_width = 0;
	int mos_height = 0;

	// Check if the image will have partial mosaic cells
	bool partial_width = false;
	bool partial_height = false;
	if (image->width % cell_size != 0) partial_width = true;
	if (image->height % cell_size != 0) partial_height = true;

	// Get the number of 'tiles' in the mosaic image
	mosaic->width = image->width;
	mosaic->height = image->height;
	mos_width = (int)ceil((double)image->width / cell_size);
	mos_height = (int)ceil((double)image->height / cell_size);

	// Allocate memory to the rbg struct of the mosaic image
	mosaic->rgb = (uchar4*)malloc(mosaic->width * mosaic->height * sizeof(uchar4*));

	// The overflow of pixels that occurs due to the partial pixel due to the use of a 1 dimensional array
	int partial_size = (image->width - (cell_size * (mos_width - 1)));
	int overflow = cell_size - partial_size;

	int mosaic_row = 0;
	// For each 'tile' of the mosaic image
	int i = 0;
#pragma omp parallel for shared(image, mosaic) private(i,mosaic_row)
	for (i = 0; i < mos_width * mos_height; i++) {
		// Get the row that the tile occupies, as it is integer division the value will be truncated to the 
		// lower value.
		mosaic_row = (i / mos_width);

		unsigned long x = 0;
		unsigned long y = 0;
		unsigned long z = 0;
		short av_r = 0;
		short av_g = 0;
		short av_b = 0;

		int index = 0;
		int pixel_row = 0;

		// A counter is needed incase of a partial tile to ensure that the colour value is accurate
		int counter = 0;

		// For 2 times, 0 = work out rgb totals, 1 = assign averages to pixels 
		int k = 0;
#pragma omp parallel for private(k)
		for (k = 0; k < 2; k++) {
			// Assign the averages
			if (k == 1) {
				av_r = (short)(x / counter);
				av_g = (short)(y / counter);
				av_b = (short)(z / counter);
			}

			pixel_row = 0;
			// For c*c pixels in the mosaic tile
			for (int j = 0; j < cell_pixels; j++) {
				// If the cell isnt 0 AND (the image is partial AND is the last cell in the image)
				if (i != 0 && ((partial_width && ((mosaic_row*mos_width) + (mos_width - 1)) == i))) {
					if (j != 0 && j % partial_size == 0) {
						pixel_row += 1;
						j = (cell_size * pixel_row);
					}
				}
				else {
					// If j has reached the end of the 'pixel', increase the pixel row by 1
					if (j != 0 && (j % cell_size) == 0) pixel_row += 1;
				}

				index = (j - (pixel_row * c)) + (i * c) + image->width * (pixel_row + (mosaic_row * (c - 1)));
				if (mosaic_row > 0 && partial_width) index -= overflow * mosaic_row;

				// If j has reached the pixel height or the index has reached the image height, then we can stop for this pixel
				if (index >= image->height * image->width || (j >= cell_pixels)) break;

				// Calculate the totals and increase the counter for the pixel
				if (k == 0) {
					counter += 1;
					x += image->rgb[index].x;
					y += image->rgb[index].y;
					z += image->rgb[index].z;
				}
				// Assign the average mosaic colour to the pixel
				else if (k == 1) {
					mosaic->rgb[index].x = (unsigned char)av_r;
					mosaic->rgb[index].y = (unsigned char)av_g;
					mosaic->rgb[index].z = (unsigned char)av_b;
				}
			}
		}
	}
	return mosaic;
}

// My second openmp implementation focussed on more parallel for loops as well as testing to see if setting
// the nested attribute would have any positive effect. This meant restructuring the mosaic calculation loop
// further as in the previous versions j was being manipulated which wouldn't work for an openmp for loop.
ppm *do_mosaic_OPENMP_II(ppm *image) {
	ppm *mosaic;

	// Allocate memory for the mosaic image
	mosaic = (ppm *)malloc(sizeof(ppm));

	// As c is a global variable, make it local so it is able to be cached.
	int cell_size = c;
	int cell_pixels = cell_size * cell_size;

	int mos_width = 0;
	int mos_height = 0;

	// Check if the image will have partial mosaic cells
	bool partial_width = false;
	bool partial_height = false;
	if (image->width % cell_size != 0) partial_width = true;
	if (image->height % cell_size != 0) partial_height = true;

	// Get the number of 'tiles' in the mosaic image
	mosaic->width = image->width;
	mosaic->height = image->height;
	mos_width = (int)ceil((double)image->width / cell_size);
	mos_height = (int)ceil((double)image->height / cell_size);

	// Allocate memory to the rbg struct of the mosaic image
	mosaic->rgb = (uchar4*)malloc(mosaic->width * mosaic->height * sizeof(uchar4*));

	// The overflow of pixels that occurs due to the partial pixel due to the use of a 1 dimensional array
	int partial_size = (image->width - (cell_size * (mos_width - 1)));
	int overflow = cell_size - partial_size;

	int mosaic_row = 0;
	// For each 'tile' of the mosaic image
	int i = 0;
#pragma omp parallel for shared(image, mosaic) private(i,mosaic_row)
	for (i = 0; i < mos_width * mos_height; i++) {
		// Get the row that the tile occupies, as it is integer division the value will be truncated to the 
		// lower value.
		mosaic_row = (i / mos_width);

		unsigned long x = 0;
		unsigned long y = 0;
		unsigned long z = 0;
		short av_r = 0;
		short av_g = 0;
		short av_b = 0;

		int index = 0;
		int pixel_row = 0;

		// A counter is needed incase of a partial tile to ensure that the colour value is accurate
		int counter = 0;

		// For 2 times, 0 = work out rgb totals, 1 = assign averages to pixels 
		int k = 0;
		//#pragma omp parallel for private(k)
		for (k = 0; k < 2; k++) {
			// Assign the averages
			if (k == 1) {
				av_r = (short)(x / counter);
				av_g = (short)(y / counter);
				av_b = (short)(z / counter);
			}

			pixel_row = 0;

			// If the mosaic tile is partial
			if (i != 0 && ((partial_width && ((mosaic_row*mos_width) + (mos_width - 1)) == i))) {
				int j = 0;
				// For c*partial pixels in the partial mosaic tile
#pragma omp parallel for private(j) reduction(+: x,y,z)
				for (j = 0; j < cell_size*partial_size; j++) {
					// If j has reached the end of the pixel width, go to the next row
					if (j != 0 && j % partial_size == 0) pixel_row += 1;

					// Work out the current index and account for the overflow if a partial mosaic image
					index = (j - (pixel_row * c)) + (i * c) + image->width * (pixel_row + (mosaic_row * (c - 1)));
					index += (overflow * pixel_row) - (overflow*mosaic_row);
					
					// If j has reached the pixel height or the index has reached the image height, then we can stop for this pixel
					if (index >= image->height * image->width || (j >= (cell_size*partial_size))) break;

					// Calculate the totals and increase the counter for the pixel
					if (k == 0) {
						counter += 1;
						x += image->rgb[index].x;
						y += image->rgb[index].y;
						z += image->rgb[index].z;
					}
					// Assign the average mosaic colour to the pixel
					else if (k == 1) {
						mosaic->rgb[index].x = (unsigned char)av_r;
						mosaic->rgb[index].y = (unsigned char)av_g;
						mosaic->rgb[index].z = (unsigned char)av_b;
					}
				}
			}
			// Else if it is a full tile
			else {
				// For c*c pixels in the mosaic tile
				int j = 0;
#pragma omp parallel for private(j) reduction(+: x,y,z)
				for (j = 0; j < cell_pixels; j++) {
					// If j has reached the end of the 'pixel', increase the pixel row by 1
					if (j != 0 && (j % cell_size) == 0) pixel_row += 1;

					// Work out the current index and account for the overflow if a partial mosaic image
					index = (j - (pixel_row * c)) + (i * c) + image->width * (pixel_row + (mosaic_row * (c - 1)));
					if (mosaic_row > 0 && partial_width) index -= overflow * mosaic_row;

					// If j has reached the pixel height or the index has reached the image height, then we can stop for this pixel
					if (index >= image->height * image->width || (j >= cell_pixels)) break;

					// Calculate the totals and increase the counter for the pixel
					if (k == 0) {
						counter += 1;
						x += image->rgb[index].x;
						y += image->rgb[index].y;
						z += image->rgb[index].z;
					}
					// Assign the average mosaic colour to the pixel
					else if (k == 1) {
						mosaic->rgb[index].x = (unsigned char)av_r;
						mosaic->rgb[index].y = (unsigned char)av_g;
						mosaic->rgb[index].z = (unsigned char)av_b;
					}
				}
			}
		}
	}
	return mosaic;
}

// My final implementation looked into scheduling where the main loop has dynamic scheduling and the inner
// tile loops have a chunk size of C.
ppm *do_mosaic_OPENMP_III(ppm *image) {
	ppm *mosaic;

	// Allocate memory for the mosaic image
	mosaic = (ppm *)malloc(sizeof(ppm));

	// As c is a global variable, make it local so it is able to be cached.
	int cell_size = c;
	int cell_pixels = cell_size * cell_size;

	int mos_width = 0;
	int mos_height = 0;

	// Check if the image will have partial mosaic cells
	bool partial_width = false;
	bool partial_height = false;
	if (image->width % cell_size != 0) partial_width = true;
	if (image->height % cell_size != 0) partial_height = true;

	// Get the number of 'tiles' in the mosaic image
	mosaic->width = image->width;
	mosaic->height = image->height;
	mos_width = (int)ceil((double)image->width / cell_size);
	mos_height = (int)ceil((double)image->height / cell_size);

	// Allocate memory to the rbg struct of the mosaic image
	mosaic->rgb = (uchar4*)malloc(mosaic->width * mosaic->height * sizeof(uchar4*));

	// The overflow of pixels that occurs due to the partial pixel due to the use of a 1 dimensional array
	int partial_size = (image->width - (cell_size * (mos_width - 1)));
	int overflow = cell_size - partial_size;

	int mosaic_row = 0;
	// For each 'tile' of the mosaic image, using a dynamic scheduling approach
	int i = 0;
#pragma omp parallel for shared(image, mosaic) private(i,mosaic_row) schedule(dynamic)
	for (i = 0; i < mos_width * mos_height; i++) {
		// Get the row that the tile occupies, as it is integer division the value will be truncated to the 
		// lower value.
		mosaic_row = (i / mos_width);

		unsigned long x = 0;
		unsigned long y = 0;
		unsigned long z = 0;
		short av_r = 0;
		short av_g = 0;
		short av_b = 0;

		int index = 0;
		int pixel_row = 0;

		// A counter is needed incase of a partial tile to ensure that the colour value is accurate
		int counter = 0;

		// For 2 times, 0 = work out rgb totals, 1 = assign averages to pixels 
		int k = 0;
		for (k = 0; k < 2; k++) {
			// Assign the averages
			if (k == 1) {
				av_r = (short)(x / counter);
				av_g = (short)(y / counter);
				av_b = (short)(z / counter);
			}

			pixel_row = 0;

			// If the mosaic tile is partial
			if (i != 0 && ((partial_width && ((mosaic_row*mos_width) + (mos_width - 1)) == i))) {
				int j = 0;
				// For c*partial pixels in the partial mosaic tile
#pragma omp parallel for private(j,index) reduction(+: x,y,z) 
				for (j = 0; j < cell_size*partial_size; j++) {
					// If j has reached the end of the pixel width, go to the next row
					if (j != 0 && j % partial_size == 0) pixel_row += 1;

					// Work out the current index and account for the overflow if a partial mosaic image
					index = (j - (pixel_row * c)) + (i * c) + image->width * (pixel_row + (mosaic_row * (c - 1)));
					index += (overflow * pixel_row) - (overflow*mosaic_row);

					// If j has reached the pixel height or the index has reached the image height, then we can stop for this pixel
					if (index >= image->height * image->width || (j >= (cell_size*partial_size))) break;

					// Calculate the totals and increase the counter for the pixel
					if (k == 0) {
						counter += 1;
						x += image->rgb[index].x;
						y += image->rgb[index].y;
						z += image->rgb[index].z;
					}
					// Assign the average mosaic colour to the pixel
					else if (k == 1) {
						mosaic->rgb[index].x = (unsigned char)av_r;
						mosaic->rgb[index].y = (unsigned char)av_g;
						mosaic->rgb[index].z = (unsigned char)av_b;
					}
				}
			}
			// Else if it is a full tile
			else {
				// For c*c pixels in the mosaic tile
				int j = 0;
#pragma omp parallel for private(j,index) reduction(+: x,y,z) schedule(static,cell_size)
				for (j = 0; j < cell_pixels; j++) {
					// If j has reached the end of the 'pixel', increase the pixel row by 1
					if (j != 0 && (j % cell_size) == 0) pixel_row += 1;

					// Work out the current index and account for the overflow if a partial mosaic image
					index = (j - (pixel_row * c)) + (i * c) + image->width * (pixel_row + (mosaic_row * (c - 1)));
					if (mosaic_row > 0 && partial_width) index -= overflow * mosaic_row;

					// If j has reached the pixel height or the index has reached the image height, then we can stop for this pixel
					if (index >= image->height * image->width || (j >= cell_pixels)) break;

					// Calculate the totals and increase the counter for the pixel
					if (k == 0) {
						counter += 1;
						x += image->rgb[index].x;
						y += image->rgb[index].y;
						z += image->rgb[index].z;
					}
					// Assign the average mosaic colour to the pixel
					else if (k == 1) {
						mosaic->rgb[index].x = (unsigned char)av_r;
						mosaic->rgb[index].y = (unsigned char)av_g;
						mosaic->rgb[index].z = (unsigned char)av_b;
					}
				}
			}
		}
	}
	return mosaic;
}