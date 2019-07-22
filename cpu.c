#include "mosaic.cuh"

extern unsigned int c;

// CPU average_colour - Loop through every pixel of the image to get the total
// rgb value. Then divide these by the number of pixels in the image to get the
// average.
void average_colour_value(ppm *image, unsigned long currentPixel[]) {
	int num_pixels = image->width * image->height;

	for (int i = 0; i < num_pixels; i++) {
		currentPixel[0] += image->rgb[i].x;
		currentPixel[1] += image->rgb[i].y;
		currentPixel[2] += image->rgb[i].z;
	}

	currentPixel[0] /= num_pixels;
	currentPixel[1] /= num_pixels;
	currentPixel[2] /= num_pixels;

}

// CPU do_mosaic - The main mosaic loop. Loop through each mosaic tile and calculate the
// average value for it. Then loop through the pixels again but this time assign them
// to the new mosaic image pixels.
ppm *do_mosaic(ppm *image) {
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
	if (image->width % c != 0) partial_width = true;
	if (image->height % c != 0) partial_height = true;

	// Get the number of 'tiles' in the mosaic image
	mosaic->width = image->width;
	mosaic->height = image->height;
	mos_width = (int)ceil((double)image->width / cell_size);
	mos_height = (int)ceil((double)image->height / cell_size);

	// Allocate memory to the rbg struct of the mosaic image
	mosaic->rgb = (uchar4*)malloc(mosaic->width * mosaic->height * sizeof(uchar4*));

	// The overflow of pixels that occurs due to the partial pixel due to the use of a 1 dimensional array
	int partial_size = (image->width - (cell_size * (mos_width - 1)));
	int overflow = c - partial_size;

	int mosaic_row = 0;
	// For each 'tile' of the mosaic image
	for (int i = 0; i < mos_width * mos_height; i++) {
		// If i has reached the end of the line, increase the row by 1
		if (i != 0 && (i % mos_width) == 0) mosaic_row += 1;

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
		for (int k = 0; k < 2; k++) {

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

				// Work out the current index and account for the overflow if a partial mosaic image
				index = (j - (pixel_row * cell_size)) + (i * cell_size) + image->width * (pixel_row + (mosaic_row * (cell_size - 1)));
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