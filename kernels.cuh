#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.hpp"

texture<uchar4, cudaTextureType1D, cudaReadModeElementType> ppmTexture1D;

/*
 * Given a thread, work out which block it is a part of and assign its colour to the new rgb.
 *
 * d_sum - The sum of values for each block.
 * d_rgb_output - The rgb values of the new image.
 * image_width - The width of the image we are making.
 * work - The amount of pixels each thread has to get the value of.
 *
 */
__global__ void mosaic_CUDA_tile(uchar4 *d_sum, uchar4 *d_rgb_output, int image_width, int work) {
	// The x and y values for the thread to get from the device rgb. 
	// If working with a c value larger than 32, also work out the offset needed.
	int x = work == 1 ? (threadIdx.x + blockIdx.x * blockDim.x) : (threadIdx.x + blockIdx.x * blockDim.x) + ((int)sqrt((double)work) - 1) * blockIdx.x * blockDim.x;
	int y = work == 1 ? (threadIdx.y + blockIdx.y * blockDim.y) : (threadIdx.y + blockIdx.y * blockDim.y) + ((int)sqrt((double)work) - 1) * blockIdx.y * blockDim.y;

	// Get the index of the block to get the colour from.
	int block_index = (blockIdx.y * gridDim.x + blockIdx.x);

	// Get the average pixel colour for this block (tile)
	uchar4 pixel = d_sum[block_index];

	// Get the index of the pixels for the current thread. 
	for (int i = 0; i < sqrt((double)work); i++) {
		for (int j = 0; j < sqrt((double)work); j++) {
			int x_offset = (x + j * blockDim.x);
			int y_offset = (y + i * blockDim.y);

			// Check bound conditions
			if (x_offset < 0 || x_offset >= image_width)
				continue;
			if (y_offset < 0 || y_offset >= image_width)
				continue;

			int thread_index = y_offset * image_width + x_offset;
			// Apply this colour to our rgb
			d_rgb_output[thread_index] = pixel;
		}
	}
}

__global__ void mosaic_CUDA_tile_final(uchar4 *d_sum, uchar4 *d_rgb_output, int image_width, int work, uint4 *average) {
	int load = (int)sqrt((double)work);

	// The index of the thread that we are working with to put into shared memory
	int tid = threadIdx.y * blockDim.y + threadIdx.x;

	// The x and y values for the thread to get from the device rgb. 
	// If working with a c value larger than 32, also work out the offset needed.
	int x = work == 1 ? (threadIdx.x + blockIdx.x * blockDim.x) : (threadIdx.x + blockIdx.x * blockDim.x) + (load - 1) * blockIdx.x * blockDim.x;
	int y = work == 1 ? (threadIdx.y + blockIdx.y * blockDim.y) : (threadIdx.y + blockIdx.y * blockDim.y) + (load - 1) * blockIdx.y * blockDim.y;

	// Get the index of the block to get the colour from.
	int block_index = (blockIdx.y * gridDim.x + blockIdx.x);

	// Get the average pixel colour for this block (tile)
	uchar4 pixel = d_sum[block_index];

	// Get the index of the pixels for the current thread. 
	for (int i = 0; i < load; i++) {
		for (int j = 0; j < load; j++) {
			int x_offset = (x + j * blockDim.x);
			int y_offset = (y + i * blockDim.y);

			// Check bound conditions
			if (x_offset < 0 || x_offset >= image_width)
				continue;
			if (y_offset < 0 || y_offset >= image_width)
				continue;

			int thread_index = y_offset * image_width + x_offset;
			// Apply this colour to our rgb
			d_rgb_output[thread_index] = pixel;
	
		}
	}

	if (tid == 0) {
		atomicAdd(&average[0].x, (int)pixel.x);
		atomicAdd(&average[0].y, (int)pixel.y);
		atomicAdd(&average[0].z, (int)pixel.z);
	}
}

/*
 * Block level reduction definition for working out the average, is
 * conflict free due to strided shared memory access.
 *
 * d_rgb - The device rgb containing all pixel values.
 * d_sum - The sum of values for each block.
 * image_width - The width of the image we are making.
 * work - The amount of pixels each thread has to calculate the sum of.
 *
 */
__global__ void average_CUDA_block(uchar4 *d_rgb, uchar4 *d_sum, int image_width, int work) {
	extern __shared__ uint4 s_tile[];

	// The index of the thread that we are working with to put into shared memory
	int tid = threadIdx.y * blockDim.y + threadIdx.x;

	// The x and y values for the thread to get from the device rgb. 
	// If working with a c value larger than 32, also work out the offset needed.
	int x = work == 1 ? (threadIdx.x + blockIdx.x * blockDim.x) : (threadIdx.x + blockIdx.x * blockDim.x) + ((int)sqrt((double)work) - 1) * blockIdx.x * blockDim.x;
	int y = work == 1 ? (threadIdx.y + blockIdx.y * blockDim.y) : (threadIdx.y + blockIdx.y * blockDim.y) + ((int)sqrt((double)work) - 1) * blockIdx.y * blockDim.y;

	// Get the index of the pixels for the current thread.
	uint4 thread_sum = make_uint4(0, 0, 0, 0);
	for (int i = 0; i < sqrt((double)work); i++) {
		for (int j = 0; j < sqrt((double)work); j++) {
			int x_offset = (x + j * blockDim.x);
			int y_offset = (y + i * blockDim.y);

			// Check bound conditions
			if (x_offset < 0 || x_offset >= image_width)
				continue;
			if (y_offset < 0 || y_offset >= image_width)
				continue;

			int thread_index = y_offset * image_width + x_offset;
				
			uchar4 pixel = tex1Dfetch(ppmTexture1D, thread_index);
			thread_sum.x += pixel.x;
			thread_sum.y += pixel.y;
			thread_sum.z += pixel.z;		
		}
	}

	// Load single into shared memory
	s_tile[tid].x = thread_sum.x;
	s_tile[tid].y = thread_sum.y;
	s_tile[tid].z = thread_sum.z;
	__syncthreads();

	int block_size = (blockDim.x * blockDim.y);
	// Conflict free reduction due to strided shared memory
	for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			s_tile[tid].x += s_tile[tid + stride].x;
			s_tile[tid].y += s_tile[tid + stride].y;
			s_tile[tid].z += s_tile[tid + stride].z;
		}
		__syncthreads();
	}

	// Write result
	if (tid == 0) {
		// The actual size of the tile if larger than 32
		int cell_size = block_size * work;

		// The index of the block that each thread can refer back to to get the colour value
		int block_index = (blockIdx.y * gridDim.x + blockIdx.x);
		d_sum[block_index].x = s_tile[0].x / cell_size;
		d_sum[block_index].y = s_tile[0].y / cell_size;
		d_sum[block_index].z = s_tile[0].z / cell_size;
	}
}

__global__ void average_CUDA_block_final(uchar4 *d_rgb, uchar4 *d_sum, int image_width, int work) {
	extern __shared__ uint4 s_tile[];

	// The index of the thread 
	int tid = threadIdx.y * blockDim.y + threadIdx.x;

	int load = (int)sqrt((double)work);

	// The x and y values for the thread to get from the device rgb. 
	// If working with a c value larger than 32, also work out the offset needed.
	int x = work == 1 ? (threadIdx.x + blockIdx.x * blockDim.x) : (threadIdx.x + blockIdx.x * blockDim.x) + (load - 1) * blockIdx.x * blockDim.x;
	int y = work == 1 ? (threadIdx.y + blockIdx.y * blockDim.y) : (threadIdx.y + blockIdx.y * blockDim.y) + (load - 1) * blockIdx.y * blockDim.y;

	int block_size = (blockDim.x * blockDim.y);

	// Get the index of the pixels for the current thread.
	uint4 thread_sum = make_uint4(0, 0, 0, 0);
	for (int i = 0; i < load; i++) {
		for (int j = 0; j < load; j++) {
			int x_offset = (x + j * blockDim.x);
			int y_offset = (y + i * blockDim.y);

			// Check bound conditions
			if (x_offset < 0 || x_offset >= image_width)
				continue;
			if (y_offset < 0 || y_offset >= image_width)
				continue;

			int thread_index = y_offset * image_width + x_offset;

			thread_sum.x += d_rgb[thread_index].x;
			thread_sum.y += d_rgb[thread_index].y;
			thread_sum.z += d_rgb[thread_index].z;
		}
	}

	// Load single into shared memory
	s_tile[tid].x = thread_sum.x;
	s_tile[tid].y = thread_sum.y;
	s_tile[tid].z = thread_sum.z;
	__syncthreads();

	// Conflict free reduction due to strided shared memory
	for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			s_tile[tid].x += s_tile[tid + stride].x;
			s_tile[tid].y += s_tile[tid + stride].y;
			s_tile[tid].z += s_tile[tid + stride].z;
		}
		__syncthreads();
	}

	// Write result
	if (tid == 0) {
		// The actual size of the tile if larger than 32
		int cell_size = block_size * work;

		// The index of the block that each thread can refer back to to get the colour value
		int block_index = (blockIdx.y * gridDim.x + blockIdx.x);
		d_sum[block_index].x = s_tile[0].x / cell_size;
		d_sum[block_index].y = s_tile[0].y / cell_size;
		d_sum[block_index].z = s_tile[0].z / cell_size;
	}
}

/* 
 * Warp shuffle definition for working out average, does not need to use shared
 * memory and allows for implicit synchronisation.
 *
 * d_rgb - The device rgb containing all pixel values.
 * d_sum - The sum of values for each block.
 * image_width - The width of the image we are making.
 * work - The amount of pixels each thread has to calculate the sum of.
 *
 */
__global__ void average_CUDA_warp(uchar4 *d_rgb, uchar4 *d_sum, int image_width, int work, int c) {
	int load = (int)sqrt((double)work);

	// The x and y values for the thread to get from the device rgb. 
	// If working with a c value larger than 32, also work out the offset needed.
	int x = work == 1 ? (threadIdx.x + blockIdx.x * blockDim.x) : (threadIdx.x + blockIdx.x * blockDim.x) + (load - 1) * blockIdx.x * blockDim.x;
	int y = work == 1 ? (threadIdx.y + blockIdx.y * blockDim.y) : (threadIdx.y + blockIdx.y * blockDim.y) + (load - 1) * blockIdx.y * blockDim.y;

	int block_size = (blockDim.x * blockDim.y);

	// Get the index of the pixels for the current thread.
	uint4 thread_sum = make_uint4(0, 0, 0, 0);
	for (int i = 0; i < load; i++) {
		for (int j = 0; j < load; j++) {
			int x_offset = (x + j * blockDim.x);
			int y_offset = (y + i * blockDim.y);

			// Check bound conditions
			if (x_offset < 0 || x_offset >= image_width)
				continue;
			if (y_offset < 0 || y_offset >= image_width)
				continue;

			int thread_index = y_offset * image_width + x_offset;

			thread_sum.x += d_rgb[thread_index].x;
			thread_sum.y += d_rgb[thread_index].y;
			thread_sum.z += d_rgb[thread_index].z;		
		}
	}

	// Get index
	int block_index = (blockIdx.y * gridDim.x + blockIdx.x);

	// Shuffles to the nth left neighbour whilst increasing the local sum for
	// this block.
	for (int offset = block_size / 2; offset > 0; offset >>= 1) {
		thread_sum.x += __shfl_down_sync(0xffffffff,thread_sum.x, offset);
		thread_sum.y += __shfl_down_sync(0xffffffff,thread_sum.y, offset);
		thread_sum.z += __shfl_down_sync(0xffffffff,thread_sum.z, offset);
	}

	// Write the result of the local sum
	if (threadIdx.x % c == 0 && threadIdx.y % c == 0){
		// The actual size of the tile if larger than 32
		int cell_size = block_size * work;
		d_sum[block_index].x = (thread_sum.x / cell_size);
		d_sum[block_index].y = (thread_sum.y / cell_size);
		d_sum[block_index].z = (thread_sum.z / cell_size);
	}
}

/*
 * Reduce the average values on the GPU, atomic add the x,y,z values to ensure
 * that the average safely increments correctly.
 *
 * d_sum - The sum of values for each block.
 * average - The average value colour value of the image (to be divided)
 *
 */
__global__ void reduce(uchar4 *d_sum, uint4 *average) {
	int block_index = (blockIdx.y * gridDim.x + blockIdx.x);
	uchar4 pixel = d_sum[block_index];
	atomicAdd(&average[0].x, (int)pixel.x);
	atomicAdd(&average[0].y, (int)pixel.y);
	atomicAdd(&average[0].z, (int)pixel.z);
}