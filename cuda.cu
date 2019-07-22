#include "kernels.cuh"
#include "mosaic.cuh"

extern unsigned int c;
unsigned int block_size = 0;

/* 
 * When the average function has finished, call this function to reduce the sum
 * to get the total average on the CPU.
 *
 * h_average_colour - The vector to reduce the block level results from.
 */
void CUDART_CB CallbackReduce(void *h_average_colour) {
	uchar4 *h_data = (uchar4 *)h_average_colour;
	unsigned int n = block_size;

	long3 average = make_long3(0, 0, 0);
	// Reduce the block level results on CPU
	for (unsigned int i = 0; i < n; i++) {
		// Load a single value and add to the total average.
		uchar4 av = h_data[i];
		average.x += (long)(av.x);
		average.y += (long)(av.y);
		average.z += (long)(av.z);
	}

	// Divide and round the totals to the closest integer to give the average.
	average.x = div_round(average.x, n);
	average.y = div_round(average.y, n);
	average.z = div_round(average.z, n);

	// Output the average colour value for the image
	fprintf(stderr, "CUDA Average image colour red = %d, green = %d, blue = %d \n", average.x, average.y, average.z);
}

/*
 * The main CUDA function which calls the kernels to do the mosaic and average
 * functionality.
 *
 * image - The ppm struct of the image we are performing on.
 * iteration - Which iteration of the code to use.
 *
 * returns - The new 'mosaic' ppm image.
 */
__host__ ppm *perform_CUDA(ppm *image, int iteration) {
	unsigned int image_size = (image->width * image->height);

	// Device declarations
	uchar4 *d_rgb, *d_rgb_output, *d_average_colour;
	uint4 *d_average;

	// Create the steams that will be used in iterations 2,3
	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	// Allocate memory on the CPU for the image
	ppm *h_image = (ppm *)malloc(sizeof(ppm));
	uchar4 *h_rgb = (uchar4 *)malloc(image_size * sizeof(uchar4));
	uchar4 *h_average_colour = (uchar4 *)malloc(image_size * sizeof(uchar4));
	uint4 *h_average = (uint4 *)malloc(sizeof(uint4));
	
	// Cuda timing creation
	cudaEvent_t start, stop, execution, execution_stop;
	float ms, ems;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&execution);
	cudaEventCreate(&execution_stop);

	//  Start timing
	cudaEventRecord(start, 0);

	// Cuda layout for mosaic
	// As max threads per block can be 1024 (32,32), if c is higher then default to this value.
	int tpb = fmin(c, 32);
	dim3 mosBlocksPerGrid((int)ceil((double)image->width / c), (int)ceil((double)image->height / c));
	dim3 mosThreadsPerBlock(tpb, tpb);

	// Work out the amount of work needed if c is greater than 32.
	int work = c <= 32 ? 1 : (c * c) / (32 * 32);

	block_size = mosBlocksPerGrid.x * mosBlocksPerGrid.y;

	// Allocate memory on the GPU for the image
	cudaMalloc((void **)&d_rgb_output, image_size * sizeof(uchar4));
	cudaMalloc((void **)&d_rgb, image_size * sizeof(uchar4));
	cudaMalloc((void **)&d_average_colour, block_size * sizeof(uchar4));
	cudaMalloc((void **)&d_average, sizeof(uint4));
	checkCUDAError("CUDA malloc");

	// Copy the rgb values of the image to device memory
	cudaMemcpyAsync(d_rgb, image->rgb, image_size * sizeof(uchar4), cudaMemcpyHostToDevice, stream1);
	checkCUDAError("CUDA memcpy to device");

	// Perform the mosaic filter
	unsigned int sm_size = sizeof(uint4) * mosThreadsPerBlock.x * mosThreadsPerBlock.y;

	if (iteration == 1) {
		// 1D Texture Bind
		cudaBindTexture(0, ppmTexture1D, d_rgb, image_size * sizeof(uchar4));
		checkCUDAError("ppmTexture1D bind");

		// Block	
		average_CUDA_block << <mosBlocksPerGrid, mosThreadsPerBlock, sm_size >> > (d_rgb, d_average_colour, image->width, work);
		checkCUDAError("mosaic block it1");
		mosaic_CUDA_tile << <mosBlocksPerGrid, mosThreadsPerBlock >> > (d_average_colour, d_rgb_output, image->width, work);
		checkCUDAError("mosaic tile it1");
	}
	else if (iteration == 2) {
		// Warp and reduce
		cudaEventRecord(execution, 0);
		average_CUDA_warp << <mosBlocksPerGrid, mosThreadsPerBlock >> > (d_rgb, d_average_colour, image->width, work, tpb);
		checkCUDAError("mosaic warp it2");
		cudaDeviceSynchronize();
		dim3 avThreadsPerBlock(1, 1);
		reduce << < mosBlocksPerGrid, avThreadsPerBlock, 0, stream1 >> > (d_average_colour, d_average);
		mosaic_CUDA_tile << <mosBlocksPerGrid, mosThreadsPerBlock, 0, stream2 >> > (d_average_colour, d_rgb_output, image->width, work);
		checkCUDAError("mosaic tile it2");
	}
	else if (iteration == 3) {
		// Block and reduce
		cudaEventRecord(execution, 0);
		average_CUDA_block_final << <mosBlocksPerGrid, mosThreadsPerBlock, sm_size >> > (d_rgb, d_average_colour, image->width, work);
		checkCUDAError("mosaic block it3");
		cudaDeviceSynchronize();
		mosaic_CUDA_tile_final << <mosBlocksPerGrid, mosThreadsPerBlock >> > (d_average_colour, d_rgb_output, image->width, work, d_average);
		checkCUDAError("mosaic tile it3");
	}

	// Sync
	cudaDeviceSynchronize();

	if (iteration != 1) {
		cudaEventRecord(execution_stop, 0);
		cudaEventSynchronize(execution_stop);
		cudaEventElapsedTime(&ems, execution, execution_stop);
		printf("CUDA mode execution time took %f ms\n", ems);
	}

	// Copy the image back from the GPU
	cudaMemcpyAsync(h_rgb, d_rgb_output, image_size * sizeof(uchar4), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(h_average_colour, d_average_colour, block_size * sizeof(uchar4), cudaMemcpyDeviceToHost, stream2);
	cudaMemcpyAsync(h_average, d_average, sizeof(ulong4), cudaMemcpyDeviceToHost, stream3);
	checkCUDAError("CUDA memcpy from device");

	// Reduce the final values on the CPU
	if(iteration == 1)
		cudaLaunchHostFunc(stream1, CallbackReduce, (void *)h_average_colour);
	else
		fprintf(stderr, "CUDA Average image colour red = %d, green = %d, blue = %d \n", div_round(h_average[0].x, block_size), div_round(h_average[0].y, block_size), div_round(h_average[0].z, block_size));
		
	// End Timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);

	//output timings
	
	printf("CUDA mode total time took %f ms\n", ms);

	// Free device memory 
	cudaFree(d_rgb);
	cudaFree(d_rgb_output);
	cudaFree(d_average_colour);
	cudaFree(d_average);
	cudaUnbindTexture(ppmTexture1D); 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	checkCUDAError("Cuda free and destroy");
	
	// Put data back into a host ppm struct
	h_image->height = image->height;
	h_image->width = image->width;
	h_image->rgb = h_rgb;

	// Free host memory
	free(h_average_colour);
	free(h_average);

	return h_image;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// A simple function that divides two ints and rounds to the nearest int.
unsigned int div_round(unsigned int a, unsigned int b) {
	return (a + (b / 2)) / b;
}