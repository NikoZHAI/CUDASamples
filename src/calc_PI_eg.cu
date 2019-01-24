/* !!! This is an exemple of USC !!! */

// Using CUDA device to calculate pi
#include <stdio.h>
#include <iostream>
#include <string>
#include <cuda.h>
#include <chrono>


#define NBIN  1000000000 // Number of bins
#define NUM_BLOCK  128   // Number of thread blocks
#define NUM_THREAD 128   // Number of threads per block
#define AMPLIFIER  32768 // maybe working?

int tid;
double pi = 0;


static void HandleError( cudaError_t err,
						 const char *file,
						 int line )
{
	if (err != cudaSuccess) {
	printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
	file, line );
	exit( EXIT_FAILURE );
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


// Kernel that executes on the CUDA device
__global__ void cal_pi(double *sum, int nbin, float step, int nthreads, int nblocks) {
	int i;
	double x;

	// Sequential thread index across the blocks
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	
	for (i=idx; i < nbin; i+=nthreads*nblocks) {
		x = (i+.5f) / nbin;
		sum[idx] += 4.f/(1.f + x*x);
	}
}

// Main routine that executes on the host
int main(int argc, char* argv[]) {

	int n_steps =  1 << std::stoi(argv[1]);
	printf("N = %i\n", n_steps);

	dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
	double *sumHost, *sumDev;  // Pointer to host & device arrays

	// float step = 1.0/NBIN;  // Step size
	double step = 1.0/n_steps;  // Step size
	size_t size = NUM_BLOCK*NUM_THREAD*sizeof(double);  //Array memory size
	
	sumHost = (double *)malloc(size);  //  Allocate array on host
	HANDLE_ERROR(cudaMalloc((void **) &sumDev, size));  // Allocate array on device
	
	auto t1 = std::chrono::system_clock::now();
	// Initialize array in device to 0
	cudaMemset(sumDev, 0., size);
	
	// Do calculation on device
	cal_pi <<<dimGrid, dimBlock>>> (sumDev, n_steps, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel
	cudaDeviceSynchronize();

	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
		pi += sumHost[tid];
	
	pi /= n_steps;

	auto t2 = std::chrono::system_clock::now();

	// Print results
	printf("PI = %1.18f\n",pi);

	std::cout << "Computation time: "
			  << std::chrono::duration <double, std::milli> (t2 - t1).count()
			  << " ms.\n";

	// Cleanup
	free(sumHost); 
	cudaFree(sumDev);

	return 0;
}