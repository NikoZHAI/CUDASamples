#include <stdio.h>
#include <iostream>
#include <string>
#include <cuda.h>
#include <chrono>


#define NUM_BLOCK  128   // Number of thread blocks
#define NUM_THREAD 128   // Number of threads per block

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
		x = (i+.5) / nbin;
		sum[idx] += 4./(1. + x*x);
	}
}

// Main routine that executes on the host
int main(int argc, char* argv[]) {

	/* Settings */
	int n_steps  =  1 << std::stoi(argv[1]);
	int nblocks;
	int nthreads;
	int min_nblocks;

	cudaOccupancyMaxPotentialBlockSize(&min_nblocks, &nthreads, 
									   cal_pi, 0, 0);
	nblocks = (n_steps + nthreads - 1) / nthreads;

	dim3 dimGrid(nblocks,1,1);  	// Grid dimensions
	dim3 dimBlock(nthreads,1,1);  // Block dimensions
	double *sumHost, *sumDev;  		// Pointer to host & device arrays

	double step = 1.0/n_steps;  						// Step size
	size_t size = nblocks*nthreads*sizeof(double);  // Size of the device array

	printf("         N          = %11i\n", n_steps);
	printf("  N thread blocks   = %11i\n", nblocks);
	printf("N threads per block = %11i\n", nblocks);

	sumHost = (double *)malloc(size);  //  Allocate array on host
	HANDLE_ERROR(cudaMalloc((void **) &sumDev, size));  // Allocate array on device
	
	auto t1 = std::chrono::system_clock::now();
	// Initialization
	HANDLE_ERROR(cudaMemset(sumDev, 0., size));
	
	/* Invoke the CUDA kernel */
	cal_pi <<<dimGrid, dimBlock>>> (sumDev, n_steps, step, nthreads, nblocks); // call CUDA kernel
	cudaDeviceSynchronize();  // Wait for calculations finished

	/* Reduction */
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<nblocks*nthreads; ++tid) {
		pi += sumHost[tid];
	}
	pi /= n_steps;
	auto t2 = std::chrono::system_clock::now();

	/* Results */
	printf("\nPI = %1.18f\n",pi);
	std::cout << "Computation time: "
			  << std::chrono::duration <double, std::milli> (t2 - t1).count()
			  << " ms.\n";

	// Free memory
	free(sumHost); 
	cudaFree(sumDev);

	return 0;
}