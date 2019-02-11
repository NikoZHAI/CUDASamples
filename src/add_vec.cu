/* Good Example of handling main-stack overflow */

#include <iostream>
#include <cmath>
#include <cuda.h>
#include <string>
#include <chrono>


#define PI 3.1415927


static void HandleError(cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
    file, line );
    exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void addVector(int vec_size, float* v1, float* v2, float* vec_out) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    while(tid < vec_size) {
        vec_out[tid] = v1[tid] + v2[tid];
        tid += blockDim.x * gridDim.x;
    }
}


int main(int argc, char* argv[]) {

    size_t vec_len = 1 << std::stoi(argv[1]);
    size_t size    = vec_len * sizeof(float);

    float *v1    = (float *)malloc(size);
    float *v2    = (float *)malloc(size);
    float *v_out = (float *)malloc(size);
    float *dev_v1, *dev_v2, *dev_v_out;

    // CUDA Morphology
    int nthreads = std::stoi(argv[2]);
    int nblocks  = std::stoi(argv[3]);
    // int nblocks  = (vec_len+nthreads-1) / nthreads;
    std::cout << "Number of threads per block:  " << std::to_string(nthreads) << '\n'
              << "Number of blocks in the grid: " << std::to_string(nblocks)  << '\n'
              << "Total threads: " << std::to_string(nthreads*nblocks) << "\n"
              << "Vector length: "  << std::to_string(vec_len) << std::endl;
    dim3 nBlocks(nblocks, 1, 1);
    dim3 nThreads(nthreads, 1, 1);
    
    // Initiate values
    for(size_t i=0; i<vec_len; ++i) {
        v1[i] = std::sin(i*PI*1E-2);
        v2[i] = std::cos(i*PI*1E-2);
    }
    
    auto t1 = std::chrono::system_clock::now();

    HANDLE_ERROR( cudaMalloc((void**) &dev_v1, size) );
    HANDLE_ERROR( cudaMalloc((void**) &dev_v2, size) );
    HANDLE_ERROR( cudaMalloc((void**) &dev_v_out, size) );

    HANDLE_ERROR( cudaMemcpy(dev_v1, v1, size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_v2, v2, size, cudaMemcpyHostToDevice) );

    addVector<<< nBlocks, nThreads >>>(vec_len, dev_v1, dev_v2, dev_v_out);
    cudaDeviceSynchronize();
    
    HANDLE_ERROR( cudaMemcpy(v_out, dev_v_out, size, cudaMemcpyDeviceToHost) );

    auto t2 = std::chrono::system_clock::now();

    // Check results
    for(size_t i=0; i<vec_len; ++i) {
        if(v1[i] + v2[i] != v_out[i]){
            std::string err_message = "value dismatch at index " + std::to_string(i) + ".\n";
            printf("v1[%i] = %.18f \nv2[%i] = %.18f\nv_out[%i] = %.18f.\n", 
                   i, v1[i], i, v2[i], i, v_out[i]);
            throw std::runtime_error(err_message);
        }
    }

    printf("Work done, time consummed: %.5fms.\n",
            std::chrono::duration <double, std::milli> (t2 - t1).count());
    
    cudaFree(dev_v1);
    cudaFree(dev_v2);
    cudaFree(dev_v_out);
    free(v1);
    free(v2);
    free(v_out);

    return 0;
}

