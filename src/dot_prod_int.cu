#include <iostream>
#include <string>
#include <cmath>
#include <chrono>
#include <cuda.h>

const size_t nThreadsPerBlock = 128;


static void HandleError(cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
            file, line );
    exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


// Direct use of global memory between threads may be very wrong?
__global__ void dotProd(int length, int *u, int *v, int *out) {
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned tid_const = threadIdx.x + blockDim.x * blockIdx.x;
    int temp = 0;

    while (tid < length) {
        temp += u[tid] * v[tid];
        tid += blockDim.x * gridDim.x;
    }
    out[tid_const] = temp;
}


__global__ void dotProdWithSharedMem(int length, int *u, int *v, int *out) {
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned cid = threadIdx.x;
    __shared__ int cache[nThreadsPerBlock];
    int temp = 0;

    while (tid < length) {
        temp += u[tid] * v[tid];
        tid  += blockDim.x * gridDim.x;
    }

    cache[cid] = temp;
    __syncthreads();
    
    int i = blockDim.x/2;
    while (i != 0) {
        if (cid < i) {
            cache[cid] += cache[cid + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cid == 0) {
        out[blockIdx.x] = cache[0];
    }
}


int main(int argc, char* argv[]) {
    
    size_t vec_len  = 1 << std::stoi(argv[1]);
    size_t size     = vec_len * sizeof(int);
    size_t nthreads = std::stoi(argv[2]);
    size_t nblocks  = std::stoi(argv[3]);
    // size_t nblocks  = (vec_len + nthreads - 1) / nthreads;
    size_t size_out   = nthreads*nblocks*sizeof(int);
    size_t size_out_2 = nblocks*sizeof(int);

    int *u     = (int *)malloc(size);
    int *v     = (int *)malloc(size);
    int *out   = (int *)malloc(size_out);
    int *out_2 = (int *)malloc(size_out_2);
    
    int *dev_u, *dev_v, *dev_out, *dev_out_2; // Device arrays
   
    int res_gpu = 0;
    int res_gpu_2 = 0;
    int res_cpu = 0;

    dim3 dimGrid(nblocks, 1, 1);
    dim3 dimBlocks(nthreads, 1, 1);

    // Initiate values
    for(size_t i=0; i<vec_len; ++i) {
        u[i] = i;
        v[i] = i;
    }

    HANDLE_ERROR( cudaMalloc((void**)&dev_u, size) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_v, size) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_out, size_out) );
    HANDLE_ERROR( cudaMalloc((void**)&dev_out_2, size_out_2) );
    HANDLE_ERROR( cudaMemcpy(dev_u, u, size, cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_v, v, size, cudaMemcpyHostToDevice) );
    //HANDLE_ERROR( cudaMemset(dev_out, 0, size_out) );
    //HANDLE_ERROR( cudaMemset(dev_out_2, 0, size_out_2) );



    auto t1_gpu = std::chrono::system_clock::now();

    dotProd <<<dimGrid, dimBlocks>>> (vec_len, dev_u, dev_v, dev_out);

    cudaDeviceSynchronize();
    HANDLE_ERROR( cudaMemcpy(out, dev_out, size_out, cudaMemcpyDeviceToHost) );

    // Reduction
    for(size_t i=0; i<nthreads*nblocks; ++i) {
        res_gpu += out[i];
    }

    auto t2_gpu = std::chrono::system_clock::now();

    
    // CPU version for result-check
    for(size_t i=0; i<vec_len; ++i) {
        res_cpu += u[i] * v[i];
    }
    auto t2_cpu = std::chrono::system_clock::now();

    // GPU version with shared memory
    dotProdWithSharedMem <<<dimGrid, dimBlocks>>> (vec_len, dev_u, dev_v, dev_out_2);

    cudaDeviceSynchronize();
    HANDLE_ERROR( cudaMemcpy(out_2, dev_out_2, size_out_2, cudaMemcpyDeviceToHost) );

    // Reduction
    for(size_t i=0; i<nblocks; ++i) {
        res_gpu_2 += out_2[i];
    }
    auto t2_gpu_2 = std::chrono::system_clock::now();

    double t_gpu   = std::chrono::duration <double, std::milli> (t2_gpu - t1_gpu).count();
    double t_gpu_2 = std::chrono::duration <double, std::milli> (t2_gpu_2 - t2_cpu).count();
    double t_cpu   = std::chrono::duration <double, std::milli> (t2_cpu - t2_gpu).count();

    printf("GPU result: %i, time consummed: %.5f ms\n", res_gpu, t_gpu);
    printf("GPU result: %i, time consummed: %.5f ms\n", res_gpu_2, t_gpu_2);
    printf("CPU result: %i, time consummed: %.5f ms\n", res_cpu, t_cpu);

    cudaFree(dev_u);
    cudaFree(dev_v);
    cudaFree(dev_out);
    cudaFree(dev_out_2);
    free(u);
    free(v);
    free(out);
    free(out_2);

    return 0;
}

