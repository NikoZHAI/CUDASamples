#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
#include <cuda.h>


typedef unsigned long long ullong;


__global__
void calc_PI (int t_n, float *t_res) {
    
    float _tp, _pi(0.);

    for (int i = 0; i < t_n; ++i) {
        _tp = (i + .5) / t_n;
        _pi += 4. / (1. + _tp*_tp);
    }

    *t_res = _pi/t_n;
}


int main(int argc, char const *argv[])
{   
    int   _pow = std::stoi(argv[1]);
    int   n = 1<<_pow;  // n = 2^_pow
    float *device_res;
    float host_res;
    cudaMalloc((void **)&device_res, sizeof(float));
    // cudaMemcpy(device_res, &, sizeof(float), cudaMemcpyHostToDevice);

    std::cout << n << '\n';
    auto t1 = std::chrono::system_clock::now();
    calc_PI<<<1, 256>>>(n, device_res);
    auto t2 = std::chrono::system_clock::now();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    cudaMemcpy(&host_res, device_res, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << std::to_string(*device_res) << '\n'
              << "Computational time: "
              << std::chrono::duration <double, std::milli> (t2 - t1).count()
              << " ms.\n";
    
    // Free 
    cudaFree(device_res); cudaFree(device_n);

    return 0;
}


