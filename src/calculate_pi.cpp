#include <iostream>
#include <cmath>
#include <string>
#include <chrono>


typedef unsigned long long ullong;


__global__
double calc_PI (ullong t_n) {
    
    double _tp, _pi(0.);

    for (int i = 0; i < t_n; ++i) {
        _tp = (i + .5) / t_n;
        _pi += 4. / (1. + _tp*_tp);
    }

    return _pi/t_n;
}


int main(int argc, char const *argv[])
{
    
    int      _pow = std::stoi(argv[1]);
    ullong n = 1<<_pow;
    double   res;

    std::cout<< n;
    auto t1 = std::chrono::system_clock::now();
    res = calc_PI<<<1, 1>>>(n);
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    auto t2 = std::chrono::system_clock::now();

    std::cout << "Result: " << std::to_string(res) << '\n'
              << "Computational time: "
              << std::chrono::duration <double, std::milli> (t2 - t1).count()
              << " ms.\n";

    return 0;
}


