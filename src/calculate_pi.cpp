#include <iostream>
#include <cmath>
#include <string>
#include <chrono>


typedef unsigned long long ullong;

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
    ullong   n = 1<<_pow;
    double   res;

    std::cout << "n = " << n << '\n';
    auto t1 = std::chrono::system_clock::now();
    res = calc_PI(n);
    auto t2 = std::chrono::system_clock::now();

    printf("PI = %1.18f\n", res);
    std::cout << "Computation time: "
              << std::chrono::duration <double, std::milli> (t2 - t1).count()
              << " ms.\n";

    return 0;
}


