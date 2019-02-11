#include <iostream>
#include <stdio.h>
#include <cmath>

int main(int argc, char *argv[]){

    int n = 1<<std::stoi(argv[1]);
    double x_float, x_double;


    for(int i = n/2; i < n; i++)
    {
        x_float = (i + .5f) / n;
        printf("%10i:f %1.20f\n", i, x_float);

        x_double = (i + .5) / n;
        printf("%10i:d %1.20f\n", i, x_double);
    }


    return 0;
}

