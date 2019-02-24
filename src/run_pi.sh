#!/bin/bash 

for i in 16384 32768 65536 131072
    do
        nvcc calc_PI_neo.cu -o pi.o -DNTHREADS=512 -DNBLOCKS=$i
        for j in {1..20}
            do
                pi.o
            done
    done

