# CUDA Programming

## Install CUDA

## Hello World

```c
#include <stdio.h>

// Run on GPU(when we see __global__ it means it run on GPU)
// __global__ can be called by CPU, but __device__ only can be called by __global__ or __device__
__global__ void kernel()
{
    printf("Hello World!");
}

// Run on CPU
int main()
{
    // <<<variable_1, variable2>>> 
    // * kernel function has variable_1 thread blocks
    // * In each thread blocks has variable_2 threads
    kernel<<<1, 1>>>(); 
    return 0;
}
```
> * Size: Grid(huge) > Block > Thread(small)
> * Execution unit is `Block`.


