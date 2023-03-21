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
> * Size: `Grid`(huge) > `Block` > `Thread`(small)
> * Execution unit is `Block`.

## Data transmission between CPU and GPU
```c
#include <stdio.h>

__global__ void add(int a, int b, int *c)
{
    *c = a + b;
}

int main() 
{
    int cpu_c;
    int *gpu_c;
    
    // melloc on GPU
    // cudaError_t cudaMalloc (void **devPtr, size_t size)
    cudaMalloc((void**)&gpu_c, sizeof(int));
    
    // calculate on GPU
    add<<<1, 1>>>(1, 2, gpu_c);
    
    // transmision the data from GPU to CPU
    // cudaError_t cudaMemcpy (void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    cudaMemcpy(&cpu_c, gpu_c, sizeof(int), cudaMemcpyDeviceToHost);
    
    // cudaError_t cudaFree (void* devPtr)
    cudaFree(gpu_c);
    
    return 0;
}
```
> Flow: cudaMelloc() -> cudaMemcpy() -> cudaFree()
