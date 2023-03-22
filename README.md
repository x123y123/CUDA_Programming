# CUDA Programming

## Install CUDA and compile
* [Install reference](https://shuuutin-cg.medium.com/ubuntu18-04%E5%AE%89%E8%A3%9Dcuda%E4%BB%A5%E5%8F%8A%E6%94%B9%E8%AE%8Acuda%E7%89%88%E6%9C%AC-b8ac917f880f
)
* Compile Nvcc
```shell
# Example: after we write a test.cu we can use following command to compile and run
$ nvcc test.cu 
$ ./a.out
```
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
