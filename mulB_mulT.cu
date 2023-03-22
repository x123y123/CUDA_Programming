/* add in multi-blocks and 1 thread in gpu */
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N   5000

__global__ void add(int *a, int *b, int *c)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N) {
        c[tid] = a[tid] + b[tid];

        // execute 512 thread-block * 512 thread per time
        tid += blockDim.x * gridDim.x;
    }
}

int main()
{
    int cpu_a[N], cpu_b[N], cpu_c[N];
    int *gpu_a, *gpu_b, *gpu_c;

    cudaMalloc((void**)&gpu_a, N * sizeof(int));
    cudaMalloc((void**)&gpu_b, N * sizeof(int));
    cudaMalloc((void**)&gpu_c, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        cpu_a[i] = 2 * i;
        cpu_b[i] = i;
    }

    cudaMemcpy(gpu_a, cpu_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, cpu_b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    add<<<512, 512>>>(gpu_a, gpu_b, gpu_c);

    cudaMemcpy(cpu_c, gpu_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // waiting for all threads finish
    cudaDeviceSynchronize();

    printf("Finish\n");
    for (int i = 0; i < N; i++) 
        printf("cpu_c[%d] = %d\n", i, cpu_c[i]);
    
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    
    return 0;
}
