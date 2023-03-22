/* add in multi-threads and 1 block in gpu */
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N   5

__global__ void add(int *a, int *b, int *c)
{
    // current thread id
    int tid = threadIdx.x;   
    c[tid] = a[tid] + b[tid];
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
    
    add<<<1, N>>>(gpu_a, gpu_b, gpu_c);

    cudaMemcpy(cpu_c, gpu_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Finish\n");
    for (int i = 0; i < N; i++) 
        printf("cpu_c[%d] = %d\n", i, cpu_c[i]);
    
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    
    return 0;
}
