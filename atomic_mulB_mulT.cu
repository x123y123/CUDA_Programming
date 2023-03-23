#include <stdio.h>

#define NUM_THREAD 1000
#define SIZE 10

#define BLOCK_WIDTH 100 // number of threads in a thread_block

__global__ void add_with_atomic(int *gpu_data)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    tid = tid % SIZE;
    //gpu_data[tid] += 1;           // not using atomic to add will cause some issues
    atomicAdd(&gpu_data[tid], 1);   //using atomic operation to add
    printf("%d\n", gpu_data[tid]);
}

int main()
{
    int cpu_data[SIZE];
    int *gpu_data;
    const int ARRAY_BYTES = SIZE * sizeof(int);

    for (int i = 0; i < SIZE; i++)
        cpu_data[i] = i * i;

    cudaMalloc((void**)&gpu_data, ARRAY_BYTES);
    cudaMemset((void*)gpu_data, 0, ARRAY_BYTES);
    add_with_atomic<<<NUM_THREAD / BLOCK_WIDTH, BLOCK_WIDTH>>>(gpu_data);
    
    cudaMemcpy(cpu_data, gpu_data, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaFree(gpu_data);
    
    return 0 ;

}
