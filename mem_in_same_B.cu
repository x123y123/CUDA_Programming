#include <stdio.h>


__global__ void share_memory(float *gpu_data)
{
    int tid = threadIdx.x;
    float sum = 0.0f, average;

    // declare shared memory
    __shared__ float shar_mem[10];
    shar_mem[tid] = gpu_data[tid];
    
    // make sure the shared data write is complete
    __syncthreads();

    for (int i = 0; i < tid; i++)
        sum += shar_mem[i];

    average = sum / (tid + 1.0f);
    gpu_data[tid] = average;
    printf("%f\n", gpu_data[tid]);
    printf("\n");
    shar_mem[tid] = average;
    printf("%f\n", shar_mem[tid]);
    printf("\n");

}

int main()
{
    float cpu_data[10];
    float *gpu_data;

    for (int i = 0; i < 10; i++)
        cpu_data[i] = i * i;

    cudaMalloc((void**)&gpu_data, sizeof(float) * 10);
    cudaMemcpy(gpu_data, cpu_data, sizeof(float) * 10, cudaMemcpyHostToDevice);
    share_memory<<<1, 10>>>(gpu_data);
    
    cudaMemcpy(cpu_data, gpu_data, sizeof(float) * 10, cudaMemcpyDeviceToHost);
    cudaFree(gpu_data);
    
    return 0 ;

}
