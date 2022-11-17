/* A simple CUDA code to learn the memory work between cpu and gpu. */


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1024000

int data[DATA_SIZE];
void generate_numbers(int *number, int size)
{
    for(int i = 0; i < size; i++) {
        number[i] = rand() % 50;                                // get random numbers in range of 0 ~ 49
    }
}


__global__ static void sum_of_squares(int *num, int* result)    // declare with __global__ will let function run in gpu, but it only can use void type
{
    int gpu_sum = 0;
    int i;
    for(i = 0; i < DATA_SIZE; i++) {
        gpu_sum += num[i] * num[i];
    }

    *result = gpu_sum;
}


int main()                                              
{
    generate_numbers(data, DATA_SIZE);

    int *gpudata, *result;
    cudaMalloc((void**) &gpudata, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int));
    cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

    sum_of_squares<<<1, 1, 0>>>(gpudata, result);       // <<<numbers of block, numbers of thread, sizeof shared memory>>>(var...); 

    int gpu_sum;
    cudaMemcpy(&gpu_sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    
    int cpu_sum = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        cpu_sum += data[i] * data[i];
    }


    printf("gpu_sum: %d\n", gpu_sum);
    printf("cpu_sum: %d\n", cpu_sum);
    
    cudaFree(gpudata);
    cudaFree(result);
    return 0;
}
