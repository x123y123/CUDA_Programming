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

    int *gpu_data, *result, gpu_sum, cpu_sum = 0;
    cudaMalloc((void**) &gpu_data, sizeof(int) * DATA_SIZE);
    cudaMalloc((void**) &result, sizeof(int));
    cudaMemcpy(gpu_data, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice); // cpu to gpu

    sum_of_squares<<<1, 1, 0>>>(gpu_data, result);                               // <<<numbers of block, numbers of thread, sizeof shared memory>>>(var...); 

    cudaMemcpy(&gpu_sum, result, sizeof(int), cudaMemcpyDeviceToHost);           // gpu to cpu
   

    // cpu check data
    for(int i = 0; i < DATA_SIZE; i++) {
        cpu_sum += data[i] * data[i];
    }


    printf("gpu_sum: %d\n", gpu_sum);
    printf("cpu_sum: %d\n", cpu_sum);
    
    cudaFree(gpu_data);
    cudaFree(result);
    return 0;
}
