#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */

 __global__ void improved_Saxpy( float *d_y, const float *d_x,
    const float alpha, const uint32_t arraySize)
    { // yi = xi * alpha + yi
    // every thread process 4 elements at a time
    uint32_t tid = (threadIdx.x+blockIdx.x*blockDim.x)*4;
    // the elements that all threads on GPU can process at a time
    uint32_t dim = gridDim.x*blockDim.x*4;
    for(uint32_t i = tid; i < arraySize; i += dim)
    asm volatile ("{\t\n"
    // registers to store input operands
    ".reg .f32 a1,b1,c1,d1;\n\t"
    ".reg .f32 a2,b2,c2,d2;\n\t"
    // loading with vectorized, 128-bit instructions
    "ld.global.v4.f32 {a1,b1,c1,d1},[%0];\n\t"
    "ld.global.v4.f32 {a2,b2,c2,d2},[%1];\n\t"
    // core math
    "fma.rn.f32
    "fma.rn.f32
    "fma.rn.f32
    "fma.rn.f32
    operations
    a2,a1,%2,a2;\n\t"
    b2,b1,%2,b2;\n\t"
    c2,c1,%2,c2;\n\t"
    d2,d1,%2,d2;\n\t"
    // storing results with a vectorized, 128-bit write instruction
    "st.global.v4.f32 [%1],{a2,b2,c2,d2};\n\t"
    "}" :: "l"(d_x+i),"l"(d_y+i), "f"(alpha) : "memory"
    );
}

__global__ void poli1(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = idx;

    if (idx < N)
        poli[idx] = 4 * x * x * x + 3 * x * x - 7 * x + 5;
}

__global__ void poli2(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = idx;

    if (idx < N)
        poli[idx] = 3 * x * x - 7 * x + 5;
}

__global__ void poli3(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = idx;

    if (idx < N)
        poli[idx] = 5 + 5 * x + 5 * x * x + 5 * x * x * x + 5 * x * x * x * x + 5 * x * x * x * x * x;
}

__global__ void poli4(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = idx;

    if (idx < N)
        poli[idx] = 5 + 5 * x + 5 * x * sqrt(x) + 5 * sqrt(x)
            * x * x + 5 * x * sqrt(x) * x * x + 5 * x * sqrt(x) * sqrt(x) * x * x;
}

int main() {
    int nElem = 1 << 27;

    size_t nBytes = nElem * sizeof(float);

    float* h_polinomy = (float*)malloc(nBytes);

    float* d_polinomy;
    cudaMalloc((float**)&d_polinomy, nBytes);
  
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    poli1<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);
    
    poli2<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    poli3<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    poli4<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_polinomy);
    free(h_polinomy);
}
