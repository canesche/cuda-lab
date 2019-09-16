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

__global__ void poli1(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = idx;

    if (idx < N)
        poli[idx] = 4 * x * x * x + 4 * x * x - 4 * x + 4;
}

__global__ void poli1F1(float* poli, const int N) {
    int idx = 2 * threadIdx.x + blockIdx.x * (2 * blockDim.x);

    if (idx < N) {
        float x = idx;
        float y = idx + 1;

        poli[idx] = 4 * x * x * x + 4 * x * x - 4 * x + 4;
        poli[idx + 1] = 4 * y * y * y + 4 * y * y - 4 * y + 4;
    }
}

__global__ void poli1F2(float2* poli, const int N) {
    int idx = 2 * threadIdx.x + blockIdx.x * (2 * blockDim.x);

    if (idx < N) {
        float x = idx;
        float y = idx + 1;

        poli[idx].x = 4 * x * x * x + 4 * x * x - 4 * x + 4;
        poli[idx].y = 4 * y * y * y + 4 * y * y - 4 * y + 4;
    }
}
/*
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
        poli[idx] = 5 + 5 * x + 5 * x * sqrt(x) + 5 * sqrt(x) * x * x + 5 * x *
            sqrt(x) * x * x + 5 * x * sqrt(x) * sqrt(x) * x * x;
}
*/
int main() {
    //int nElem = 1 << 26;
    int nElem = 1 << 4;

    size_t nBytes = nElem * sizeof(float);

    float* h_polinomy = (float*)malloc(nBytes);
    float* h_polinomyF1 = (float*)malloc(nBytes);
    float* h_polinomyF2 = (float*)malloc(nBytes);

    float* d_polinomy;
    cudaMalloc((float**)&d_polinomy, nBytes);
  
    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    poli1<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    dim3 blockF2 (iLen / 2);
    dim3 gridF2  ((nElem + blockF2.x - 1) / blockF2.x);

    poli1F1<<<gridF2, blockF2>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomyF1, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    poli1F2<<<gridF2, blockF2>>>((float2*)d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomyF2, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nElem; ++i) {

        printf("(%f, %f, %f) ", h_polinomy[i], h_polinomyF1[i], h_polinomyF2[i]);
/*
        if (abs(h_polinomy[i] - h_polinomyF2[i]) > 1e-10) {
            puts("Deu ruim");
            break;
        }*/
    }

    printf("%f %f\n", h_polinomy[0], h_polinomyF2[0]);
    fflush(stdout);

    /*
    poli2<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    poli3<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    poli4<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();

    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost);
*/
    cudaFree(d_polinomy);
    free(h_polinomyF2);
    free(h_polinomyF1);
    free(h_polinomy);
}
