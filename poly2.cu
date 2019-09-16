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

__global__ void init(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
	poli[idx] = idx;
}

__global__ void poli1(float* poli, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
		float x = poli[idx];
        poli[idx] = 4 * x * x * x + 4 * x * x - 4 * x + 4;
    }
}

__global__ void poli1U2(float* poli, const int N) {
    int idx = 2 * threadIdx.x + blockIdx.x * (2 * blockDim.x);

    if (idx < N) {
        float x = poli[idx];
        float y = poli[idx + 1];

        poli[idx] = 4 * x * x * x + 4 * x * x - 4 * x + 4;
        poli[idx + 1] = 4 * y * y * y + 4 * y * y - 4 * y + 4;
     }
}

__global__ void poli1U4(float* poli, const int N) {
    int idx = 4 * threadIdx.x + blockIdx.x * (4 * blockDim.x);

    if (idx < N) {
        float x = poli[idx];
        float y = poli[idx + 1];
		float z = poli[idx + 2];
		float w = poli[idx + 3];

        poli[idx] = 4 * x * x * x + 4 * x * x - 4 * x + 4;
        poli[idx + 1] = 4 * y * y * y + 4 * y * y - 4 * y + 4;
		poli[idx + 2] = 4 * z * z * z + 4 * z * z - 4 * z + 4;
		poli[idx + 3] = 4 * w * w * w + 4 * w * w - 4 * w + 4;
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
        poli[idx] = 5 + 5 * x + 5 * x * x + 5 * x * x * x + 5 * x * x * x * x + 5 * x * x * x * x * x}

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
    float* h_polinomyU2 = (float*)malloc(nBytes);
    float* h_polinomyU4 = (float*)malloc(nBytes);

    float* d_polinomy;
    cudaMalloc((float**)&d_polinomy, nBytes);

    int block = min(512, nElem);
    int grid = nElem % block == 0 ?
	nElem / block : nElem / block + 1;

    int blockU2 = min(512, nElem / 2);
    int gridU2 = (nElem / 2) % blockU2 == 0 ?
	(nElem / 2) / blockU2 : (nElem / 2) / blockU2 + 1;

    int blockU4 = min(512, nElem / 4);
    int gridU4 = (nElem / 4) % blockU4 == 0 ?
	(nElem / 4) / blockU4 : (nElem / 4) / blockU4 + 1;

    printf("Block, Grid: %d %d\nBlockU2, GridU2: %d %d\nBlockU4, GridU4: %d %d\n",
    	block, grid, blockU2, gridU2, blockU4, gridU4);

    init<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();
    poli1<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();
    cudaMemcpy(h_polinomy, d_polinomy, nBytes, cudaMemcpyDeviceToHost); 

    init<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();
    poli1U2<<<gridU2, blockU2>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();
    cudaMemcpy(h_polinomyU2, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    init<<<grid, block>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();
    poli1U4<<<gridU4, blockU4>>>(d_polinomy, nElem);
    cudaDeviceSynchronize();
    cudaMemcpy(h_polinomyU4, d_polinomy, nBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nElem; ++i) {
        printf("(%f, %f, %f) ", h_polinomy[i], h_polinomyU2[i], h_polinomyU4[i]);
/*
        if (abs(h_polinomy[i] - h_polinomyF2[i]) > 1e-10) {
            puts("Deu ruim");
            break;
        }*/
    }

    puts("");

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
    free(h_polinomyU4);
    free(h_polinomyU2);
    free(h_polinomy);
}
