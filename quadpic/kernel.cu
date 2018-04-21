
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define noOfIterations 2500
#define noOfQuads noOfIterations*3

#define rows 8//image rows
#define columns 8//image columns
#define noOfThreadsInBlock 16
//#define noOfBlocks 4
#define noOfThreadInBlockBy2 noOfThreadsInBlock/2
//#define gridDim3 dim3(noOfBlocks/2, noOfBlocks/2)
//#define blockDim3 dim3(noOfThreadInBlockBy2, noOfThreadInBlockBy2)

cudaError_t generateOutputWithCuda(int *c, const int *a, unsigned int size);

__device__ int getAbsoluteIndex(int threadIndexX, int threadIndexY, int quadStartX, int quadStartY) {
	int xIndexInGrid = blockIdx.x*blockDim.x + threadIndexX;//global x in the grid
	int yIndexInGrid = blockIdx.y*blockDim.y + threadIndexY;//global y in the grid

	xIndexInGrid += quadStartX;
	yIndexInGrid += quadStartY;

	return xIndexInGrid + yIndexInGrid*rows;
}

__global__ void getAverageKernel(const int *a, int *mutex, float* average, int quadNo, int quadStartX, int quadStartY, int quadEndX, int quadEndY)
{
	int threadAbsX = (threadIdx.x + blockDim.x*blockIdx.x) + quadStartX;
	int threadAbsY = (threadIdx.y + blockDim.y*blockIdx.y) + quadStartY;

	if (threadAbsX <= quadEndX && threadAbsY <= quadEndY /*&& blockIdx.x == 0 && blockIdx.y == 0*/) {
		__shared__ int sum[noOfThreadInBlockBy2];
		int index = getAbsoluteIndex(threadIdx.x, threadIdx.y, quadStartX, quadStartY);
		int threadIndex = threadIdx.x + threadIdx.y*blockDim.x;//local 1d index into the block
		int localKernelNoOfThreadsHalf = (blockDim.x * blockDim.y) / 2;
		unsigned int i = localKernelNoOfThreadsHalf;
		while (i != 0) {
			if (threadIndex < i) {
				if (i == localKernelNoOfThreadsHalf) {//first iteration

					sum[threadIndex] = a[index];
					//printf("\nSum[%d] = %d", threadIndex, index);
					int x2 = threadIdx.x;// + i) % blockDim.x;
					int y2 = threadIdx.y + (i / blockDim.x);
					int threadAbsX2 = (x2 + blockDim.x*blockIdx.x) + quadStartX;
					int threadAbsY2 = (y2 + blockDim.y*blockIdx.y) + quadStartY;
					if (threadAbsX2 <= quadEndX && threadAbsY2 <= quadEndY) {
						int index2 = getAbsoluteIndex(x2, y2, quadStartX, quadStartY);

						sum[threadIndex] += a[index2];
						//printf("\nSum[%d] 2nd = %d, threadIdy: %d, i: %d, blockDimy: %d, calc: %d, y2: %d", threadIndex, index2, threadIdx.y, i, blockDim.x, (i+threadIdx.y)/blockDim.x, y2);
					}
				}
				else {
					sum[threadIndex] += sum[threadIndex + i];
				}

			}

			__syncthreads();
			i /= 2;
		}

		if (threadIndex == 0) {
			//int blockIndex = blockIdx.x + blockIdx.y*blockDim.x;
			while (atomicCAS(mutex + quadNo, 0, 1) != 0);  //lock
			printf("\nBlock sum is %d", sum[0]);
			average[quadNo] += (float)sum[0] / ((quadEndX - quadStartX + 1)*(quadEndY - quadStartY + 1));
			atomicExch(mutex + quadNo, 0);  //unlock
		}
	}
}

__global__ void getScoreAndPaintKernel(const int *a, int *mutex, float* average, float* score, int quadNo, int quadStartX, int quadStartY, int quadEndX, int quadEndY) {

	int threadAbsX = (threadIdx.x + blockDim.x*blockIdx.x) + quadStartX;
	int threadAbsY = (threadIdx.y + blockDim.y*blockIdx.y) + quadStartY;

	if (threadAbsX <= quadEndX && threadAbsY <= quadEndY) {
		__shared__ float error[noOfThreadInBlockBy2];
		int index = getAbsoluteIndex(threadIdx.x, threadIdx.y, quadStartX, quadStartY);
		int threadIndex = threadIdx.x + threadIdx.y*blockDim.x;//local 1d index into the block
		int localKernelNoOfThreadsHalf = (blockDim.x * blockDim.y) / 2;
		unsigned int i = localKernelNoOfThreadsHalf;
		float avg = average[quadNo];
		while (i != 0) {
			if (threadIndex < i) {
				if (i == localKernelNoOfThreadsHalf) {//first iteration

					error[threadIndex] = pow(a[index] - avg, 2);

					int x2 = threadIdx.x;// + i) % blockDim.x;
					int y2 = threadIdx.y + (i / blockDim.x);
					int threadAbsX2 = (x2 + blockDim.x*blockIdx.x) + quadStartX;
					int threadAbsY2 = (y2 + blockDim.y*blockIdx.y) + quadStartY;
					if (threadAbsX2 <= quadEndX && threadAbsY2 <= quadEndY) {
						int index2 = getAbsoluteIndex(x2, y2, quadStartX, quadStartY);
						error[threadIndex] += pow(a[index2] - avg, 2);
					}
				}
				else {
					error[threadIndex] += error[threadIndex + i];
				}

			}

			__syncthreads();
			i /= 2;
		}

		if (threadIndex == 0) {
			while (atomicCAS(mutex + quadNo, 0, 1) != 0);  //lock
			score[quadNo] += error[0] / ((quadEndX - quadStartX + 1)*(quadEndY - quadStartY + 1));
			atomicExch(mutex + quadNo, 0);  //unlock
		}
	}
}

__global__ void getMaxScoreKernel(float* globalScores, int* maxScoreIndex, int currentTotalQuads) {
	__shared__ float maxValues[512];
	__shared__ int maxIndex[512];

	int localKernelNoOfThreadsHalf = blockDim.x / 2;
	unsigned int i = localKernelNoOfThreadsHalf;

	int threadGlobalIndex = blockIdx.x*blockDim.x + threadIdx.x;

	if (threadGlobalIndex < currentTotalQuads) {
		while (i != 0) {
			if (threadIdx.x < i) {
				if (i == localKernelNoOfThreadsHalf) {//first iteration

					maxValues[threadIdx.x] = globalScores[threadGlobalIndex];
					maxIndex[threadIdx.x] = threadGlobalIndex;

					int threadGlobalIndex2 = threadGlobalIndex + i;
					
					if (threadGlobalIndex2 < currentTotalQuads) {
						if (globalScores[threadGlobalIndex2] > maxValues[threadIdx.x]) {
							maxValues[threadIdx.x] = globalScores[threadGlobalIndex2];
							maxIndex[threadIdx.x] = threadGlobalIndex2;
						}
					}
				}
				else {
					if (maxValues[threadIdx.x + i] > maxValues[threadIdx.x]) {
						maxValues[threadIdx.x] = maxValues[threadIdx.x + i];
						maxIndex[threadIdx.x] = maxIndex[threadIdx.x + i];
					}
				}
			}

			__syncthreads();
			i /= 2;
		}

		if (threadIdx.x == 0) {
			printf("\nMax value is %f, %d", maxValues[0], maxIndex[0]);
		}
	}
}

__device__ struct quad {
	short int startX;
	short int startY;
	short int endX;
	short int endY;
};


__global__ void kernelToRuleThemAll(int *c, const int *a, int *mutex, float* average, float *score, float* globalScores, int* maxScoreIndex) {

	quad quads[noOfQuads];

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

	//TODO shared memory size
	//getAverageKernel <<<dim3(4,4), dim3(4,4), 0, s1>>>(a, mutex, average, 1, 0, 0, 7, 7);
	//getScoreAndPaintKernel << <dim3(4, 4), dim3(4,4), 0, s1>> >(a, mutex, average, score, 1, 0, 0, 7, 7);
	//TODO remember to square root error

	//find max
	globalScores[0] = 1;
	globalScores[1] = 2;
	globalScores[2] = 3;
	globalScores[3] = 4;
	globalScores[4] = 5;
	globalScores[5] = 6;
	getMaxScoreKernel<<<1, 8>>>(globalScores, maxScoreIndex, 6);

	/*cudaDeviceSynchronize();
	printf("\nAverage is %f", average[1]);
	printf("\nScore is %f", score[1]);*/
}

int main()
{
	/*size_t free, total;
	cudaMemGetInfo(&free,
		&total
	);
	printf("\nCUDA FREE: %zu, TOTAL %zu", free, total);*/

    const int arraySize = rows*columns;
    /*const int imageData[arraySize] = { 0, 1,	4, 5,
									   2, 3,	6, 7,	
									   
									   8, 9,	12, 13,
									   10, 11,	14, 15,
									};*/

	int imageData[64];
	int id;
	for (id = 0; id < 64; id++) {
		imageData[id] = id;
	}
    int c[rows*columns] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = generateOutputWithCuda(c, imageData, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "generateOutputWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t generateOutputWithCuda(int *c, const int *a, unsigned int size)
{
    int *dev_a = 0;
    int *dev_c = 0;
	int *dev_mutex = 0;
	float *dev_average = 0;
	float *dev_score = 0;
	float *dev_globalScores = 0;
	int *dev_maxScoreIndex = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_mutex, 4 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_average, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_score, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_globalScores, noOfQuads * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_maxScoreIndex, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU.
	kernelToRuleThemAll << <1, 1 >> >(dev_c, dev_a, dev_mutex, dev_average, dev_score, dev_globalScores, dev_maxScoreIndex);
    //getAverageKernel<<<gridDim3, blockDim3>>>(dev_c, dev_a, dev_mutex, dev_average, 0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Main kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernelToRuleThemAll!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, rows*columns * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	/*float avg[4] = { 0 };
	float *average = avg;
	cudaStatus = cudaMemcpy(average, dev_average, 4 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}*/

	/*int i = 0;
	for (i = 0; i < noOfBlocks; i++) {
		printf("\nOutput %d", c[i]);
	}*/

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    
    return cudaStatus;
}

//one kernel to rule them all, app store near you
// reduction, dynamic parallelism, streams
