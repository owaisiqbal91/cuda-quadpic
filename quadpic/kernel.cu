
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "Cimg.h"

#include <stdio.h>

#define noOfIterations 2500
#define noOfQuads noOfIterations*3

#define rows 32//image rows
#define columns 32//image columns
#define noOfThreadsInBlock 16
//#define noOfBlocks 4
#define noOfThreadInBlockBy2 noOfThreadsInBlock/2
//#define gridDim3 dim3(noOfBlocks/2, noOfBlocks/2)
//#define blockDim3 dim3(noOfThreadInBlockBy2, noOfThreadInBlockBy2)

//using namespace cimg_library;

cudaError_t generateOutputWithCuda(int *cRed, int *cGreen, int *cBlue, const int *aRed, const int *aGreen, const int *aBlue, unsigned int size);

__device__ int getAbsoluteIndex(int threadIndexX, int threadIndexY, int quadStartX, int quadStartY) {
	int xIndexInGrid = blockIdx.x*blockDim.x + threadIndexX;//global x in the grid
	int yIndexInGrid = blockIdx.y*blockDim.y + threadIndexY;//global y in the grid

	xIndexInGrid += quadStartX;
	yIndexInGrid += quadStartY;

	return xIndexInGrid + yIndexInGrid*rows;
}

__global__ void getAverageKernel(const int *aRed, const int *aGreen, const int *aBlue, int *mutex, float* average, int quadNo, int quadStartX, int quadStartY, int quadEndX, int quadEndY)
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

					sum[threadIndex] = aRed[index];
					//printf("\nSum[%d] = %d", threadIndex, index);
					int x2 = threadIdx.x;// + i) % blockDim.x;
					int y2 = threadIdx.y + (i / blockDim.x);
					int threadAbsX2 = (x2 + blockDim.x*blockIdx.x) + quadStartX;
					int threadAbsY2 = (y2 + blockDim.y*blockIdx.y) + quadStartY;
					if (threadAbsX2 <= quadEndX && threadAbsY2 <= quadEndY) {
						int index2 = getAbsoluteIndex(x2, y2, quadStartX, quadStartY);

						sum[threadIndex] += aRed[index2];
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
			average[quadNo] += (float)sum[0] / ((quadEndX - quadStartX + 1)*(quadEndY - quadStartY + 1));
			atomicExch(mutex + quadNo, 0);  //unlock
		}
	}
}

__global__ void getScoreAndPaintKernel(const int *aRed, const int *aGreen, const int *aBlue, int *cRed, int *cGreen, int *cBlue, int *mutex, float* average, float* score, int quadNo, int quadStartX, int quadStartY, int quadEndX, int quadEndY) {

	int threadAbsX = (threadIdx.x + blockDim.x*blockIdx.x) + quadStartX;
	int threadAbsY = (threadIdx.y + blockDim.y*blockIdx.y) + quadStartY;

	if (threadAbsX <= quadEndX && threadAbsY <= quadEndY) {
		__shared__ float error[noOfThreadInBlockBy2];
		int index = getAbsoluteIndex(threadIdx.x, threadIdx.y, quadStartX, quadStartY);
		int threadIndex = threadIdx.x + threadIdx.y*blockDim.x;//local 1d index into the block
		int localKernelNoOfThreadsHalf = (blockDim.x * blockDim.y) / 2;
		unsigned int i = localKernelNoOfThreadsHalf;
		float avg = average[quadNo];
		cRed[index] = avg;
		while (i != 0) {
			if (threadIndex < i) {
				if (i == localKernelNoOfThreadsHalf) {//first iteration

					error[threadIndex] = pow(aRed[index] - avg, 2);

					int x2 = threadIdx.x;// + i) % blockDim.x;
					int y2 = threadIdx.y + (i / blockDim.x);
					int threadAbsX2 = (x2 + blockDim.x*blockIdx.x) + quadStartX;
					int threadAbsY2 = (y2 + blockDim.y*blockIdx.y) + quadStartY;
					if (threadAbsX2 <= quadEndX && threadAbsY2 <= quadEndY) {
						int index2 = getAbsoluteIndex(x2, y2, quadStartX, quadStartY);
						error[threadIndex] += pow(aRed[index2] - avg, 2);
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
			//printf("\nError[0] %f", error[0]);
			while (atomicCAS(mutex + quadNo, 0, 1) != 0);  //lock
			//printf("\nBlock %d %d for quad %d error is %f", blockIdx.x, blockIdx.y, quadNo, error[0]);
			//printf("\nQuad %d startX %d startY %d endX %d endY %d", quadNo, quadStartX, quadStartY, quadEndX, quadEndY);
			score[quadNo] += error[0] / ((quadEndX - quadStartX + 1)*(quadEndY - quadStartY + 1));
			atomicExch(mutex + quadNo, 0);  //unlock
		}
	}
}

__global__ void getMaxScoreKernel(float* globalScores, float* maxScore, int* maxScoreIndex, int currentTotalQuads, int *mutex) {
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
			//printf("\nMax value is %f, %d", maxValues[0], maxIndex[0]);

			while (atomicCAS(mutex, 0, 1) != 0);  //lock
			if (maxValues[0] > maxScore[0]) {
				maxScore[0] = maxValues[0];
				maxScoreIndex[0] = maxIndex[0];
			}
			atomicExch(mutex, 0);  //unlock
		}
	}
}

__device__ struct quad {
	short int startX;
	short int startY;
	short int endX;
	short int endY;
};


__global__ void kernelToRuleThemAll(int *cRed, int *cGreen, int *cBlue, const int *aRed, const int *aGreen, const int *aBlue, int *mutex, float* average, float *score, float* globalScores, float* maxScore, int* maxScoreIndex) {

	quad quads[noOfQuads];
	quads[0].startX = 0;
	quads[0].startY = 0;
	quads[0].endX = columns - 1;
	quads[0].endY = rows - 1;

	int quadToSplit = 0;
	int currentTotalQuads = 1;

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
	cudaStream_t s2;
	cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
	cudaStream_t s3;
	cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);
	cudaStream_t s4;
	cudaStreamCreateWithFlags(&s4, cudaStreamNonBlocking);

	const int blockDimSize = 2;
	dim3 blockdim = dim3(blockDimSize, blockDimSize);
	int i;
	for (i = 0; i < 2; i++) {
		//printf("\n\n----------------New SPLIT -----------------");
		int quadW = (quads[quadToSplit].endX - quads[quadToSplit].startX + 1) / 2;
		int quadH = (quads[quadToSplit].endY - quads[quadToSplit].startY + 1) / 2;

		//set grid dim
		int gridDimSize = quadW / blockDimSize;
		gridDimSize = gridDimSize == 0? 1: gridDimSize;
		//printf("\nGrid dim size %d, Quad w & height %d %d", gridDimSize, quadW, quadH);
		dim3 griddim = dim3(gridDimSize, gridDimSize);//will have to use ceil function for non multiple of blockdim

		//reset working variables
		score[0] = 0;
		score[1] = 0;
		score[2] = 0;
		score[3] = 0;
		average[0] = 0;
		average[1] = 0;
		average[2] = 0;
		average[3] = 0;
		maxScore[0] = 0;
		maxScoreIndex[0] = 0;

		//QUAD 2
		quads[currentTotalQuads].startX = quads[quadToSplit].startX + quadW; 
		quads[currentTotalQuads].startY = quads[quadToSplit].startY;
		quads[currentTotalQuads].endX = quads[currentTotalQuads].startX + quadW - 1;
		quads[currentTotalQuads].endY = quads[currentTotalQuads].startY + quadH - 1;
		//launch kernels for the quad index 1
		getAverageKernel<<<griddim, blockdim, 0, s1>>>(aRed, aGreen, aBlue, mutex, average, 1, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		getScoreAndPaintKernel<<<griddim, blockdim, 0, s1>>>(aRed, aGreen, aBlue, cRed, cGreen, cBlue, mutex, average, score, 1, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		currentTotalQuads++;
		printf("\nstartx %d, starty %d,   endx %d, endy %d", quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);

		//QUAD 3
		quads[currentTotalQuads].startX = quads[quadToSplit].startX;
		quads[currentTotalQuads].startY = quads[quadToSplit].startY + quadH;
		quads[currentTotalQuads].endX = quads[currentTotalQuads].startX + quadW - 1;
		quads[currentTotalQuads].endY = quads[currentTotalQuads].startY + quadH - 1;
		//launch kernels for the quad index 2
		getAverageKernel <<<griddim, blockdim, 0, s2 >>>(aRed, aGreen, aBlue, mutex, average, 2, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		getScoreAndPaintKernel <<<griddim, blockdim, 0, s2 >>>(aRed, aGreen, aBlue, cRed, cGreen, cBlue, mutex, average, score, 2, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		currentTotalQuads++;

		//QUAD 4
		quads[currentTotalQuads].startX = quads[quadToSplit].startX + quadW;
		quads[currentTotalQuads].startY = quads[quadToSplit].startY + quadH;
		quads[currentTotalQuads].endX = quads[currentTotalQuads].startX + quadW - 1;
		quads[currentTotalQuads].endY = quads[currentTotalQuads].startY + quadH - 1;
		//launch kernels for the quad index 3
		getAverageKernel <<<griddim, blockdim, 0, s3 >>>(aRed, aGreen, aBlue, mutex, average, 3, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		getScoreAndPaintKernel <<<griddim, blockdim, 0, s3 >>>(aRed, aGreen, aBlue, cRed, cGreen, cBlue, mutex, average, score, 3, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		currentTotalQuads++;

		//QUAD 1
		quads[quadToSplit].endX = quads[quadToSplit].startX + quadW - 1; 
		quads[quadToSplit].endY = quads[quadToSplit].startY + quadH - 1;
		//launch kernels for the quad index 3
		getAverageKernel <<<griddim, blockdim, 0, s4 >>>(aRed, aGreen, aBlue, mutex, average, 0, quads[quadToSplit].startX, quads[quadToSplit].startY, quads[quadToSplit].endX, quads[quadToSplit].endY);
		getScoreAndPaintKernel <<<griddim, blockdim, 0, s4 >>>(aRed, aGreen, aBlue, cRed, cGreen, cBlue, mutex, average, score, 0, quads[quadToSplit].startX, quads[quadToSplit].startY, quads[quadToSplit].endX, quads[quadToSplit].endY);

		cudaDeviceSynchronize();
		//TODO error calculation for rgb and add to score
		printf("\n%f, %f, %f, %f", score[0], score[1], score[2], score[3]);
		//quad 0
		globalScores[quadToSplit] = sqrt(score[0]);
		//quad 1
		globalScores[currentTotalQuads-3] = sqrt(score[1]);
		//quad 2
		globalScores[currentTotalQuads-2] = sqrt(score[2]);
		//quad 3
		globalScores[currentTotalQuads-1] = sqrt(score[3]);

		//find max
		int maxKernelBlockSize = currentTotalQuads > 1024? 1024 : currentTotalQuads;
		int maxKernelGridSize = ceil((float)currentTotalQuads /1024);
		getMaxScoreKernel <<<maxKernelGridSize, maxKernelBlockSize>>> (globalScores, maxScore, maxScoreIndex, 6, mutex);
		cudaDeviceSynchronize();
		printf("\nMax index is %d, max is %f", maxScoreIndex[0], globalScores[maxScoreIndex[0]]);
		//TODO select index with max to split next
		if (maxScoreIndex[0] != 0) {
			quadToSplit = currentTotalQuads - (4 - maxScoreIndex[0]);
		}

		printf("\nQuad %d", 0);
		printf("\nAverage is %f", average[0]);
		printf("\nScore is %f", globalScores[quadToSplit]);
		printf("\n");
		printf("\nQuad %d", 1);
		printf("\nAverage is %f", average[1]);
		printf("\nScore is %f", globalScores[currentTotalQuads - 3]);
		printf("\n");
		printf("\nQuad %d", 2);
		printf("\nAverage is %f", average[2]);
		printf("\nScore is %f", globalScores[currentTotalQuads - 2]);
		printf("\n");
		printf("\nQuad %d", 3);
		printf("\nAverage is %f", average[3]);
		printf("\nScore is %f", globalScores[currentTotalQuads - 1]);
		printf("\n");

		int k = 0;
		for (k = 0; k < 256; k++) {
			printf("\n%d", cRed[k]);
		}
	}

	//delete this
	/*int j = 0;
	for (j = 0; j < currentTotalQuads; j++) {
		printf("\nQuad[%d] : (%d, %d) to (%d, %d)", j, quads[j].startX, quads[j].startY, quads[j].endX, quads[j].endY);
	}*/
}

int main()
{
	/*size_t free, total;
	cudaMemGetInfo(&free,
		&total
	);
	printf("\nCUDA FREE: %zu, TOTAL %zu", free, total);*/

	//CImg<unsigned char> image("C:\\Projects\\Visual Studio\\quadpic\\quadpic\\o.bmp");

	/*CImg<float> src("image.jpg");
	int width = src.width();
	int height = src.height();
	unsigned char* ptr = src.data(10, 10);*/

    const int arraySize = rows*columns;
    /*const int imageData[arraySize] = { 0, 1,	4, 5,
									   2, 3,	6, 7,	
									   
									   8, 9,	12, 13,
									   10, 11,	14, 15,
									};*/

	int imageDataR[1024];
	int id;
	for (id = 0; id < 1024; id++) {
		imageDataR[id] = id;
	}
	int imageDataG[1024];
	for (id = 0; id < 1024; id++) {
		imageDataG[id] = id;
	}
	int imageDataB[1024];
	for (id = 0; id < 1024; id++) {
		imageDataB[id] = id;
	}


    int cRed[rows*columns] = { 0 };
	int cGreen[rows*columns] = { 0 };
	int cBlue[rows*columns] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = generateOutputWithCuda(cRed, cGreen, cBlue, imageDataR, imageDataG, imageDataB, arraySize);
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
cudaError_t generateOutputWithCuda(int *cRed, int *cGreen, int *cBlue, const int *aRed, const int *aGreen, const int *aBlue, unsigned int size)
{
    int *dev_aRed = 0;
	int *dev_aGreen = 0;
	int *dev_aBlue = 0;

    int *dev_cRed = 0;
	int *dev_cGreen = 0;
	int *dev_cBlue = 0;

	int *dev_mutex = 0;
	float *dev_average = 0;
	float *dev_score = 0;

	float *dev_globalScores = 0;
	float *dev_maxScore = 0;
	int *dev_maxScoreIndex = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_cRed, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_cGreen, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cBlue, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    cudaStatus = cudaMalloc((void**)&dev_aRed, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_aGreen, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_aBlue, size * sizeof(int));
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

	cudaStatus = cudaMalloc((void**)&dev_maxScore, sizeof(float));
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
    cudaStatus = cudaMemcpy(dev_aRed, aRed, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_aGreen, aGreen, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_aBlue, aBlue, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

    // Launch a kernel on the GPU.
	kernelToRuleThemAll << <1, 1 >> >(dev_cRed, dev_cGreen, dev_cBlue, dev_aRed, dev_aBlue, dev_aGreen, dev_mutex, dev_average, dev_score, dev_globalScores, dev_maxScore, dev_maxScoreIndex);
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

    // Copy output vectors from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(cRed, dev_cRed, rows*columns * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(cGreen, dev_cRed, rows*columns * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cBlue, dev_cRed, rows*columns * sizeof(int), cudaMemcpyDeviceToHost);
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
    cudaFree(dev_cRed);
	cudaFree(dev_cGreen);
	cudaFree(dev_cBlue);
    cudaFree(dev_aRed);
	cudaFree(dev_aGreen);
	cudaFree(dev_aBlue);
    
    return cudaStatus;
}

//one kernel to rule them all, app store near you
// reduction, dynamic parallelism, streams
//TODO set shared memory parameter for child kernel launches?
//TODO setting of mutex variable for average/score kernels and for max kernel<--mutex[0]

//for square images and power of 2 you dont need abs conditions -> will save branch predictions
