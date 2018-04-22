#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Cimg.h"

#include <stdio.h>

#define noOfIterations 2500
#define noOfQuads noOfIterations*3

#define rows 256//image rows
#define columns 256//image columns
#define noOfThreadsInBlock 1024
//#define noOfBlocks 4
#define noOfThreadInBlockBy2 noOfThreadsInBlock/2
//#define gridDim3 dim3(noOfBlocks/2, noOfBlocks/2)
//#define blockDim3 dim3(noOfThreadInBlockBy2, noOfThreadInBlockBy2)

using namespace cimg_library;

cudaError_t generateOutputWithCuda(int *cRed, int *cGreen, int *cBlue, const int *aRed, const int *aGreen, const int *aBlue, unsigned int size);

__device__ int getAbsoluteIndex(int threadIndexX, int threadIndexY, int quadStartX, int quadStartY) {
	int xIndexInGrid = blockIdx.x*blockDim.x + threadIndexX;//global x in the grid
	int yIndexInGrid = blockIdx.y*blockDim.y + threadIndexY;//global y in the grid

	xIndexInGrid += quadStartX;
	yIndexInGrid += quadStartY;

	return xIndexInGrid + yIndexInGrid*rows;
}

__global__ void getAverageKernel(const int *aRed, const int *aGreen, const int *aBlue, int *mutex, float* averageRed, float* averageGreen, float* averageBlue, int quadNo, int quadStartX, int quadStartY, int quadEndX, int quadEndY)
{
	int threadAbsX = (threadIdx.x + blockDim.x*blockIdx.x) + quadStartX;
	int threadAbsY = (threadIdx.y + blockDim.y*blockIdx.y) + quadStartY;

	if (threadAbsX <= quadEndX && threadAbsY <= quadEndY /*&& blockIdx.x == 0 && blockIdx.y == 0*/) {
		__shared__ int sumRed[noOfThreadInBlockBy2];
		__shared__ int sumGreen[noOfThreadInBlockBy2];
		__shared__ int sumBlue[noOfThreadInBlockBy2];
		int index = getAbsoluteIndex(threadIdx.x, threadIdx.y, quadStartX, quadStartY);
		int threadIndex = threadIdx.x + threadIdx.y*blockDim.x;//local 1d index into the block
		int localKernelNoOfThreadsHalf = (blockDim.x * blockDim.y) / 2;
		unsigned int i = localKernelNoOfThreadsHalf;
		while (i != 0) {
			if (threadIndex < i) {
				int x2 = threadIdx.x;// + i) % blockDim.x;
				int y2 = threadIdx.y + (i / blockDim.x);
				int threadAbsX2 = (x2 + blockDim.x*blockIdx.x) + quadStartX;
				int threadAbsY2 = (y2 + blockDim.y*blockIdx.y) + quadStartY;
				if (i == localKernelNoOfThreadsHalf) {//first iteration
					/*printf("\nblockdim %d", blockDim.x);
					if (blockDim.x == 16) 
						printf("\n16x16 %d %d", threadIndex, threadIndex + i);*/
					sumRed[threadIndex] = 0;
					sumRed[threadIndex] = aRed[index];
					sumGreen[threadIndex] = aGreen[index];
					sumBlue[threadIndex] = aBlue[index];
					if (threadAbsX2 <= quadEndX && threadAbsY2 <= quadEndY) {
						int index2 = getAbsoluteIndex(x2, y2, quadStartX, quadStartY);

						sumRed[threadIndex] += aRed[index2];
						sumGreen[threadIndex] += aGreen[index2];
						sumBlue[threadIndex] += aBlue[index2];
						//printf("\nSum[%d] 2nd = %d, threadIdy: %d, i: %d, blockDimy: %d, calc: %d, y2: %d", threadIndex, index2, threadIdx.y, i, blockDim.x, (i+threadIdx.y)/blockDim.x, y2);
					}
				}
				else {
					if (threadAbsX2 <= quadEndX && threadAbsY2 <= quadEndY) {
						//if (blockDim.x == 16) printf("\n16x16 %d %d", threadIndex, threadIndex+i);
						sumRed[threadIndex] += sumRed[threadIndex + i];
						sumGreen[threadIndex] += sumGreen[threadIndex + i];
						sumBlue[threadIndex] += sumBlue[threadIndex + i];
					}
				}

			}

			__syncthreads();
			i /= 2;
		}

		if (threadIndex == 0) {
			//int blockIndex = blockIdx.x + blockIdx.y*blockDim.x;
			int totalInQuad = (quadEndX - quadStartX + 1)*(quadEndY - quadStartY + 1);
			while (atomicCAS(mutex + quadNo, 0, 1) != 0);  //lock
			averageRed[quadNo] += (float)sumRed[0] / totalInQuad;
			averageGreen[quadNo] += (float)sumGreen[0] / totalInQuad;
			averageBlue[quadNo] += (float)sumBlue[0] / totalInQuad;
			atomicExch(mutex + quadNo, 0);  //unlock
		}
	}
}

__global__ void getScoreAndPaintKernel(const int *aRed, const int *aGreen, const int *aBlue, int *cRed, int *cGreen, int *cBlue, int *mutex, float* averageRed, float* averageGreen, float* averageBlue, float* scoreRed, float* scoreGreen, float* scoreBlue, int quadNo, int quadStartX, int quadStartY, int quadEndX, int quadEndY) {

	int threadAbsX = (threadIdx.x + blockDim.x*blockIdx.x) + quadStartX;
	int threadAbsY = (threadIdx.y + blockDim.y*blockIdx.y) + quadStartY;

	if (threadAbsX <= quadEndX && threadAbsY <= quadEndY) {
		__shared__ float errorRed[noOfThreadInBlockBy2];
		__shared__ float errorGreen[noOfThreadInBlockBy2];
		__shared__ float errorBlue[noOfThreadInBlockBy2];
		int index = getAbsoluteIndex(threadIdx.x, threadIdx.y, quadStartX, quadStartY);
		int threadIndex = threadIdx.x + threadIdx.y*blockDim.x;//local 1d index into the block
		int localKernelNoOfThreadsHalf = (blockDim.x * blockDim.y) / 2;
		unsigned int i = localKernelNoOfThreadsHalf;
		int avgGreen = averageGreen[quadNo];
		cGreen[index] = avgGreen;
		int avgBlue = averageBlue[quadNo];
		cBlue[index] = avgBlue;
		int avgRed = averageRed[quadNo];
		cRed[index] = avgRed;
		while (i != 0) {
			if (threadIndex < i) {
				if (i == localKernelNoOfThreadsHalf) {//first iteration

					errorGreen[threadIndex] = pow((float)aGreen[index] - avgGreen, (float)2);
					errorBlue[threadIndex] = pow((float)aBlue[index] - avgBlue, (float)2);
					errorRed[threadIndex] = pow((float)aRed[index] - avgRed, (float)2);

					int x2 = threadIdx.x;// + i) % blockDim.x;
					int y2 = threadIdx.y + (i / blockDim.x);
					int threadAbsX2 = (x2 + blockDim.x*blockIdx.x) + quadStartX;
					int threadAbsY2 = (y2 + blockDim.y*blockIdx.y) + quadStartY;
					if (threadAbsX2 <= quadEndX && threadAbsY2 <= quadEndY) {
						int index2 = getAbsoluteIndex(x2, y2, quadStartX, quadStartY);
						errorGreen[threadIndex] += pow((float)aGreen[index2] - avgGreen, (float)2);
						errorBlue[threadIndex] += pow((float)aBlue[index2] - avgBlue, (float)2);
						errorRed[threadIndex] += pow((float)aRed[index2] - avgRed, (float)2);
					}
				}
				else {
					errorGreen[threadIndex] += errorGreen[threadIndex + i];
					errorBlue[threadIndex] += errorBlue[threadIndex + i];
					errorRed[threadIndex] += errorRed[threadIndex + i];
				}

			}

			__syncthreads();
			i /= 2;
		}

		if (threadIndex == 0) {
			int totalInQuad = (quadEndX - quadStartX + 1)*(quadEndY - quadStartY + 1);
			/*if (quadNo == 0)
				printf("errorGreen %f", errorGreen[0]);*/
			while (atomicCAS(mutex + quadNo, 0, 1) != 0);  //lock
			scoreRed[quadNo] += errorRed[0] / totalInQuad;
			scoreGreen[quadNo] += errorGreen[0] / totalInQuad;
			scoreBlue[quadNo] += errorBlue[0] / totalInQuad;
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

					//printf("\nThread global index %d, thread index %d, i %d", threadGlobalIndex, threadIdx.x, i);
					maxValues[threadIdx.x] = globalScores[threadGlobalIndex];
					maxIndex[threadIdx.x] = threadGlobalIndex;

					int threadGlobalIndex2 = threadGlobalIndex + i;
					//printf("\nGlobalIndex2 %d value2 %f", threadGlobalIndex2, globalScores[threadGlobalIndex2]);
					
					if (threadGlobalIndex2 < currentTotalQuads) {
						//printf("\nComparing globalScores[%d](%f) with globalScores[%d](%f)", threadGlobalIndex, globalScores[threadGlobalIndex], threadGlobalIndex2, globalScores[threadGlobalIndex2]);
						if (globalScores[threadGlobalIndex2] > maxValues[threadIdx.x]) {
							//printf("\nValue %f  > %f, index ", globalScores[threadGlobalIndex2], maxValues[threadIdx.x]);
							maxValues[threadIdx.x] = globalScores[threadGlobalIndex2];
							maxIndex[threadIdx.x] = threadGlobalIndex2;
						}
					}
				}
				else {
					//printf("\nComparing values (%f) with (%f) for index %d", maxValues[threadIdx.x], maxValues[threadIdx.x + i], i);
					if (threadGlobalIndex + i < currentTotalQuads){
						if (maxValues[threadIdx.x + i] > maxValues[threadIdx.x]) {
							//printf("\nValue %f  > %f, index ", maxValues[threadIdx.x + i], maxValues[threadIdx.x]);
							//printf("\nComparing m[%d](%f) with globalScores[%d](%f)", threadGlobalIndex, globalScores[threadGlobalIndex], threadGlobalIndex2, globalScores[threadGlobalIndex2]);
							maxValues[threadIdx.x] = maxValues[threadIdx.x + i];
							maxIndex[threadIdx.x] = maxIndex[threadIdx.x + i];
						}
					}
				}
			}

			//printf("\n------------------------------");

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


__global__ void kernelToRuleThemAll(int *cRed, int *cGreen, int *cBlue, const int *aRed, const int *aGreen, const int *aBlue, int *mutex, float* averageRed, float* averageGreen, float* averageBlue, float *scoreRed, float *scoreGreen, float *scoreBlue, float* globalScores, float* maxScore, int* maxScoreIndex) {

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

	int blockDimSize = 32;
	dim3 blockdim = dim3(blockDimSize, blockDimSize);
	int i;
	for (i = 0; i < 20; i++) {
		//printf("\n\n----------------New SPLIT -----------------");
		int quadW = (quads[quadToSplit].endX - quads[quadToSplit].startX + 1) / 2;
		int quadH = (quads[quadToSplit].endY - quads[quadToSplit].startY + 1) / 2;

		blockDimSize = 32;
		if (quadW < blockDimSize)
			blockDimSize = quadW;

		blockdim = dim3(blockDimSize, blockDimSize);

		//set grid dim
		int gridDimSize = quadW / blockDimSize;
		gridDimSize = gridDimSize == 0 ? 1 : gridDimSize;
		printf("\nGrid dim size %d, block size %d, Quad w & height %d %d", gridDimSize, blockDimSize, quadW, quadH);
		dim3 griddim = dim3(gridDimSize, gridDimSize);//will have to use ceil function for non multiple of blockdim

		//reset working variables
		scoreRed[0] = 0;
		scoreRed[1] = 0;
		scoreRed[2] = 0;
		scoreRed[3] = 0;
		scoreGreen[0] = 0;
		scoreGreen[1] = 0;
		scoreGreen[2] = 0;
		scoreGreen[3] = 0;
		scoreBlue[0] = 0;
		scoreBlue[1] = 0;
		scoreBlue[2] = 0;
		scoreBlue[3] = 0;
		averageRed[0] = 0;
		averageRed[1] = 0;
		averageRed[2] = 0;
		averageRed[3] = 0;
		averageGreen[0] = 0;
		averageGreen[1] = 0;
		averageGreen[2] = 0;
		averageGreen[3] = 0;
		averageBlue[0] = 0;
		averageBlue[1] = 0;
		averageBlue[2] = 0;
		averageBlue[3] = 0;
		maxScore[0] = 0;
		maxScoreIndex[0] = 0;

		//QUAD 2
		quads[currentTotalQuads].startX = quads[quadToSplit].startX + quadW; 
		quads[currentTotalQuads].startY = quads[quadToSplit].startY;
		quads[currentTotalQuads].endX = quads[currentTotalQuads].startX + quadW - 1;
		quads[currentTotalQuads].endY = quads[currentTotalQuads].startY + quadH - 1;
		//launch kernels for the quad index 1
		getAverageKernel<<<griddim, blockdim, 0, s1>>>(aRed, aGreen, aBlue, mutex, averageRed, averageGreen, averageBlue, 1, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		getScoreAndPaintKernel<<<griddim, blockdim, 0, s1>>>(aRed, aGreen, aBlue, cRed, cGreen, cBlue, mutex, averageRed, averageGreen, averageBlue, scoreRed, scoreGreen, scoreBlue, 1, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		currentTotalQuads++;

		//QUAD 3
		quads[currentTotalQuads].startX = quads[quadToSplit].startX;
		quads[currentTotalQuads].startY = quads[quadToSplit].startY + quadH;
		quads[currentTotalQuads].endX = quads[currentTotalQuads].startX + quadW - 1;
		quads[currentTotalQuads].endY = quads[currentTotalQuads].startY + quadH - 1;
		//launch kernels for the quad index 2
		getAverageKernel <<<griddim, blockdim, 0, s2 >>>(aRed, aGreen, aBlue, mutex, averageRed, averageGreen, averageBlue, 2, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		getScoreAndPaintKernel <<<griddim, blockdim, 0, s2 >>>(aRed, aGreen, aBlue, cRed, cGreen, cBlue, mutex, averageRed, averageGreen, averageBlue, scoreRed, scoreGreen, scoreBlue, 2, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		currentTotalQuads++;

		//QUAD 4
		quads[currentTotalQuads].startX = quads[quadToSplit].startX + quadW;
		quads[currentTotalQuads].startY = quads[quadToSplit].startY + quadH;
		quads[currentTotalQuads].endX = quads[currentTotalQuads].startX + quadW - 1;
		quads[currentTotalQuads].endY = quads[currentTotalQuads].startY + quadH - 1;
		//launch kernels for the quad index 3
		getAverageKernel <<<griddim, blockdim, 0, s3 >>>(aRed, aGreen, aBlue, mutex, averageRed, averageGreen, averageBlue, 3, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		getScoreAndPaintKernel <<<griddim, blockdim, 0, s3 >>>(aRed, aGreen, aBlue, cRed, cGreen, cBlue, mutex, averageRed, averageGreen, averageBlue, scoreRed, scoreGreen, scoreBlue, 3, quads[currentTotalQuads].startX, quads[currentTotalQuads].startY, quads[currentTotalQuads].endX, quads[currentTotalQuads].endY);
		currentTotalQuads++;

		//QUAD 1
		quads[quadToSplit].endX = quads[quadToSplit].startX + quadW - 1; 
		quads[quadToSplit].endY = quads[quadToSplit].startY + quadH - 1;
		//launch kernels for the quad index 3
		getAverageKernel <<<griddim, blockdim, 0, s4 >>>(aRed, aGreen, aBlue, mutex, averageRed, averageGreen, averageBlue, 0, quads[quadToSplit].startX, quads[quadToSplit].startY, quads[quadToSplit].endX, quads[quadToSplit].endY);
		getScoreAndPaintKernel <<<griddim, blockdim, 0, s4 >>>(aRed, aGreen, aBlue, cRed, cGreen, cBlue, mutex, averageRed, averageGreen, averageBlue, scoreRed, scoreGreen, scoreBlue, 0, quads[quadToSplit].startX, quads[quadToSplit].startY, quads[quadToSplit].endX, quads[quadToSplit].endY);

		cudaDeviceSynchronize();
		//quad 0
		printf("\nRed avg %f", averageRed[0]);
		printf("\nGreen avg %f", averageGreen[0]);
		printf("\nBlue avg %f", averageBlue[0]);
		printf("\nRed error %f", sqrt(scoreRed[0]));
		printf("\nGreen error %f", sqrt(scoreGreen[0]));
		printf("\nBlue error %f", sqrt(scoreBlue[0]));
		globalScores[quadToSplit] = (sqrt(scoreRed[0])*0.3) + (sqrt(scoreGreen[0])*0.6) + (sqrt(scoreBlue[0])*0.11);
		globalScores[quadToSplit] *= powf(quadH*quadW, 0.25);
		//quad 1
		globalScores[currentTotalQuads-3] = (sqrt(scoreRed[1])*0.3) + (sqrt(scoreGreen[1])*0.6) + (sqrt(scoreBlue[1])*0.11);
		globalScores[currentTotalQuads - 3] *= powf(quadH*quadW, 0.25);
		//quad 2
		globalScores[currentTotalQuads-2] = (sqrt(scoreRed[2])*0.3) + (sqrt(scoreGreen[2])*0.6) + (sqrt(scoreBlue[2])*0.11);
		globalScores[currentTotalQuads - 2] *= powf(quadH*quadW, 0.25);
		//quad 3
		globalScores[currentTotalQuads-1] = (sqrt(scoreRed[3])*0.3) + (sqrt(scoreGreen[3])*0.6) + (sqrt(scoreBlue[3])*0.11);
		globalScores[currentTotalQuads - 1] *= powf(quadH*quadW, 0.25);
		//printf("\nScores: %f, %f, %f, %f", globalScores[quadToSplit], globalScores[currentTotalQuads - 3], globalScores[currentTotalQuads - 2], globalScores[currentTotalQuads - 1]);

		int h;
		printf("\nScores ");
		for (h = 0; h < currentTotalQuads; h++) {
			printf("%f ", globalScores[h]);
		}

		//find max
		int maxKernelBlockSize = 1024;// currentTotalQuads > 1024 ? 1024 : currentTotalQuads;
		maxKernelBlockSize = maxKernelBlockSize % 2 == 0 ? maxKernelBlockSize : maxKernelBlockSize + 1;
		int maxKernelGridSize = ceil((float)currentTotalQuads /1024);
		getMaxScoreKernel <<<maxKernelGridSize, maxKernelBlockSize>>> (globalScores, maxScore, maxScoreIndex, currentTotalQuads, mutex);
		cudaDeviceSynchronize();
		printf("\nMax index is %d, max is %f", maxScoreIndex[0], globalScores[maxScoreIndex[0]]);
		//TODO select index with max to split next
		//if (maxScoreIndex[0] != 0) {
		quadToSplit = maxScoreIndex[0];// currentTotalQuads - (4 - maxScoreIndex[0]);
			printf("\nQuadToSplit is %d", quadToSplit);
		//}

		/*printf("\nQuad %d", 0);
		printf("\nAverage is %f", averageGreen[0]);
		printf("\nScore is %f", globalScores[quadToSplit]);
		printf("\n");
		printf("\nQuad %d", 1);
		printf("\nAverage is %f", averageGreen[1]);
		printf("\nScore is %f", globalScores[currentTotalQuads - 3]);
		printf("\n");
		printf("\nQuad %d", 2);
		printf("\nAverage is %f", averageGreen[2]);
		printf("\nScore is %f", globalScores[currentTotalQuads - 2]);
		printf("\n");
		printf("\nQuad %d", 3);
		printf("\nAverage is %f", averageGreen[3]);
		printf("\nScore is %f", globalScores[currentTotalQuads - 1]);*/
		printf("\n");

		/*int k = 0;
		for (k = 0; k < 256; k++) {
			printf("\n%d %d %d", cRed[k], cGreen[k], cBlue[k]);
		}*/
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

	CImg<unsigned char> image("C:\\Projects\\Visual Studio\\quadpic\\quadpic\\mandl.bmp");

	int v, b;
	int count = 0;
	int *indataR = new int[rows * columns];
	int *indataG = new int[rows * columns];
	int *indataB = new int[rows * columns];
	for (v = 0; v < rows; v++) {
		for (b = 0; b < columns; b++) {

			indataR[count] = image(b, v, 0, 0);
			indataG[count] = image(b, v, 0, 1);
			indataB[count] = image(b, v, 0, 2);
			count++;
		}
	}

    const int arraySize = rows*columns;
    /*const int imageData[arraySize] = { 0, 1,	4, 5,
									   2, 3,	6, 7,	
									   
									   8, 9,	12, 13,
									   10, 11,	14, 15,
									};*/

	/*int imageDataR[1024];
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
	}*/


	int *cRed = new int[rows*columns];
	int *cGreen = new int[rows*columns];
	int *cBlue = new int[rows*columns];

    // Add vectors in parallel.
    cudaError_t cudaStatus = generateOutputWithCuda(cRed, cGreen, cBlue, indataR, indataG, indataB, arraySize);
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

	count = 0;
	for (v = 0; v < 256; v++) {
		for (b = 0; b < 256; b++) {
			const unsigned char color[] = { cRed[count], cGreen[count], cBlue[count] };
			//printf("\nCDATA %d %d %d", cRed[count], cGreen[count], cBlue[count]);
			image.draw_point(b, v, 0, color, 1);

			count++;
		}
	}

	image.display("ASD");

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
	float *dev_averageRed = 0;
	float *dev_averageGreen = 0;
	float *dev_averageBlue = 0;
	float *dev_scoreRed = 0;
	float *dev_scoreGreen = 0;
	float *dev_scoreBlue = 0;

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

	cudaStatus = cudaMalloc((void**)&dev_averageRed, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_averageGreen, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_averageBlue, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_scoreRed, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_scoreGreen, 4 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_scoreBlue, 4 * sizeof(float));
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
	kernelToRuleThemAll << <1, 1 >> >(dev_cRed, dev_cGreen, dev_cBlue, dev_aRed, dev_aGreen, dev_aBlue, dev_mutex, dev_averageRed, dev_averageGreen, dev_averageBlue, dev_scoreRed, dev_scoreGreen, dev_scoreBlue, dev_globalScores, dev_maxScore, dev_maxScoreIndex);
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

	cudaStatus = cudaMemcpy(cGreen, dev_cGreen, rows*columns * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(cBlue, dev_cBlue, rows*columns * sizeof(int), cudaMemcpyDeviceToHost);
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
