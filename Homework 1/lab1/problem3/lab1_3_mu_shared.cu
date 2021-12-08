#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>

 
#define BLOCKSIZE 16
 
typedef struct{
  size_t width;
  size_t height;
  size_t stride;
  float * elements;
 
}matrix_t;
 
__device__ float getElement(const matrix_t * mat, int row, int col){
  return mat->elements[mat->stride * row + col];
}
 
__device__ void setElement(matrix_t * mat, int row, int col, float value){
  mat->elements[mat->stride * row + col] = value;
}
 
__device__ matrix_t getSubMatrix(matrix_t mat, int row, int col){
  matrix_t matAns;
  matAns.width = BLOCKSIZE;
  matAns.height = BLOCKSIZE;
  matAns.stride = mat.stride;
     
        matAns.elements = mat.elements +  row * BLOCKSIZE * mat.stride + col * BLOCKSIZE;
 
	return matAns;
}

__global__ void matMulKernel(matrix_t ma, matrix_t mb, matrix_t mc){
  float cValue = 0;   
  
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
 
  int row = threadIdx.y;
  int col = threadIdx.x;
  for (int subIdx = 0; subIdx < ma.width / BLOCKSIZE; ++subIdx){
      __shared__ float s_subMa[BLOCKSIZE][BLOCKSIZE];
      __shared__ float s_subMb[BLOCKSIZE][BLOCKSIZE];
 
      matrix_t subMatA = getSubMatrix(ma, blockRow, subIdx);
      matrix_t subMatB = getSubMatrix(mb, subIdx, blockCol);
 
      s_subMa[row][col] = getElement(&subMatA, row, col);
      s_subMb[row][col] = getElement(&subMatB, row, col);
 
      __syncthreads();
 
      for (int k = 0; k < BLOCKSIZE; ++k) {
	       cValue += (s_subMa[row][k] * s_subMb[k][col]);
      }
 
      __syncthreads();
    }
 
  matrix_t subMatC = getSubMatrix(mc, blockRow, blockCol);
  setElement(&subMatC, row, col, cValue);
}

void callMatMulKernel(){
  matrix_t matA;
  matA.width = 1024;
  matA.height = 2048;
  matA.stride = matA.width;
  matA.elements = (float *)malloc(matA.width * matA.height * sizeof(float));
 
  matrix_t matB;
  matB.width = 1024;
  matB.height = 1024;
  matB.stride = matB.width;
  matB.elements = (float *)malloc(matB.width * matB.height * sizeof(float));
 
  matrix_t matAns;
  matAns.width = matB.width;
  matAns.height = matA.height;
  matAns.stride = matAns.width;
  matAns.elements = (float *)malloc(matAns.width * matAns.height * sizeof(float));
  memset(matAns.elements,0, matAns.width * matAns.height * sizeof(float));
 
  for (int i = 0; i < matA.width * matA.height; ++i){
    matA.elements[i] = i * 0.1;
  }
  for (int i = 0; i < matB.width * matB.height; ++i){
    matB.elements[i] = i * 0.1;
  }
 
  matrix_t d_matA;
  d_matA.width = matA.width;
  d_matA.height = matA.height;
  d_matA.stride = matA.stride;
    size_t size = d_matA.width *  d_matA.height * sizeof(float);
    cudaMalloc(&d_matA.elements, size);
    cudaMemcpy(d_matA.elements, matA.elements, size, cudaMemcpyHostToDevice);
 
    matrix_t d_matB;
    d_matB.width = matB.width;
    d_matB.height = matB.height;
    d_matB.stride = matB.stride;
    size = d_matB.width * d_matB.height * sizeof(float);
    cudaMalloc(&d_matB.elements, size);
    cudaMemcpy(d_matB.elements, matB.elements, size, cudaMemcpyHostToDevice);
 
    matrix_t d_matC;
    d_matC.width = matAns.width;
    d_matC.height = matAns.height;
    d_matC.stride = matAns.stride;
    cudaMalloc(&d_matC.elements,d_matC.width * d_matC.height * sizeof(float));
 
    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blocksPerGrid(matB.width / threadsPerBlock.x, matA.height / threadsPerBlock.y);
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_matA, d_matB, d_matC);
 
    cudaMemcpy(matAns.elements, d_matC.elements,d_matC.width * d_matC.height * sizeof(float),cudaMemcpyDeviceToHost);
 
    cudaFree(d_matA.elements);
    cudaFree(d_matB.elements);
    cudaFree(d_matC.elements);
 
    free(matA.elements);
    free(matB.elements);
    free(matAns.elements);
    return;
}
 
int main(){
  struct timeval tv1, tv0;
  gettimeofday(&tv0, NULL);
  callMatMulKernel();
  gettimeofday(&tv1, NULL);
  printf("time: %lf\n", double(tv1.tv_usec - tv0.tv_usec)/1000000 + (double)(tv1.tv_sec - tv0.tv_sec));
  return 0;
}
