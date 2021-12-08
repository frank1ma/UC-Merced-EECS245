#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>


#define RADIUS        3
#define BLOCK_SIZE    256
#define NUM_ELEMENTS  (4096*2)

static void handleError(cudaError_t err,const char *file,int line) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        exit( EXIT_FAILURE );
    }
}

#define cudaCheck( err ) (handleError( err, __FILE__, __LINE__ ))

__constant__ int const_arr[NUM_ELEMENTS + 2 * RADIUS];

__global__ void stencil_1d_constant(int *out) {
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int result = 0;
    for(int offset = -RADIUS; offset <= RADIUS; offset++)
    {
         result += const_arr[gindex + RADIUS  + offset];	       
    }
    out[gindex] = result;
}

int main(){
  unsigned int i;
  int h_in[NUM_ELEMENTS + 2 * RADIUS], h_out[NUM_ELEMENTS];
  int  *d_out;
  
  struct timeval tv1, tv0;

  for( i = 0; i < (NUM_ELEMENTS + 2*RADIUS); ++i ) {
    h_in[i] = 1; // With a value of 1 and RADIUS of 3, all output values should be 7
  }

  cudaCheck( cudaMalloc( &d_out, NUM_ELEMENTS * sizeof(int)) );

  cudaMemcpyToSymbol(const_arr, h_in, (NUM_ELEMENTS + 2*RADIUS) * sizeof(int));

  gettimeofday(&tv0, NULL);

  stencil_1d_constant<<< (NUM_ELEMENTS + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE >>> (d_out);
   
  gettimeofday(&tv1, NULL);

  cudaCheck(cudaPeekAtLastError());
  cudaCheck( cudaMemcpy( h_out, d_out, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost) );
  
  printf("time: %lf\n", double(tv1.tv_usec - tv0.tv_usec)/1000000 + (double)(tv1.tv_sec - tv0.tv_sec));
 
  for( i = 0; i < NUM_ELEMENTS; ++i ){
    if (h_out[i] != 7){
      printf("Element h_out[%d] == %d != 7\n", i, h_out[i]);
      break;
    }
  }
  if (i == NUM_ELEMENTS){
    printf("SUCCESS!\n");
   }
 
  //cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}
