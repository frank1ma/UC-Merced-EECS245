#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char * argv[]) {

  int world_rank;
  int i;
  int data[999];
  int rev_data[333];
  int root_rev_data[333];
  int rev_count = 333;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  clock_t start_t, end_t, total_t;
  start_t = clock();

  if (world_rank == 0) {
    
    for(i=0;i<999;i++){
      data[i]=i;
    }

   MPI_Scatter(&data,333,MPI_INT,&rev_data,rev_count,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Barrier(MPI_COMM_WORLD);
  }
  else {
    MPI_Barrier(MPI_COMM_WORLD);
    for(i=0;i<333;i++){
      rev_data[i]=rev_data[i]+world_rank;
    }
  } 
  MPI_Gather(&rev_data,333,MPI_INT,&root_rev_data,rev_count,MPI_INT,0,MPI_COMM_WORLD);
   if (world_rank == 0) {
    
    for(i=333*world_rank-333;i<333*(world_rank+1);i++){
      data[i]=data[i]+root_rev_data[i];
    }
  }
   for(i=0;i<999;i++){
        printf("data on process %d is %d\n",world_rank,data[i]);
    } 
  MPI_Finalize();
  end_t = clock();
  total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000;
  printf("Total time taken by CPU: %lu\n", total_t  );
  return 0;
}