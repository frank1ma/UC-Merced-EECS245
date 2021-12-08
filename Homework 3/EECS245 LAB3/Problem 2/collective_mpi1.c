#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char * argv[]) {

  int world_rank;
  int i;
  int data,send_data;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  clock_t start_t, end_t, total_t;
  start_t = clock();

  if (world_rank == 0) {
    data = 10;
    send_data = data / 2;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&send_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }
  else {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&send_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    data=send_data;
  } 
  printf("data on process %d is %d\n",world_rank,data);
  MPI_Finalize();
  end_t = clock();
  total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000;
  printf("Total time taken by CPU: %lu\n", total_t  );
  return 0;
}