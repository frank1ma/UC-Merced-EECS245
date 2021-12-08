#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int world_rank;
    int world_size;
    int dim_sizes[2];
    int wrap_around[2];
    int reorder=1;
    int coordinates[2];
    int grid_rank;
    MPI_Comm grid_comm;
    dim_sizes[0]=2;
    dim_sizes[1]=2;
    wrap_around[0]=1;
    wrap_around[1]=1;
    clock_t start_t, end_t, total_t;
    start_t = clock();

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);
    MPI_Cart_create(MPI_COMM_WORLD,2,dim_sizes,wrap_around,reorder,&grid_comm);
    MPI_Cart_coords(grid_comm,world_rank,2,coordinates);
    MPI_Cart_rank(grid_comm,coordinates,&grid_rank);
    
    printf("process %d local rank is %d\n",world_rank,grid_rank);
    printf("process %d average with east is %f\n",world_rank,(double)(grid_rank+grid_rank+1)/2);
    printf("process %d average with west is %f\n",world_rank,(double)(grid_rank+grid_rank-1)/2);
    printf("process %d average with north is %f\n",world_rank,(double)(grid_rank+dim_sizes[1])/2);
    printf("process %d average with south is %f\n",world_rank,(double)(grid_rank-dim_sizes[1])/2);
    MPI_Finalize();
    end_t = clock();
    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000;
    printf("Total time taken by CPU: %lu\n", total_t  );
    return 0;
}