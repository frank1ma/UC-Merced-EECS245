#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
int main(int argc, char *argv[]) {
    int world_rank;
    int world_size;
    int token;
    MPI_Comm column_comm1,column_comm2;
    clock_t start_t, end_t, total_t;
    start_t = clock();
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);
    
    MPI_Comm_split (MPI_COMM_WORLD,0,world_rank,&column_comm1);
    MPI_Comm_split (MPI_COMM_WORLD,1,world_rank,&column_comm2);
    // if world_size is less than 2 
    if (world_size < 2) {
        printf("Try a number greater than 2");
        exit(1);
    }
    
    if (world_rank == 0) {
        token = 0;
        MPI_Send(&token, 1, MPI_INT, world_rank+1, 0, column_comm1);
        MPI_Send(&token, 1, MPI_INT, world_rank+1, 0, column_comm2);
        printf("Comm 1 process %d sends to process %d,value = %d\n",world_rank, world_rank+1,token);
        printf("Comm 2 process %d sends to process %d,value = %d\n",world_rank, world_rank+1,token);
        MPI_Recv(&token, 1, MPI_INT, world_size-1, 0, column_comm1, MPI_STATUS_IGNORE);
         MPI_Recv(&token, 1, MPI_INT, world_size-1, 0, column_comm2, MPI_STATUS_IGNORE);
        printf("Comm 1 process %d receives process %d,value = %d\n",world_rank, world_rank+1,token);
        printf("Comm 2 process %d receives process %d,value = %d\n",world_rank, world_rank+1,token);
    } else {
        MPI_Recv(&token, 1, MPI_INT, world_rank-1, 0, column_comm1, MPI_STATUS_IGNORE);
        MPI_Recv(&token, 1, MPI_INT, world_rank-1, 0, column_comm2, MPI_STATUS_IGNORE);
        if (token == world_rank-1)
        printf("Comm 1 process %d receives process %d,value = %d\n",world_rank, world_rank-1,token);
        printf("Comm 2 process %d receives process %d,value = %d\n",world_rank, world_rank-1,token);
        token = world_rank;
        if (world_rank == world_size - 1){
            MPI_Send(&token, 1, MPI_INT, 0, 0, column_comm1);
            MPI_Send(&token, 1, MPI_INT, 0, 0, column_comm2);
            printf("Comm 1 process %d sends to process %d,value= %d\n",world_rank, 0,token);
            printf("Comm 2 process %d sends to process %d,value= %d\n",world_rank, 0,token);
        }
        else{
            MPI_Send(&token, 1, MPI_INT, world_rank+1, 0, column_comm1);
            MPI_Send(&token, 1, MPI_INT, world_rank+1, 0, column_comm2);
            printf("Comm 1 process %d sends to process %d,value= %d\n",world_rank, world_rank+1,token);
            printf("Comm 2 process %d sends to process %d,value= %d\n",world_rank, world_rank+1,token); 
        }
              
    }

    MPI_Finalize();
    end_t = clock();
    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000;
    printf("Total time taken by CPU: %lu\n", total_t  );
    return 0;
}