#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int world_rank;
    int world_size;
    int token;
    int buff[10];
    int i;
    MPI_Request sreps[10];
    MPI_Request rreps[10];
    MPI_Status  sstatus[10];
    MPI_Status  rstatus[10];
    clock_t start_t, end_t, total_t;
    start_t = clock();
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size (MPI_COMM_WORLD, &world_size);

    // if world_size is less than 2 
    if (world_size < 2) {
        printf("Try a number greater than 2");
        exit(1);
    }
    for(i=0;i<world_size;i++)
        buff[i]=i;

   
    if (world_rank == 0) {
        token = 0;
        buff[0] = token;
        MPI_Isend(&buff[0], 1, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD,&sreps[0]);
        printf("process %d sends to process %d,value = %d\n",world_rank, world_rank+1,(int)buff[0]);
        MPI_Irecv(&buff[world_size-1], 1, MPI_INT, world_size-1, 0, MPI_COMM_WORLD, &rreps[1]);
        printf("process %d receives process %d,value = %d\n",world_rank, world_size-1,(int)buff[world_size-1]);
    } else {
        token = world_rank;
        buff[world_rank] = token;
        //MPI_Wait(&send_req[world_rank-1],&send_status[world_rank-1]);
        MPI_Irecv(&buff[world_rank-1], 1, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, &rreps[world_rank]);
        printf("process %d receives process %d,value = %d\n",world_rank, world_rank-1,(int)buff[world_rank-1]);
        //MPI_Wait(&rev_req[world_rank],&rev_status[world_rank]);
        
        if (world_rank == world_size - 1){
            MPI_Isend(&buff[world_rank], 1, MPI_INT, 0, 0, MPI_COMM_WORLD,&sreps[world_rank]);
            printf("process %d sends to process %d,value= %d\n",world_rank, 0,(int)buff[world_rank]);
        }
        else{
            MPI_Isend(&buff[world_rank], 1, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD,&sreps[world_rank]);
            printf("process %d sends to process %d,value= %d\n",world_rank, world_rank+1,(int)buff[world_rank]); 
            }
              
    }
    MPI_Finalize();
    end_t = clock();
    //total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC * 1000;
    //printf("Total time taken by CPU: %lu\n", total_t  );
    return 0;
}