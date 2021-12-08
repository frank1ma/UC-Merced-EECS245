// dynamic schedule 
// openmp -prefxi sum 
#include "stdio.h"
#include <iostream>
#include <omp.h>
#include <time.h>
using namespace std;

#define NUM_ELEMENTS 10000
#define INI_ELEMENTS 1
#define CHUNK_SIZE 20  //  my maximum num of thread is 12. chunk size shall not less than NUM_ELEMENTS/12


int main(void) {
	int x[NUM_ELEMENTS] = { 0 };
	int cp_x[NUM_ELEMENTS] = { 0 };
	int i, k, max_thread_id = 0;
	int NUM_MAX_THREADS = 12;//omp_get_max_threads();
	clock_t t1,t2;

	// Initialization
	for (i = 0; i < NUM_ELEMENTS; i++) {
		x[i] = INI_ELEMENTS;
		cp_x[i] = INI_ELEMENTS;
	}
	t1 = clock();
	// Step 1  Parallel
#pragma omp parallel for num_threads(NUM_MAX_THREADS) schedule(dynamic,CHUNK_SIZE)
	for (i = 0; i < NUM_ELEMENTS; i++) {
		int lid, j, gid;

		lid = i % CHUNK_SIZE;
		gid = omp_get_thread_num();

		if (gid > max_thread_id)
			max_thread_id = gid;

		//cout << lid << " from " << omp_get_thread_num() << endl;  //omp_get_thread_num()
		for (j = 0; j < lid; j++)
		{
			x[i] = x[i] + cp_x[i - j - 1];    // calculate local prefix sum of every chunk
		}

	}
	//for (i = 0; i < NUM_ELEMENTS; i++) {    // show division of chunk and prefix sum
	//	cout << x[i] <<  endl;
	//}	
	//cout << max_thread_id;

	// Step 2  Sequential 
	// cout << "max thread id is " << max_thread_id << endl;
	int num_chunk = NUM_ELEMENTS / CHUNK_SIZE + 1;
	cout << "num of chunk is " << num_chunk << endl;
	int* T = new int[num_chunk];
	int sum = 0;
	T[0] = 0;
	for (k = 1; k < num_chunk; k++) {    // show division of chunk and prefix sum
		sum = T[k - 1];
		T[k] = x[k * CHUNK_SIZE - 1];
		T[k] = T[k] + sum;
		//cout << T[k] <<endl;
	}

	// Step 3 Parallel : every thread adds T[threadid] to all its element
	#pragma omp parallel for schedule(dynamic,CHUNK_SIZE)
		for (i = 0; i < NUM_ELEMENTS; i++) {
		x[i] = x[i] + T[i/CHUNK_SIZE];
		}

	t2 = clock();

	// Step 4 check values of prefix sum
	

	for (i = 0; i < NUM_ELEMENTS; i++) {
		cout << x[i] << " ";
		if ((i + 1) % 10 == 0)
			cout << endl;
	}
	cout << "The prefix sum sequence is above" << endl;
	cout << "Dynamic Schdedule : total time is " << (t2 - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms " << endl;
	cout << "number of elements is " << NUM_ELEMENTS << ". Values are all " << INI_ELEMENTS << " . " << endl;
	cout << "Chunk size is " << CHUNK_SIZE << endl;
	cout << endl;
	cout << "press any key to exit" << endl;
	getchar();

	delete[]T;
	return 0;
}
