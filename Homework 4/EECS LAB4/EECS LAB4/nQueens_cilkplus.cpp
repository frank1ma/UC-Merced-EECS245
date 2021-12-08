// Shangjie Ma
// EECS245 N-Queens Problem Cilk Plus
// 2019-10-29
#include <iostream>
#include <cilk/cilk.h>
#include <sys/time.h>
#include <math.h>
#include <cilk/reducer_opadd.h>

using namespace std;

//  Check validation
bool CheckPlaceQueen(int *pQueen,int prow,int pcol)
{
	// (prow,pcol) is the posiiton where you want to place Next Queen
	// pQueen is the pointer for array of placed Queens.
	// indice of this array represents the row number placed.
	// value of each element represents the col number placed.
	// The goal is check if the next Queen is placable at (prow,pcol)

	if(prow==1)
		return true;
	for(int i=0;i<prow-1;i++)
	{
		if(pQueen[i]==pcol || (pQueen[i]+prow-i-1==pcol) || (pQueen[i]-prow+i+1==pcol))
			return false;
	}
	return true;
}

// recursive core for placement
bool SolvePlaceQueens(int *pQueen,int n,int prow,int *pcount,int *pswitch)
{
	if(prow > n){
		*pcount +=1 ;
		if (*pswitch ==1){
			cout <<"One solution is given by:" << endl;
			for(int i=0;i<n;i++){
    			for(int j=1;j<n+1;j++){
       				if(j==pQueen[i])
       					cout<<"1";
       				else
       					cout <<"0";
       			}
       		cout <<endl;
			}
		}
	    *pswitch = 0;
		return true;
	}
	
	int pcol=1;     // start from the first column
	bool presult;

	while(pcol <(n+1)){      
		presult=CheckPlaceQueen(pQueen,prow,pcol);  // check each col placable or not

		//show detailed results 
		//cout<<"row no."<<prow<<","<<"col no."<<pcol<<" is "<<presult<<endl;

		if(presult)                                 // if you can place queen at this row
		{
			pQueen[prow-1]=pcol;                     // place it
			SolvePlaceQueens(pQueen,n,prow+1,pcount,pswitch); // move to next row 
		}
		pcol+=1;                                    // no, then move to next column
	}
    return false;

}



int main()
{
    int nQueen;
    bool result=false;
    bool result2=false;
    cilk::reducer_opadd<int> sum;
    clock_t tstart;
    clock_t tend;

    tstart = clock();
    // msg
    cout<<"This program is to show number of solutions of N-Queens Problem" <<endl;
	cout<<"The chessboard will be NxN.Please enter the number of Queens(N)"<<endl;
	cin >> nQueen;

	// number of solutions
    int final_count;
    int pswitch = 1;

    if(nQueen==1){
    	final_count = 1;
    }
    else{
        // parallel loop + reducer
		cilk_for(int i=0;i<nQueen;i++){
			int *pQueen = new int[nQueen];
			int count = 0;
	    	pQueen[0]=i+1;
        	SolvePlaceQueens(pQueen,nQueen,2,&count,&pswitch);
        	sum +=count;
        	delete[] pQueen;
 		}
 		final_count = sum.get_value();
    }
    tend = clock();
 	cout <<nQueen<< " Queens Problem. Nice!" <<endl;
    cout << "The total number of solution is :" << final_count <<endl;
    cout<< "CPU CLOCK TIME is " <<(double)(tend-tstart)/CLOCKS_PER_SEC << " seconds" <<endl;


	return 0;
}