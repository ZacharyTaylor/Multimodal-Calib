#include "MatlabCalls.h"

//global so that matlab never has to see them
SparseScan** MoveStore;
DenseImage** BaseStore;
Pair** Pairs;

void initalizeScans(size_t numBase, size_t numMove, size_t numPairs){
	
	BaseStore = new DenseImage*[numBase];
	//setting null so can tell if it is allocated
	for(size_t i = 0; i < numBase; i++){
		BaseStore[i] = NULL;
	}

	MoveStore = new SparseScan*[numMove];
	//setting null so can tell if it is allocated
	for(size_t i = 0; i < numMove; i++){
		MoveStore[i] = NULL;
	}

	Pairs = new Pairs*[numPairs];
	//setting null so can tell if it is allocated
	for(size_t i = 0; i < numPairs; i++){
		Pairs[i] = NULL;
	}
}

void setBaseScan(size_t scanNum, size_t height, size_t width, size_t numCh, float* base){
	
	if(BaseStore[scanNum] != NULL){
		TRACE_INFO("Base %i already allocated, clearing and overwritng with new data\n", scanNum);
		delete BaseStore[scanNum];
		BaseStore[scanNum] = NULL;
	}

	BaseStore[scanNum] = new DenseImage(height, width, numCh, base);
}

void setMoveImage(size_t scanNum, size_t height, size_t width, size_t numCh, float* move){
	
	if(BaseStore[scanNum] != NULL){
		TRACE_INFO("Base %i already allocated, clearing and overwritng with new data\n", scanNum);
		delete BaseStore[scanNum];
		BaseStore[scanNum] = NULL;
	}

	BaseStore[scanNum] = new DenseImage(height, width, numCh, base);
}