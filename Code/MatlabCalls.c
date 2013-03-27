#include "MatlabCalls.h"

//global so that matlab never has to see them
SparseScan** moveStore = NULL;
size_t numMove = 0;

DenseImage** baseStore = NULL;
size_t numBase = 0;

Pair** pairs = NULL;
size_t numPairs = 0;

size_t getNumMove(void){
	return numMove;
}

size_t getNumBase(void){
	return numBase;
}

size_t getNumPairs(void){
	return numPairs;
}

void clearScans(void){
	if(moveStore != NULL){
		for(size_t i = 0; i < numMove; i++){
			delete moveStore[i];
		}
	}	

	if(baseStore != NULL){
		for(size_t i = 0; i < numBase; i++){
			delete baseStore[i];
		}
	}	

	if(pairs != NULL){
		for(size_t i = 0; i < numPairs; i++){
			delete pairs[i];
		}
	}	
}

void initalizeScans(size_t numBaseIn, size_t numMoveIn, size_t numPairsIn){
	
	if((moveStore != NULL) || (baseStore != NULL) || (pairs != NULL)){
		TRACE_INFO("Scans already initalized, clearing and writing new data\n");
		clearScans();
	}

	numMove = numMoveIn;
	numBase = numBaseIn;
	numPairs = numPairsIn;

	baseStore = new DenseImage*[numBase];
	
	//setting null so can tell if it is allocated
	for(size_t i = 0; i < numBase; i++){
		baseStore[i] = NULL;
	}

	moveStore = new SparseScan*[numMove];
	
	//setting null so can tell if it is allocated
	for(size_t i = 0; i < numMove; i++){
		moveStore[i] = NULL;
	}

	pairs = new Pairs*[numPairs];
	//setting null so can tell if it is allocated
	for(size_t i = 0; i < numPairs; i++){
		pairs[i] = NULL;
	}
}

void setBaseImage(size_t scanNum, size_t height, size_t width, size_t numCh, float* base){
	
	if(scanNum >= numBase){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numBase);
		return;
	}
	if(baseStore[scanNum] != NULL){
		TRACE_INFO("Base %i already allocated, clearing and overwritng with new data", scanNum);
		delete baseStore[scanNum];
		baseStore[scanNum] = NULL;
	}

	baseStore[scanNum] = new DenseImage(height, width, numCh, base);
}

void setMoveImage(size_t scanNum, size_t height, size_t width, size_t numCh, float* move){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numMove);
		return;
	}
	if(moveStore[scanNum] != NULL){
		TRACE_INFO("Move %i already allocated, clearing and overwritng with new data", scanNum);
		delete moveStore[scanNum];
		moveStore[scanNum] = NULL;
	}

	moveStore[scanNum] = new SparseScan(IMAGE_DIM,numCh,height*width, move);
}

void setMoveScan(size_t scanNum, size_t numDim, size_t numCh, size_t numPoints, float* move){
	
	if(scanNum >= numMove){
		TRACE_ERROR("Cannot set scan %i as only %i scans exist",scanNum,numMove);
		return;
	}
	if(moveStore[scanNum] != NULL){
		TRACE_INFO("Move %i already allocated, clearing and overwritng with new data", scanNum);
		delete moveStore[scanNum];
		moveStore[scanNum] = NULL;
	}

	moveStore[scanNum] = new SparseScan(numDim,numCh,numPoints,&move[numDim*numPoints],move);
}