#include "MatlabCalls.h"

#include "Points.h"
#include "Scan.h"
#include "Pair.h"

//global so that matlab never has to see them
SparseScan** moveStore = NULL;
unsigned int numMove = 0;

DenseImage** baseStore = NULL;
unsigned int numBase = 0;

Pair** pairs = NULL;
unsigned int numPairs = 0;

unsigned int getNumMove(void){
	return numMove;
}

unsigned int getNumBase(void){
	return numBase;
}

unsigned int getNumPairs(void){
	return numPairs;
}

void clearScans(void){
	if(moveStore != NULL){
		for(unsigned int i = 0; i < numMove; i++){
			delete moveStore[i];
		}
	}	

	if(baseStore != NULL){
		for(unsigned int i = 0; i < numBase; i++){
			delete baseStore[i];
		}
	}	

	if(pairs != NULL){
		for(unsigned int i = 0; i < numPairs; i++){
			delete pairs[i];
		}
	}	
}

void initalizeScans(unsigned int numBaseIn, unsigned int numMoveIn, unsigned int numPairsIn){
	
	if((moveStore != NULL) || (baseStore != NULL) || (pairs != NULL)){
		TRACE_INFO("Scans already initalized, clearing and writing new data\n");
		clearScans();
	}

	numMove = numMoveIn;
	numBase = numBaseIn;
	numPairs = numPairsIn;

	baseStore = new DenseImage*[numBase];
	
	//setting null so can tell if it is allocated
	for(unsigned int i = 0; i < numBase; i++){
		baseStore[i] = NULL;
	}

	moveStore = new SparseScan*[numMove];
	
	//setting null so can tell if it is allocated
	for(unsigned int i = 0; i < numMove; i++){
		moveStore[i] = NULL;
	}

	pairs = new Pair*[numPairs];
	//setting null so can tell if it is allocated
	for(unsigned int i = 0; i < numPairs; i++){
		pairs[i] = NULL;
	}
}

void setBaseImage(unsigned int scanNum, unsigned int height, unsigned int width, unsigned int numCh, float* base){
	
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

void setMoveImage(unsigned int scanNum, unsigned int height, unsigned int width, unsigned int numCh, float* move){
	
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

void setMoveScan(unsigned int scanNum, unsigned int numDim, unsigned int numCh, unsigned int numPoints, float* move){
	
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