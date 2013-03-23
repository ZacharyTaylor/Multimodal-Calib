#include "common.h"

class Scan {
protected:
	const size_t numDim_;
	const size_t numCh_;

	const size_t* dimSize_;
	
	PointsList* points_;

public:
	Scan(const size_t numDim, const size_t numCh,  const size_t* dimSize) : 
		numDim_(numDim),
		numCh_(numCh),
		dimSize_(dimSize)
	{
		points_ = new PointsList(getNumPoints());		
	}

	size_t getNumDim(void){
		return numDim_;
	}

	size_t getNumCh(void){
		return numCh_;
	}

	size_t getDimSize(size_t i){
		if(i >= numDim_){
			TRACE_ERROR("tried to get size of dimension %i, where only %i dimensions exist\n",(i+1),numDim_);
			return 0;
		}
		else {
			return dimSize_[i];
		}
	}

	size_t getNumPoints(){
		int i;
		size_t numPoints = numCh_;
		
		for( i = 0; i < numDim_; i++ ){
			numPoints *= dimSize_[i];
		}

		return numPoints;
	}
	
	float* getPointsPointer(){
		if(!points_->GetOnGpu()){
			TRACE_WARNING("points were not on GPU, copying to gpu first\n");
			points_->AllocateGpu();
		}
		return points_->GetGpuPointer();
	}
};

//dense scan points stored in a little endien (changing first dimension first) grid
class DenseScan: public Scan {
public:
	DenseScan(const size_t numDim, const size_t numCh,  const size_t* dimSize): 
		Scan(numDim,numCh,dimSize)
	{
		int i;
		size_t numPoints = 1;
		
		for( i = 0; i < numDim; i++ ){
			numPoints *= dimSize[i];
		}

		points_ = new float(numPoints * numCh);
	}

	DenseScan(const size_t numDim, const size_t numCh,  const size_t* dimSize, float* points): 
		Scan(numDim,numCh,dimSize)
	{
		int i;
		size_t numPoints = 1;
		
		for( i = 0; i < numDim; i++ ){
			numPoints *= dimSize[i];
		}

		points_ = points;
	}

};

//sparse scans have location and intesity
class SparseScan: public Scan {
protected:

	PointsList* location_;

	size_t* setDimSize(const size_t numDim, const size_t numCh, const size_t numPoints){
		size_t* out = new size_t(2);
		out[0] = numPoints;
		out[1] = numCh;
	}

public:
	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints): 
		Scan(numDim,numCh,setDimSize(numDim,numCh,numPoints))
	{
		points_ = new PointsList(numPoints * numCh);
		location_ = new PointsList(numPoints * numDim);
	}

	SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, PointsList* points, PointsList* location): 
		Scan(numDim,numCh,setDimSize(numDim,numCh,numPoints))
	{	
		points_ = points;
		location_ = location;
	}

	SparseScan(DenseScan in):
		Scan(in.getNumDim(), in.getNumCh(),setDimSize(in.getNumDim(), in.getNumCh(), in.getNumPoints()))
	{
		points_ = in.getPointsPointer();

		int i,j;

		size_t* iter = new size_t[numDim_];
		
		for( i = 0; i < numDim_; i++ ){
			iter[i] = 0;
		}

		j = 0;
		bool run = true;

		//iterate over every point to fill in image locations
		while(run){
	
			for( i = 0; i < numDim_; i++ ){
				location_[i + j*numDim_] = iter[i];
			}

			iter[0]++;
			for( i = 0; i < numDim_; i++ ){
				if(iter[i] >= dimSize_[i]){
					iter[i+1]++;
					iter[i] = 0;
				}
				else {
					break;
				}
				run = false;
			}
		}

		delete[] iter;
	}

	SparseScan(DenseScan in, PointsList* location):
		Scan(in.getNumDim(), in.getNumCh(),setDimSize(in.getNumDim(), in.getNumCh(), in.getNumPoints()))
	{
		points_ = in.getPointsPointer();
		location_ = location;
	}

	float* GetLocationPointer(void){
		return location_->GetGpuPointer();
	}

};