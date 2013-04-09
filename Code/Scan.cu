#include "Scan.h"
#include "Kernel.h"

Scan::Scan(const size_t numDim, const size_t numCh,  const size_t* dimSize) : 
	numDim_(numDim),
	numCh_(numCh),
	dimSize_(dimSize)
{	
}

Scan::Scan(const size_t numDim, const size_t numCh,  const size_t* dimSize, PointsList* points) : 
	numDim_(numDim),
	numCh_(numCh),
	dimSize_(dimSize),
	points_(points){}

size_t Scan::getNumDim(void){
	return numDim_;
}

size_t Scan::getNumCh(void){
	return numCh_;
}

size_t Scan::getDimSize(size_t i){
	if(i >= numDim_){
		TRACE_ERROR("tried to get size of dimension %i, where only %i dimensions exist",(i+1),numDim_);
		return 0;
	}
	else {
		return dimSize_[i];
	}
}

size_t Scan::getNumPoints(void){
	size_t numPoints = 1;
		
	for( size_t i = 0; i < numDim_; i++ ){
		if(dimSize_[i] != 0){
			numPoints *= dimSize_[i];
		}
	}

	return numPoints;
}
	
PointsList* Scan::getPoints(void){
	return points_;
}

DenseImage::DenseImage(const size_t height, const size_t width, const size_t numCh, TextureList* points): 
	Scan(IMAGE_DIM ,numCh,setDimSize(width, height, numCh),points)
{
	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; 

}

//creates own copy of data
DenseImage::DenseImage(const size_t height, const size_t width, const size_t numCh, float* pointsIn):
	Scan(IMAGE_DIM ,numCh,setDimSize(width, height, numCh),NULL)
{
	TextureList* points = new TextureList(pointsIn, true, height, width, numCh);
	points_ = points;

	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; 
}

size_t* DenseImage::setDimSize(const size_t width, const size_t height, const size_t numCh){
	size_t* out = new size_t[3];
	out[0] = height;
	out[1] = width;
	out[3] = numCh;

	return out;
}

void DenseImage::d_interpolate(SparseScan* scan){
	if(!points_->IsOnGpu()){
		TRACE_WARNING("Dense image not on gpu, loading now");
		points_->AllocateGpu();
		points_->CpuToGpu();
		}
	
	//create texture
	cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float>();
	
	CudaSafeCall(cudaBindTextureToArray(&tex, (cudaArray_t)(points_->GetGpuPointer()), &channelDescCoeff));
	TextureList* texPoints = (TextureList*)points_;

	for(size_t i = 0; i < scan->getNumCh(); i++){

		DenseImageInterpolateKernel<<<gridSize(texPoints->GetHeight() * texPoints->GetWidth()) ,BLOCK_SIZE>>>	
			(texPoints->GetWidth(), texPoints->GetHeight(), (float*)scan->GetLocation()->GetGpuPointer(), (float)i, (float*)scan->getPoints()->GetGpuPointer(), scan->getDimSize(0));
		CudaCheckError();
	}
}

TextureList* DenseImage::getPoints(void){
	return (TextureList*)points_;
}


size_t* SparseScan::setDimSize(const size_t numCh, const size_t numDim, const size_t numPoints){
	size_t* out = new size_t(2);
	out[0] = numPoints;
	out[1] = numCh + numDim;

	return out;
}

size_t SparseScan::getNumPoints(void){
	return dimSize_[0];
}

float* SparseScan::GenLocation(size_t numDim, size_t* dimSize){

	size_t* iter = new size_t[numDim];

	size_t numEntries = 1;
		
	for( size_t i = 0; i < numDim; i++ ){
		iter[i] = 0;
		numEntries *= dimSize[i];
	}

	float* loc = new float[numEntries * numDim];

	size_t j = 0;
	bool run = true;

	//iterate over every point to fill in image locations
	while(run){
	
		for( size_t i = 0; i < numDim; i++ ){
			loc[j + numEntries*i] = (float)iter[i];
		}

		j++;
		iter[0]++;
		for( size_t i = 0; i < numDim; i++ ){
			if(iter[i] == 319){
				iter[i] = 319;
			}
			if(iter[i] >= dimSize[i]){
				if(i != (numDim-1)){
					iter[i+1]++;
				}
				iter[i] = 0;
			}
			else {
				run = true;
				break;
			}
			run = false;
		}
	}

	delete[] iter;

	return loc;
}

SparseScan::SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints): 
	Scan(numDim, numCh, setDimSize(numCh, numDim, numPoints),NULL)
{
	points_ = new PointsList(numPoints * numCh);
	location_ = new PointsList(numPoints * numDim);
}

SparseScan::SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, PointsList* points, PointsList* location): 
	Scan(numDim,numCh,setDimSize(numCh, numDim, numPoints),NULL)
{	
	points_ = points;
	location_ = location;
}

//creates own copies of data
SparseScan::SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, float* pointsIn, float* locationIn): 
	Scan(numDim,numCh,setDimSize(numCh, numDim, numPoints),NULL)
{	
	PointsList* points = new PointsList(pointsIn, numCh*numPoints, true);
	points_ = points;

	PointsList* location = new PointsList(locationIn, numDim*numPoints, true);
	location_ = location;
}

/*SparseScan::SparseScan(Scan in):
Scan(in.getNumDim(), in.getNumCh(),setDimSize(in.getNumCh(), in.getNumPoints())
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

SparseScan::SparseScan(Scan in, PointsList* location):
	Scan(in.getNumDim(), in.getNumCh(),setDimSize(in.getNumDim(), in.getNumCh(), in.getNumPoints()))
{
	points_ = in.getPointsPointer();
	location_ = location;
}*/

PointsList* SparseScan::GetLocation(void){
	return location_;
}