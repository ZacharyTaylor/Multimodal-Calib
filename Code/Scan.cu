#include "Scan.h"

Scan::Scan(const size_t numDim, const size_t numCh,  const size_t* dimSize) : 
	numDim_(numDim),
	numCh_(numCh),
	dimSize_(dimSize)
{
	points_ = new PointsList(getNumPoints());		
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
		TRACE_ERROR("tried to get size of dimension %i, where only %i dimensions exist\n",(i+1),numDim_);
		return 0;
	}
	else {
		return dimSize_[i];
	}
}

size_t Scan::getNumPoints(void){
	size_t numPoints = numCh_;
		
	for( size_t i = 0; i < numDim_; i++ ){
		numPoints *= dimSize_[i];
	}

	return numPoints;
}
	
PointsList* Scan::getPoints(void){
	return points_;
}

//dense scan points stored in a little endien (changing first dimension first) grid
DenseImage::DenseImage(const size_t height, const size_t width, const size_t numCh): 
	Scan(IMAGE_DIM ,numCh,setDimSize(width, height, numCh))
{
	points_ = new TextureList(height, width, numCh);

	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; 
}

DenseImage::DenseImage(const size_t height, const size_t width, const size_t numCh, TextureList* points): 
	Scan(IMAGE_DIM ,numCh,setDimSize(width, height, numCh))
{
	points_ = points;

	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; 
}

DenseImage::DenseImage(const size_t height, const size_t width, const size_t numCh, float* pointsIn):
	Scan(IMAGE_DIM ,numCh,setDimSize(width, height, numCh))
{
	TextureList* points = new TextureList(pointsIn, height, width, numCh);
	points_ = points;

	tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; 
}

DenseImage::~DenseImage(void){
	delete points_;
}

size_t* DenseImage::setDimSize(const size_t width, const size_t height, const size_t numCh){
	size_t* out = new size_t[3];
	out[0] = height;
	out[1] = width;
	out[3] = numCh;

	return out;
}

void DenseImage::d_interpolate(SparseScan* scan){
	//create texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(float),0,0,0,cudaChannelFormatKindFloat);
	
	for(size_t i = 0; i < scan->getNumCh(); i++){

		cudaBindTextureToArray(&tex, ((cudaArray**)(points_->GetGpuPointer()))[i], &channelDesc);

		TextureList* texPoints = (TextureList*)points_;

		DenseImageInterpolateKernel<<<gridSize(texPoints->GetHeight() * texPoints->GetWidth()) ,BLOCK_SIZE>>>
			(texPoints->GetWidth(), texPoints->GetHeight(), (float*)scan->GetLocation()->GetGpuPointer(), (float*)scan->getPoints()->GetGpuPointer(), scan->getDimSize(0));
	}
}



size_t* SparseScan::setDimSize(const size_t numCh, const size_t numPoints){
	size_t* out = new size_t(2);
	out[0] = numPoints;
	out[1] = numCh;

	return out;
}

void SparseScan::GenLocation(void){

	size_t* iter = new size_t[numDim_];

	size_t numEntries = 1;
		
	for( size_t i = 0; i < numDim_; i++ ){
		iter[i] = 0;
		numEntries *= dimSize_[i];
	}

	float* loc = new float[numEntries * numDim_];

	size_t j = 0;
	bool run = true;

	//iterate over every point to fill in image locations
	while(run){
	
		for( size_t i = 0; i < numDim_; i++ ){
			loc[j + numEntries] = (float)iter[i];
		}

		j++;
		iter[0]++;
		for( size_t i = 0; i < numDim_; i++ ){
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

	location_ = new PointsList(loc, numEntries * numDim_);
}

SparseScan::SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints): 
	Scan(numDim, numCh, setDimSize(numCh, numPoints))
{
	points_ = new PointsList(numPoints * numCh);
	location_ = new PointsList(numPoints * numDim);
}

SparseScan::SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, PointsList* points, PointsList* location): 
	Scan(numDim,numCh,setDimSize(numCh,numPoints))
{	
	points_ = points;
	location_ = location;
}

SparseScan::SparseScan(const size_t numDim, const size_t numCh,  const size_t numPoints, PointsList* points): 
	Scan(numDim,numCh,setDimSize(numCh,numPoints))
{	
	points_ = points;
	GenLocation();
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