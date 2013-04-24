#include "Scan.h"
#include "Kernel.h"
#include "CI\code\memcpy.cu"

Scan::Scan(size_t numDim, size_t numCh,  size_t* dimSize) : 
	numDim_(numDim),
	numCh_(numCh),
	dimSize_(dimSize)
{	
}

Scan::Scan(size_t numDim, size_t numCh,  size_t* dimSize, PointsList* points) : 
	numDim_(numDim),
	numCh_(numCh),
	dimSize_(dimSize),
	points_(points){}

Scan::~Scan(void){
	delete[] dimSize_;
	dimSize_ = NULL;
	delete points_;
	points_ = NULL;
}

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

void Scan::setPoints(PointsList* points){
	points_ = points;
}

DenseImage::DenseImage(const size_t height, const size_t width, const size_t numCh, TextureList* points): 
	Scan(IMAGE_DIM ,numCh,setDimSize(width, height, numCh),points)
{
	/*texRef.addressMode[0] = cudaAddressModeWrap;
	texRef.addressMode[1] = cudaAddressModeWrap;
	texRef.addressMode[2] = cudaAddressModeWrap;
	texRef.filterMode = cudaFilterModeLinear;
	texRef.normalized = false;*/ 

}

//creates own copy of data
DenseImage::DenseImage(const size_t width, const size_t height, const size_t numCh, float* pointsIn):
	Scan(IMAGE_DIM ,numCh,setDimSize(width, height, numCh),NULL)
{
	TextureList* points = new TextureList(pointsIn, true, width, height, numCh);
	points_ = points;

	/*tex.addressMode[0] = cudaAddressModeWrap;
	tex.addressMode[1] = cudaAddressModeWrap;
	tex.addressMode[2] = cudaAddressModeWrap;
	tex.filterMode = cudaFilterModeLinear;
	tex.normalized = false; */
}

DenseImage::~DenseImage(void){
	delete (TextureList*)points_;
	points_ = NULL;
}

size_t* DenseImage::setDimSize(const size_t width, const size_t height, const size_t numCh){
	size_t* out = new size_t[3];
	out[0] = width;
	out[1] = height;
	out[2] = numCh;

	return out;
}

/*__global__ void transformKernel(float *outputData, int width, int height, float theta)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = x / (float) width;
    float v = y / (float) height;

    // transform coordinates
    u -= 0.5f;
    v -= 0.5f;
    float tu = u*cosf(theta) - v*sinf(theta) + 0.5f;
    float tv = v*cosf(theta) + u*sinf(theta) + 0.5f;

    // read from texture and write to global memory
    outputData[y*width + x] = tex2D(tex, tu, tv);
}*/
__global__ void DenseImageInterpolateKernel(const size_t width, const size_t height, const size_t depth, const float* locIn, float* valsOut, const size_t numPoints){
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i >= numPoints){
		return;
	}

	float2 loc;
	loc.x = (locIn[i]+0.5f) / ((float)width);
	loc.y = (locIn[i + numPoints]+0.5f) / ((float)height);

	bool inside =
		0 < loc.x && loc.x < 1 &&
		0 < loc.y && loc.y < 1;

	if (!inside){
		valsOut[i + numPoints*depth] = 0.0f;
	}
	else{
		valsOut[i + numPoints*depth] = tex2D(tex, loc.x,loc.y);
	}
}

void DenseImage::d_interpolate(SparseScan* scan){
	if(!getPoints()->IsOnGpu()){
		TRACE_WARNING("Dense image not on gpu, loading now");
		getPoints()->AllocateGpu();
		getPoints()->CpuToGpu();
	}

	size_t width = this->getPoints()->GetWidth();
	size_t height = this->getPoints()->GetHeight();
	size_t size = width*height*sizeof(float);
	size_t numPoints = scan->getNumPoints();
   
	// Allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    for(size_t i = 0; i < this->getPoints()->GetDepth(); i++){
		// Bind the array to the texture
		cudaArray *cuArray = ((cudaArray**)(this->getPoints()->GetGpuPointer()))[i];
		CudaSafeCall(cudaBindTextureToArray(tex, cuArray, channelDesc));

		DenseImageInterpolateKernel<<<gridSize(numPoints), BLOCK_SIZE>>>(width, height, i,(float*)scan->GetLocation()->GetGpuPointer(), (float*)scan->getPoints()->GetGpuPointer(), numPoints);
		CudaCheckError();
	}
}

TextureList* DenseImage::getPoints(void){
	return (TextureList*)points_;
}


size_t* SparseScan::setDimSize(const size_t numCh, const size_t numDim, const size_t numPoints){
	size_t* out = new size_t[2];
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
	if(numDim != 0){
		points_ = new PointsList(numPoints * numCh);
	}
	else{
		points_ = NULL;
	}

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

SparseScan::~SparseScan(void){
	delete location_;
	location_ = NULL;
}

void SparseScan::changeNumCh(size_t numCh){
		numCh_ = numCh;
		dimSize_[1] = numCh_+numDim_;
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