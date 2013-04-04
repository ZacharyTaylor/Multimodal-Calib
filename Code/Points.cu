#include "Points.h"

float* PointsList::PointsSetup(float* points, const size_t numEntries, bool copy){
	if(copy){
		float* out = new float[numEntries];
		for(size_t i = 0; i < numEntries; i++){
			out[i] = points[i];
		}
		return out;
	}
	else{
		return points;
	}
}

PointsList::PointsList(float* points, const size_t numEntries, bool copy):
	points_(PointsSetup(points,numEntries,copy)),
	numEntries_(numEntries){
		TRACE_INFO("%i points set",numEntries);  
		onGpu_ = false;
}

PointsList::PointsList(const size_t numEntries):
	numEntries_(numEntries),
	points_(new float[numEntries]){
		TRACE_INFO("%i points set",numEntries);
	}

PointsList::~PointsList(){
	if(onGpu_){
		ClearGpu();
	}

	delete [] points_;
}

size_t PointsList::GetNumEntries(){
	return numEntries_;
}

float* PointsList::GetCpuPointer(){
	return points_;
}

void* PointsList::GetGpuPointer(){
	if(!onGpu_){
		TRACE_WARNING("points were not on GPU, creating gpu pointer first");
		AllocateGpu();
	}
	return d_points_;
}

bool PointsList::GetOnGpu(){
	return onGpu_;
}
	
void PointsList::AllocateGpu(void){
	CudaSafeCall(cudaMalloc((void**)&(d_points_), sizeof(float)*numEntries_));
	onGpu_ = true;
}

void PointsList::ClearGpu(void){
	if(onGpu_){
		CudaSafeCall(cudaFree(d_points_));
		onGpu_ = false;
	}
	else{
		TRACE_WARNING("nothing on gpu to clear");
	}
}

void PointsList::GpuToCpu(void){
	if(onGpu_){
		CudaSafeCall(cudaMemcpy(points_, d_points_, numEntries_*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else {
		TRACE_ERROR("No memory was allocated on gpu, returning");
	}
}

void PointsList::CpuToGpu(void){
	if(!onGpu_){
		TRACE_WARNING("No memory was allocated on gpu, allocating now");
		AllocateGpu();
	}
	TRACE_INFO("%i points to be copied to host from device", numEntries_);
	CudaSafeCall(cudaMemcpy(d_points_, points_, sizeof(float)*numEntries_, cudaMemcpyHostToDevice));
}


TextureList::TextureList(float* points, bool copy, const size_t height, const size_t width, const size_t depth):
	PointsList(points, height*width*depth, copy),
	height_(height),
	width_(width),
	depth_(depth){
	  
	onGpu_ = false;
}

TextureList::TextureList(const size_t height, const size_t width, const size_t depth):
	PointsList(height*width*depth),
	height_(height),
	width_(width),
	depth_(depth){
	  
	onGpu_ = false;
}

TextureList::~TextureList(){
	if(onGpu_){
		ClearGpu();
	}

	delete [] points_;
}

size_t TextureList::GetHeight(void){
	return height_;
}

size_t TextureList::GetWidth(void){
	return width_;
}

size_t TextureList::GetDepth(void){
	return depth_;
}

cudaArray** TextureList::GetGpuPointer(){
	if(!onGpu_){
		TRACE_WARNING("points were not on GPU, creating gpu pointer first");
		AllocateGpu();
	}
	return (cudaArray**)d_points_;
}
	
void TextureList::AllocateGpu(void){
		
	d_points_ = new cudaArray*[depth_];

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(float),0,0,0,cudaChannelFormatKindFloat);

	for(size_t i = 0; i < depth_; i++){
		CudaSafeCall(cudaMallocArray(&(((cudaArray**)d_points_)[i]), &channelDesc, width_, height_));
	}
	onGpu_ = true;
}

void TextureList::ClearGpu(void){
	if(onGpu_){	
		for(size_t i = 0; i < depth_; i++){
			CudaSafeCall(cudaFreeArray(((cudaArray**)d_points_)[i]));
		}
		onGpu_ = false;
	}
	else{
		TRACE_WARNING("nothing on gpu to clear");
	}
}

void TextureList::GpuToCpu(void){
	if(onGpu_){
		for(size_t i = 0; i < depth_; i++){
			CudaSafeCall(cudaMemcpy2DFromArray((void*)(&points_[i*width_*height_]), sizeof(float), ((cudaArray **)d_points_)[i], 0, 0, width_, height_, cudaMemcpyDeviceToHost));
		}
	}
	else {
		TRACE_ERROR("No memory was allocated on gpu, returning");
	}
}

void TextureList::CpuToGpu(void){
	if(!onGpu_){
		TRACE_WARNING("No memory was allocated on gpu, allocating now");
		AllocateGpu();
	}

	for(size_t i = 0; i < depth_; i++){
		CudaSafeCall(cudaMemcpy2DToArray(((cudaArray **)d_points_)[i], 0, 0, &points_[i*width_*height_], sizeof(float), width_, height_, cudaMemcpyHostToDevice));
	}
}

void TextureList::PrefilterTexture(void){
	if(!onGpu_){
		TRACE_WARNING("Gpu must be set up for filtering, allocating memory and copying data now");
		AllocateGpu();
		CpuToGpu();
	}
		
	for(size_t i = 0; i < depth_; i++){
		//inialize texture values
		//CubicBSplinePrefilter2D(d_points_[i], sizeof(float), width_, height_);

		GpuToCpu();
	}
}