#include "Points.h"

PointsList::PointsList(float* points, const size_t numEntries):
	points_(points),
	numEntries_(numEntries){
	  
	onGpu_ = false;
}

PointsList::PointsList(const size_t numEntries):
	numEntries_(numEntries),
	points_(new float[numEntries]){}

PointsList::~PointsList(){
	if(onGpu_){
		ClearGpu();
	}

	delete [] points_;
}

size_t PointsList::GetNumEntries(){
	return numEntries_;
}

void* PointsList::GetGpuPointer(){
	if(!onGpu_){
		TRACE_WARNING("points were not on GPU, creating gpu pointer first\n");
		AllocateGpu();
	}
	return d_points_;
}

bool PointsList::GetOnGpu(){
	return onGpu_;
}
	
void PointsList::AllocateGpu(void){
	cudaMalloc((void**)&(d_points_), numEntries_);
	onGpu_ = true;
}

void PointsList::ClearGpu(void){
	if(onGpu_){
		cudaFree(d_points_);
		onGpu_ = false;
	}
	else{
		TRACE_WARNING("nothing on gpu to clear\n");
	}
}

void PointsList::GpuToCpu(void){
	if(onGpu_){
		cudaMemcpy(d_points_, points_, sizeof(float)*numEntries_, cudaMemcpyHostToDevice);
	}
	else {
		TRACE_ERROR("No memory was allocated on gpu, returning\n");
	}
}

void PointsList::CpuToGpu(void){
	if(!onGpu_){
		TRACE_WARNING("No memory was allocated on gpu, allocating now\n");
		AllocateGpu();
	}
	cudaMemcpy(d_points_, points_, sizeof(float)*numEntries_, cudaMemcpyHostToDevice);
}


TextureList::TextureList(float* points, const size_t height, const size_t width, const size_t depth):
	PointsList(points, height*width*depth),
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
		TRACE_WARNING("points were not on GPU, creating gpu pointer first\n");
		AllocateGpu();
	}
	return (cudaArray**)d_points_;
}
	
void TextureList::AllocateGpu(void){
		
	d_points_ = new cudaArray*[depth_];

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(float),0,0,0,cudaChannelFormatKindFloat);

	for(size_t i = 0; i < depth_; i++){
		cudaMallocArray(&(((cudaArray**)d_points_)[i]), &channelDesc, width_, height_);
	}
	onGpu_ = true;
}

void TextureList::ClearGpu(void){
	if(onGpu_){	
		for(size_t i = 0; i < depth_; i++){
			cudaFreeArray(((cudaArray**)d_points_)[i]);
		}
		onGpu_ = false;
	}
	else{
		TRACE_WARNING("nothing on gpu to clear\n");
	}
}

void TextureList::GpuToCpu(void){
	if(onGpu_){
		for(size_t i = 0; i < depth_; i++){
			cudaMemcpy2DFromArray((void*)(&points_[i*width_*height_]), sizeof(float), ((cudaArray **)d_points_)[i], 0, 0, width_, height_, cudaMemcpyDeviceToHost);
		}
	}
	else {
		TRACE_ERROR("No memory was allocated on gpu, returning\n");
	}
}

void TextureList::CpuToGpu(void){
	if(!onGpu_){
		TRACE_WARNING("No memory was allocated on gpu, allocating now\n");
		AllocateGpu();
	}

	for(size_t i = 0; i < depth_; i++){
		cudaMemcpy2DToArray(((cudaArray **)d_points_)[i], 0, 0, &points_[i*width_*height_], sizeof(float), width_, height_, cudaMemcpyHostToDevice);
	}
}

void TextureList::PrefilterTexture(void){
	if(!onGpu_){
		TRACE_WARNING("Gpu must be set up for filtering, allocating memory and copying data now\n");
		AllocateGpu();
		CpuToGpu();
	}
		
	for(size_t i = 0; i < depth_; i++){
		//inialize texture values
		//CubicBSplinePrefilter2D(d_points_[i], sizeof(float), width_, height_);

		GpuToCpu();
	}
}