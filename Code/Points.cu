#include "Points.h"
#include "Kernel.h"

#include "CI\code\cubicPrefilter3D.cu"
#include "CI\code\cubicTex3D.cu"

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
	d_points_(NULL),
	numEntries_(numEntries){
		TRACE_INFO("%i points set",numEntries);  
}

PointsList::PointsList(const size_t numEntries):
	numEntries_(numEntries),
	d_points_(NULL),
	points_(new float[numEntries]){
		TRACE_INFO("%i points set",numEntries);
	}

PointsList::~PointsList(){
	if(IsOnGpu()){
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
	if(!IsOnGpu()){
		TRACE_WARNING("points were not on GPU, creating gpu pointer first");
		AllocateGpu();
	}
	return d_points_;
}

bool PointsList::IsOnGpu(){
	return (d_points_ != NULL);
}
	
void PointsList::AllocateGpu(void){
	if(d_points_ != NULL){
		TRACE_WARNING("d_points_ already full, clearing and overwriting");
		ClearGpu();
	}
	CudaSafeCall(cudaMalloc((void**)&(d_points_), sizeof(float)*numEntries_));
}

void PointsList::ClearGpu(void){
	if(IsOnGpu()){
		CudaSafeCall(cudaFree(d_points_));
		d_points_ = NULL;
	}
	else{
		TRACE_WARNING("nothing on gpu to clear");
	}
}

void PointsList::GpuToCpu(void){
	if(IsOnGpu()){
		CudaSafeCall(cudaMemcpy(points_, d_points_, numEntries_*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else {
		TRACE_ERROR("No memory was allocated on gpu, returning");
	}
}

void PointsList::CpuToGpu(void){
	if(!IsOnGpu()){
		TRACE_WARNING("No memory was allocated on gpu, allocating now");
		AllocateGpu();
	}
	TRACE_INFO("%i points to be copied from host to device", numEntries_);
	CudaSafeCall(cudaMemcpy(d_points_, points_, sizeof(float)*numEntries_, cudaMemcpyHostToDevice));
}

void TextureList::AllocateGpu(void){
	if(d_points_ != NULL){
		TRACE_WARNING("d_points_ already full, clearing and overwriting");
		ClearGpu();
	}
	const cudaExtent extent = make_cudaExtent(sizeof(float)*width_, height_, depth_);
	d_points_ = new cudaPitchedPtr;
	CudaSafeCall(cudaMalloc3D(((cudaPitchedPtr*)d_points_), extent));
}

void TextureList::GpuToCpu(void){
	if(IsOnGpu()){
		const cudaExtent extent = make_cudaExtent(sizeof(float)*width_, height_, depth_);
		cudaPitchedPtr points = make_cudaPitchedPtr( (void*)points_, sizeof(float), width_*height_*depth_, 1);

		cudaMemcpy3DParms copyParams = {0};
		copyParams.dstPtr = points;
		copyParams.extent = extent;
		copyParams.kind = cudaMemcpyDeviceToHost;

		if(texInMem_){
			copyParams.srcArray = (cudaArray_t)d_points_;
		}
		else {
			copyParams.srcPtr = *((cudaPitchedPtr*)d_points_);
		}

		CudaSafeCall( cudaMemcpy3D(&copyParams) ); 
	}
	else {
		TRACE_ERROR("No memory was allocated on gpu, returning");
	}
}

void TextureList::CpuToGpu(void){
	if(!IsOnGpu()){
		TRACE_WARNING("No memory was allocated on gpu, allocating now");
		AllocateGpu();
	}
	TRACE_INFO("%i points to be copied from host to device", numEntries_);
	const cudaExtent extent = make_cudaExtent(sizeof(float)*width_, height_, depth_);
	
	cudaPitchedPtr points = make_cudaPitchedPtr( (void*)points_, sizeof(float), width_*height_*depth_, 1);

	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr = points;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;

	if(texInMem_){
		copyParams.dstArray = (cudaArray_t)d_points_;
	}
	else {
		copyParams.dstPtr = *((cudaPitchedPtr*)d_points_);
	}

	CudaSafeCall( cudaMemcpy3D(&copyParams) ); 
}

void TextureList::ClearGpu(void){
	if(IsOnGpu()){
		if(texInMem_){
			CudaSafeCall(cudaFreeArray((cudaArray_t)d_points_));
		}
		else {
			CudaSafeCall(((cudaPitchedPtr*)d_points_)->ptr);
			free(d_points_);
		}
		d_points_ = NULL;
		texInMem_ = false;
	}
	else{
		TRACE_WARNING("nothing on gpu to clear");
	}
}

TextureList::TextureList(float* points, bool copy, const size_t height, const size_t width, const size_t depth):
	PointsList(points, height*width*depth, copy),
	height_(height),
	width_(width),
	depth_(depth),
	texInMem_(false){

	AllocateGpu();
	CpuToGpu();
	PrefilterArray();
	ArrayToTexture();
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

void TextureList::ArrayToTexture(void){

	const cudaExtent extent = make_cudaExtent(width_, height_, depth_);

	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float>();
	
	cudaArray* temp;
	
	CudaSafeCall(cudaMalloc3DArray(&temp, &channelDescCoeff, extent));
	//CudaSafeCall(cudaMemcpyToArray(temp, 0, 0, d_points_, (sizeof(float) * width_ * height_ * depth_), cudaMemcpyDeviceToDevice));
	//CudaSafeCall(cudaBindTextureToArray(tex, temp, *channelDescCoeff));
	
	texInMem_ = true;
	tex.normalized = false;  // access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;

	//stores texture
	cudaFree(d_points_);
	d_points_ = temp;
}

void TextureList::PrefilterArray(void){
	if(!IsOnGpu()){
		TRACE_WARNING("Gpu must be set up for filtering, allocating memory and copying data now");
		AllocateGpu();
		CpuToGpu();
	}
		
	//inialize texture values
	//this may have red underlines everywhere but it is right
	CubicBSplinePrefilter3D((float*)d_points_, sizeof(float), width_, height_, depth_);

	GpuToCpu();
}