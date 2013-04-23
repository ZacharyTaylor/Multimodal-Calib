#include "Points.h"
#include "Kernel.h"

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

	delete[] points_;
	points_ = NULL;
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

TextureList::~TextureList(){
	if(IsOnGpu()){
		ClearGpu();
	}
}

void TextureList::AllocateGpu(void){
	if(d_points_ != NULL){
		TRACE_WARNING("d_points_ already full, clearing and overwriting");
		ClearGpu();
	}
	CudaSafeCall(cudaMalloc((void**)&(d_points_), sizeof(float)*width_*height_*depth_));
}

void TextureList::GpuToCpu(void){
	if(IsOnGpu()){

		if(texInMem_){
			for(size_t i = 0; i < depth_; i++){
				CudaSafeCall(cudaMemcpyFromArray(&(((float*)points_)[width_*height_*i]),((cudaArray**)d_points_)[i],0,0,width_*height_*sizeof(float), cudaMemcpyDeviceToHost));
			}
		}
		else {
			CudaSafeCall(cudaMemcpy(points_,((cudaArray**)d_points_),sizeof(float)*width_*height_*depth_,cudaMemcpyDeviceToHost));
		}
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

	cudaMemcpy3DParms copyParams = {0};
	copyParams.kind = cudaMemcpyHostToDevice;

	if(texInMem_){
		for(size_t i = 0; i < depth_; i++){
			CudaSafeCall(cudaMemcpyToArray(((cudaArray**)d_points_)[i], 0, 0, &(((float*)points_)[width_*height_*i]), width_*height_*sizeof(float), cudaMemcpyHostToDevice));
		}
	}
	else {
		CudaSafeCall(cudaMemcpy(d_points_,points_,sizeof(float)*width_*height_*depth_,cudaMemcpyHostToDevice));
	}
}

void TextureList::ClearGpu(void){
	if(IsOnGpu()){
		
		if(texInMem_){
			for(size_t i = 0; i < depth_; i++){
				cudaFreeArray(((cudaArray**)d_points_)[i]);
			}
		}
		else {
			CudaSafeCall(cudaFree(d_points_));
		}
		d_points_ = NULL;
		texInMem_ = false;
	}
	else{
		TRACE_WARNING("nothing on gpu to clear");
	}
}

TextureList::TextureList(float* points, bool copy, const size_t width, const size_t height, const size_t depth):
	PointsList(points, height*width*depth, copy),
	height_(height),
	width_(width),
	depth_(depth),
	texInMem_(false){
	AllocateGpu();
	CpuToGpu();
	//PrefilterArray();
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

	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	
	cudaArray** temp = new cudaArray*[depth_];
	
	for(size_t i = 0; i < depth_; i++){
		CudaSafeCall(cudaMallocArray(&temp[i], &channelDesc, width_, height_));
		CudaSafeCall(cudaMemcpyToArray(temp[i], 0, 0, &(((float*)d_points_)[width_*height_*i]), width_*height_*sizeof(float), cudaMemcpyDeviceToDevice));
	}

	//stores texture
	ClearGpu();
	CudaCheckError();
	texInMem_ = true;
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
	//for(size_t i = 0; i < depth_; i++){
	//	float* ptr = (float*)((((cudaPitchedPtr*)d_points_)[i]).ptr);
		//RunBSplineKernel(ptr, width_,height_);
	//}


	GpuToCpu();
}