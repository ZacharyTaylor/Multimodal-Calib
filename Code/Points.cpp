#include "Points.h"

class PointsList {
protected:
	const size_t numEntries_;
	bool onGpu_;

	const float* points_;
	void* d_points_;

public:

	PointsList(float* points, const size_t numEntries):
	  points_(points),
	  numEntries_(numEntries){
	  
	  onGpu_ = false;
	}

	PointsList(const size_t numEntries):
		numEntries_(numEntries),
		points_(new float[numEntries]){}

	~PointsList(){
		if(onGpu_){
			ClearGpu();
		}

		delete [] points_;
	}

	size_t GetNumEntries(){
		return numEntries_;
	}

	float* GetGpuPointer(){
		if(!onGpu_){
			TRACE_WARNING("points were not on GPU, creating gpu pointer first\n");
			AllocateGpu();
		}
		return (float*)d_points_;
	}

	bool GetOnGpu(){
		return onGpu_;
	}
	
	void AllocateGpu(void){
		cudaMalloc((void**)&(d_points_), numEntries_);
		onGpu_ = true;
	}

	void ClearGpu(void){
		if(onGpu_){
			cudaFree(d_points_);
			onGpu_ = false;
		}
		else{
			TRACE_WARNING("nothing on gpu to clear\n");
		}
	}

	void GpuToCpu(void){
		if(onGpu_){
			cudaMemcpy(d_points_, points_, sizeof(float)*numEntries_, cudaMemcpyHostToDevice);
		}
		else {
			TRACE_ERROR("No memory was allocated on gpu, returning\n");
		}
	}

	void CpuToGpu(void){
		if(!onGpu_){
			TRACE_WARNING("No memory was allocated on gpu, allocating now\n");
			AllocateGpu();
		}
		cudaMemcpy(d_points_, points_, sizeof(float)*numEntries_, cudaMemcpyHostToDevice);
	}
}

class TextureList: public PointsList {
protected:
	const size_t height_;
	const size_t width_;
	const size_t depth_;
public:

	TextureList(float* points, const size_t height = 1, const size_t width = 1, const size_t depth = 1):
	  PointsList(points, height*width*depth),
	  height_(height),
	  width_(width),
	  depth_(depth){
	  
	  onGpu_ = false;
	}

	TextureList(const size_t height = 1, const size_t width = 1, const size_t depth = 1):
	  PointsList(height*width*depth),
	  height_(height),
	  width_(width),
	  depth_(depth){
	  
	  onGpu_ = false;
	}

	~TextureList(){
		if(onGpu_){
			ClearGpu();
		}

		delete [] points_;
	}

	size_t GetHeight(void){
		return height_;
	}

	size_t GetWidth(void){
		return width_;
	}

	size_t GetDepth(void){
		return depth_;
	}

	cudaArray* GetGpuPointer(){
		if(!onGpu_){
			TRACE_WARNING("points were not on GPU, creating gpu pointer first\n");
			AllocateGpu();
		}
		return (cudaArray**)d_points_;
	}
	
	void AllocateGpu(void){
		
		d_points_ = new cudaArray*[depth_];

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(float),0,0,0,cudaChannelFormatKindFloat);

		for(int i = 0; i < depth_; i++){
			cudaMallocArray(&(((cudaArray**)d_points_)[i]), &channelDesc, width_, height_);
		}
		onGpu_ = true;
	}

	void ClearGpu(void){
		if(onGpu_){	
			for(int i = 0; i < depth_; i++){
				cudaFreeArray(((cudaArray**)d_points_)[i]);
			}
			onGpu_ = false;
		}
		else{
			TRACE_WARNING("nothing on gpu to clear\n");
		}
	}

	void GpuToCpu(void){
		if(onGpu_){
			for(int i = 0; i < depth_; i++){
				cudaMemcpy2DFromArray((void*)(&points_[i*width_*height_]), sizeof(float), ((cudaArray **)d_points_)[i], 0, 0, width_, height_, cudaMemcpyDeviceToHost);
			}
		}
		else {
			TRACE_ERROR("No memory was allocated on gpu, returning\n");
		}
	}

	void CpuToGpu(void){
		if(!onGpu_){
			TRACE_WARNING("No memory was allocated on gpu, allocating now\n");
			AllocateGpu();
		}

		for(int i = 0; i < depth_; i++){
			cudaMemcpy2DToArray(((cudaArray **)d_points_)[i], 0, 0, &points_[i*width_*height_], sizeof(float), width_, height_, cudaMemcpyHostToDevice);
		}
	}

	void PrefilterTexture(void){
		if(!onGpu_){
			TRACE_WARNING("Gpu must be set up for filtering, allocating memory and copying data now\n");
			AllocateGpu();
			CpuToGpu();
		}
		
		for(int i = 0; i < depth_; i++){
			//inialize texture values
			//CubicBSplinePrefilter2D(d_points_[i], sizeof(float), width_, height_);

			GpuToCpu();
		}
	}
}