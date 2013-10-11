#include "ImageList.h"
#include "ScanList.h"
#include "Kernels.h"

ImageList::ImageList(void){}

ImageList::~ImageList(void){}

size_t ImageList::getHeight(size_t idx){
	if(imageD.size() <= idx){
		std::ostringstream err; err << "Cannot get height of element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	return imageD[idx].height;
}
	
size_t ImageList::getWidth(size_t idx){
	if(imageD.size() <= idx){
		std::ostringstream err; err << "Cannot get width of element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	return imageD[idx].width;
}

size_t ImageList::getDepth(size_t idx){
	if(imageD.size() <= idx){
		std::ostringstream err; err << "Cannot get depth of element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	return imageD[idx].depth;
}
	
size_t ImageList::getNumImages(void){
	return imageD.size();
}
	
float* ImageList::getIP(size_t idx, size_t depthIdx){
	if(imageD.size() <= idx){
		std::ostringstream err; err << "Cannot get pointer to element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	if(imageD[idx].depth <= depthIdx){
		std::ostringstream err; err << "Cannot get depth " << depthIdx << " as " << imageD[idx].depth << " is the images maximum depth";
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	return thrust::raw_pointer_cast(&imageD[idx].image[depthIdx*imageD[idx].height*imageD[idx].width]);
}

void ImageList::addImage(thrust::device_vector<float>& imageDIn, size_t height, size_t width, size_t depth){
	image input;
	imageD.push_back(input);
	imageD.back().depth = depth;
	imageD.back().height = height;
	imageD.back().width = width;
	imageD.back().image = imageDIn;
}


void ImageList::addImage(thrust::host_vector<float>& imageDIn, size_t height, size_t width, size_t depth){
	image input;
	imageD.push_back(input);
	imageD.back().depth = depth;
	imageD.back().height = height;
	imageD.back().width = width;
	imageD.back().image = imageDIn;
}

void ImageList::removeImage(size_t idx){
	if(imageD.size() <= idx){
		std::ostringstream err; err << "Cannot erase element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return;
	}
	imageD.erase(imageD.begin() + idx);
}

void ImageList::removeLastImage(){
	imageD.pop_back();
}

void ImageList::removeAllImages(){
	imageD.clear();
};

void ImageList::interpolateImage(size_t imageIdx, size_t scanIdx, std::vector<float*>& interLocs, std::vector<float*>& interVals, size_t numPoints, boolean linear, cudaStream_t stream){
	
	for(size_t i = 0; i < getDepth(imageIdx); i++){
		if(linear){
			LinearInterpolateKernel<<<gridSize(numPoints), BLOCK_SIZE, 0, stream>>>(
				getIP(imageIdx, i),
				interVals[i],
				getHeight(imageIdx),
				getWidth(imageIdx),
				interLocs[0],
				interLocs[1],
				numPoints);
		}
		else{
			NearNeighKernel<<<gridSize(numPoints), BLOCK_SIZE, 0, stream>>>(
				getIP(imageIdx, i),
				interVals[i],
				getHeight(imageIdx),
				getWidth(imageIdx),
				interLocs[0],
				interLocs[1],
				numPoints);
		}
	}
}