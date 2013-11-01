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
	
thrust::device_vector<float> ImageList::getImage(size_t idx){
	return imageD[idx].image;
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

void ImageList::interpolateImage(ScanList* scans, GenList* gen, size_t imageIdx, size_t scanIdx, size_t genIdx, boolean linear){

	for(size_t i = 0; i < getDepth(imageIdx); i++){
		if(linear){
			
			NearNeighKernel<<<gridSize(scans->getNumPoints(scanIdx)), BLOCK_SIZE, 0, gen->getStream(genIdx)>>>(
				getIP(imageIdx, i),
				gen->getGIP(genIdx,i,scans->getNumPoints(scanIdx)),
				getHeight(imageIdx),
				getWidth(imageIdx),
				gen->getGLP(genIdx,0,scans->getNumPoints(scanIdx)),
				gen->getGLP(genIdx,1,scans->getNumPoints(scanIdx)),
				scans->getNumPoints(scanIdx));
		}
		else{
			NearNeighKernel<<<gridSize(scans->getNumPoints(scanIdx)), BLOCK_SIZE, 0, gen->getStream(genIdx)>>>(
				getIP(imageIdx, i),
				gen->getGIP(genIdx,i,scans->getNumPoints(scanIdx)),
				getHeight(imageIdx),
				getWidth(imageIdx),
				gen->getGLP(genIdx,0,scans->getNumPoints(scanIdx)),
				gen->getGLP(genIdx,1,scans->getNumPoints(scanIdx)),
				scans->getNumPoints(scanIdx));
		}
	}
}