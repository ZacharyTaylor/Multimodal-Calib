#include "ImageList.h"

ImageList::ImageList(void){}

ImageList::~ImageList(void){}

size_t ImageList::getHeight(size_t idx){
	if(imageD.size() > idx){
		std::ostringstream err; err << "Cannot get height of element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	return imageD[idx].height;
}
	
size_t ImageList::getWidth(size_t idx){
	if(imageD.size() > idx){
		std::ostringstream err; err << "Cannot get width of element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	return imageD[idx].width;
}

size_t ImageList::getDepth(size_t idx){
	if(imageD.size() > idx){
		std::ostringstream err; err << "Cannot get depth of element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return 0;
	}

	return imageD[idx].depth;
}
	
size_t ImageList::getNumImages(void){
	return imageD.size();
}
	
float* ImageList::getIP(size_t idx){
	if(imageD.size() > idx){
		std::ostringstream err; err << "Cannot get pointer to element " << idx << " as only " << imageD.size() << " elements exist";
		mexErrMsgTxt(err.str().c_str());
		return NULL;
	}
	return thrust::raw_pointer_cast(&imageD[idx].image[0]);
}

void ImageList::addImage(thrust::device_vector<float> imageDIn, size_t height, size_t width, size_t depth){
	image input;
	imageD.push_back(input);
	imageD.back().depth = depth;
	imageD.back().height = height;
	imageD.back().width = width;
	imageD.back().image = imageDIn;
}


void ImageList::addImage(thrust::host_vector<float> imageDIn, size_t height, size_t width, size_t depth){
	image input;
	imageD.push_back(input);
	imageD.back().depth = depth;
	imageD.back().height = height;
	imageD.back().width = width;
	imageD.back().image = imageDIn;
}

void ImageList::removeImage(size_t idx){
	if(imageD.size() > idx){
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