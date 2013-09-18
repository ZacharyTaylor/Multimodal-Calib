#include "ImageList.h"

ImageList::ImageList(size_t height, size_t width, size_t depth):
		height_(height),
		width_(width),
		depth_(depth){};

ImageList::~ImageList(void){};

size_t ImageList::getHeight(void){
	return height_;
}
	
size_t ImageList::getWidth(void){
	return width_;
}

size_t ImageList::getDepth(void){
	return depth_;
}
	
size_t ImageList::getNumImages(void){
	return imageIdx.size();
}
	
float* ImageList::getIP(void){
	return thrust::raw_pointer_cast(&imageD[0]);
}

size_t* ImageList::getIdxP(void){
	return thrust::raw_pointer_cast(&imageIdx[0]);
}

void ImageList::addImage(thrust::device_vector<float> imageDIn){
	imageD.insert(imageD.end(), imageDIn.begin(), imageDIn.end());

	if(imageIdx.size()){
		imageIdx.push_back(imageIdx.back() + imageDIn.size());
	}
	else{
		imageIdx.push_back(imageDIn.size());
	}
}


void ImageList::addImage(thrust::host_vector<float> imageDIn){
	imageD.insert(imageD.end(), imageDIn.begin(), imageDIn.end());

	if(imageIdx.size()){
		imageIdx.push_back(imageIdx.back() + imageDIn.size());
	}
	else{
		imageIdx.push_back(imageDIn.size());
	}
}

void ImageList::removeImage(size_t idx){

	thrust::device_vector<float>::iterator start, end;
	
	if(idx >= imageIdx.size()){
		return;
	}
	else if((idx+1) != imageIdx.size()){
		end = imageD.begin() + imageIdx[idx+1] - 1;
	}
	else{
		end = imageD.end();
	}

	start = imageD.begin() + imageIdx[idx];
	imageD.erase(start, end);

	size_t size = imageIdx[idx];
	if(idx != 0){
		size -= imageIdx[idx-1];
	}
	for(size_t i = idx+1; i < imageIdx.size(); i++){
		imageIdx[i] -= size;
	}
	imageIdx.erase(imageIdx.begin() + idx);
}

void ImageList::removeLastImage(){
	thrust::device_vector<float>::iterator start;
		
	start = imageD.end() - imageIdx.back();
	imageD.erase(start, imageD.end());

	imageIdx.pop_back();
}

void ImageList::removeAllImages(){
	imageD.clear();
	imageIdx.clear();
};