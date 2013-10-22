#ifndef IMAGELIST_H
#define IMAGELIST_H

#include "common.h"
#include "ScanList.h"

//! holds the sensors images
class ImageList {
private:

	//! structre holding infomation about each image
	typedef struct {
		//! vector holding image data
		thrust::device_vector<float> image;
	
		//! image height
		size_t height;
		//! image width
		size_t width;
		//! image depth
		size_t depth;
	} image;

	//! vector holding all the images data
	std::vector<image> imageD;

public:

	//! Constructor
	ImageList(void);

	//! Destructor
	~ImageList(void);

	//! Gets the height of the image
	/*! /param idx index of image
	*/
	size_t getHeight(size_t idx);
	
	//! Gets the width of the image
	/*! /param idx index of image
	*/
	size_t getWidth(size_t idx);

	//! Gets the depth of the image
	/*! /param idx index of image
	*/
	size_t getDepth(size_t idx);
	
	//! Gets the number of images stored
	size_t getNumImages(void);
	
	//! Gets the pointer of the image data array
	/*! /param idx index of image
	*/
	float* getIP(size_t idx, size_t depthIdx);

	//! Adds an image to the list
	/*! \param imageDIn input image data
		\param height height of image
		\param width width of image
	*/
	void addImage(thrust::device_vector<float>& imageDIn, size_t height, size_t width, size_t depth);

	//! Adds an image to the list
	/*! \param imageDIn input image data
		\param height height of image
		\param width width of image
	*/
	void addImage(thrust::host_vector<float>& imageDIn, size_t height, size_t width, size_t depth);

	//! Removes an image from the list
	/*! \param idx index of image to remove
	*/
	void removeImage(size_t idx);

	//! Removes the last image on the list
	void removeLastImage();

	//! Removes all of the images in the list
	void removeAllImages();

	//! Interpolates specified image at given locations
	/*! \param scans list of scans with points to interpolate at
		\param imageIdx the index of the image
		\param scanIdx the index of the scan to use
		\param linear true for linear interpolation, false for nearset neighbour
	*/
	void ImageList::interpolateImage(ScanList* scans, size_t imageIdx, size_t scanIdx, boolean linear);
};

#endif //IMAGELIST_H
