#ifndef IMAGELIST_H
#define IMAGELIST_H

#include "common.h"

//! holds the sensors images
class ImageList {
private:
	//! vector holding all the images data
	thrust::device_vector<float> imageD;
	
	//! vector holding the index of images in the data vector
	thrust::device_vector<size_t> imageIdx;
	//! vector holding the index of transforms to apply to project scan onto each image
	thrust::device_vector<size_t> tformIdx;
	//! vector holding the index of which scan is assosiated with which image
	thrust::device_vector<size_t> scanIdx;

public:

	const size_t height_;
	const size_t width_;
	const size_t depth_;

	//! Constructor creates an empty scan
	ImageList(size_t height, size_t width, size_t depth);

	//! Destructor
	~ImageList(void);

	//! Gets the height of the images
	size_t getHeight(void);
	
	//! Gets the width of the images
	size_t getWidth(void);

	//! Gets the depth of the images
	size_t getDepth(void);
	
	//! Gets the number of images stored
	size_t getNumImages(void);
	
	//! Gets the pointer of the image data array
	float* getIP(void);

	//! Gets the pointer of images index
	size_t* getIdxP(void);

	//! Adds an image to the list
	/*! \param imageIn input image
	*/
	void addImage(thrust::device_vector<float> imageDIn);

	//! Adds an image to the list
	/*! \param imageIn input image
	*/
	void addImage(thrust::host_vector<float> imageDIn);

	//! Removes an image from the list
	/*! \param idx index of image to remove
	*/
	void removeImage(size_t idx);

	//! Removes the last image on the list
	void removeLastImage();

	//! Removes all of the images in the list
	void removeAllImages();
};

#endif //IMAGELIST_H
