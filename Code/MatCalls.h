#ifndef MATLAB_CALLS_H
#define MATLAB_CALLS_H

//must use "extern c" when creating dll so matlab can read it, but matlab can't see "extern c" as it is a c++ term.
#if(_WIN32) 
	#ifdef __cplusplus
		#define DllExport  extern "C" __declspec( dllexport )
	#endif

	#ifndef __cplusplus
		#define DllExport
	#endif
#else
	#ifdef __cplusplus
		#define DllExport  extern "C" 
	#endif

	#ifndef __cplusplus
		#define DllExport
	#endif
#endif

//! Gets if the camera is panoramic
DllExport unsigned int getIfPanoramic(unsigned int idx);
//! Gets number of points in moving scan depth
DllExport unsigned int getNumPoints(unsigned int idx);
//! Gets the number of dimensions in a moving scan
DllExport unsigned int getNumDim(unsigned int idx);
//! Gets the base image depth
DllExport unsigned int getImageDepth(unsigned int idx);
//! Gets the number of channels in a moving scan
DllExport unsigned int getNumCh(unsigned int idx);
//! Gets the number of images
DllExport unsigned int getNumImages(void);
//! Gets the width of an image
DllExport unsigned int getImageWidth(unsigned int idx);
//! Gets the height of an image
DllExport unsigned int getImageHeight(unsigned int idx);

//! Clears the memory allocated for the scans
DllExport void clearScans(void);
//! Clears the memory allocated for the images
DllExport void clearImages(void);
//! Clears the memory allocated for transforms
DllExport void clearTforms(void);
//! Clears the memory allocated for extra parameters of inherited classes
DllExport void clearExtras(void);
//! Clears indices
DllExport void clearIndices(void);
//! Clears everything
DllExport void clearEverything(void);

//! Sets up things ready to perform camera calibration
DllExport void initalizeCamera(void);

//! Sets up things ready to perform image calibration
DllExport void initalizeImage(void);

//! Adds a moving scan ready for calibration
/*! \param moveLIn array holding location of points, treated as an array of size (length,numDim) given in column major order
	\param moveIIn array holding intensity of points, treated as an array of size (length,numCh) given in colum major order
	\param length number of points
	\param numDim dimensionality of points
	\param numCh number of data channels points have
*/
DllExport void addMovingScan(float* moveLIn, float* moveIIn, unsigned int length, unsigned int numDim, unsigned int numCh);

//! Adds a base image ready for calibration
/*! \param baseIn array holding image, dimensions order x,y,z (note needs coverting from matlabs y,x,z order).
	\param height height of image
	\param width width of image
	\param depth depth of image
	\param tformIdx index of the transform that will be applied to project moving scans onto this image
	\param scanIdx index of the scan that will be projected onto this image
*/
DllExport void addBaseImage(float* baseIn, unsigned int height, unsigned int width, unsigned int depth);

//! Adds a transform that will be used to project moving scans onto base images
/*! \param tformIn array holding transform in column major order (y,x)
	\param tformSizeX size of tform in X direction
	\param tformSizeY size of tfrom in Y direction
*/
DllExport void addTform(float* tformIn, unsigned int tformSizeX, unsigned int tformSizeY);

//! Adds a camera for use with cameraTransform, note may crash things if setup for other transform 
/*! \param cIn input camera array in coloum major form
	\param panoramic true if camera is panoramic, false otherwise
*/
DllExport void addCamera(float* cIn, bool panoramic);

//! Adds index of transform that matches to each image
/*! \param index the index of the transform to use for the corrosponding image
	\param length length of index being added
*/
DllExport void addTformIndex(unsigned int* index, unsigned int length);

//! Adds index of scan that matches to each image
/*! \param index the index of the scan to use for the corrosponding image
	\param length length of index being added
*/
DllExport void addScanIndex(unsigned int* index, unsigned int length);

//! Adds index of camera that matches to each image
/*! \param index the index of the camera to use for the corrosponding image
	\param length length of index being added
*/
DllExport void addCameraIndex(unsigned int* index, unsigned int length);

//! Sets up SSD metric
DllExport void setupSSDMetric(void);

//! Sets up GOM metric
DllExport void setupGOMMetric(void);

//! Sets up GOMS metric
DllExport void setupGOMSMetric(void);

//! Sets up MI metric
DllExport void setupMIMetric(void);

//! Sets up NMI metric
DllExport void setupNMIMetric(void);

//!Evalutaes and returns metric
DllExport float evalMetric(void);

//! Outputs the current image of the stored scan
/*! \param image image to output
	\param width width of the image
	\param height height of the image
	\param moveNum the index of the scan to use
	\param dilate number of pixels to dilate each point by
	\param imageColour false to colour using scan intensities, true to colour using image intensities
*/
DllExport void outputImage(float* image, unsigned int width, unsigned int height, unsigned int moveNum, unsigned int dilate, bool imageColour);

//! Outputs a scan coloured by the corrosponding image
/*! \param scan scan to ouput to
	\param moveNum index to use
*/
DllExport void colourScan(float* scan, unsigned int moveNum);

#endif //MATLAB_CALLS_H
