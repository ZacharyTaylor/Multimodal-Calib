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

//! Gets the number of moving scans
DllExport unsigned int getNumMove(void);
//! Gets the number of base scans
DllExport unsigned int getNumBase(void);

//! Clears the memory allocated for the scans
DllExport void clearScans(void);
//! Clears the memory allocated for metrics
DllExport void clearMetric(void);
//! Clears the memory allocated for transforms
DllExport void clearTform(void);
//! Clears the memory allocated for rendering images
DllExport void clearRender(void);

//! Initalizes the Scans
/*!
	\param numBaseIn number of base scans
	\param numMoveIn number of moving scans
*/
DllExport void initalizeScans(unsigned int numBaseIn, unsigned int numMoveIn);

//! Loads an image from matlab to use as a base scan
/*!
	\param scanNum the number of the scan to place the image in
	\param width width of the image
	\param height height of the image
	\param numCh number of colour channels image has
	\param base the image to use as a base scan
*/
DllExport void setBaseImage(unsigned int scanNum, unsigned int width, unsigned int height, unsigned int numCh, float* base);
//! Loads an image from matlab to use as a moving scan
/*!
	\param scanNum the number of the scan to place the image in
	\param width width of the image
	\param height height of the image
	\param numCh number of colour channels image has
	\param move the image to use as a moving scan
*/
DllExport void setMoveImage(unsigned int scanNum, unsigned int width, unsigned int height, unsigned int numCh, float* move);
//! Loads a pointcloud from matlab to use as a moving scan
/*!
	\param scanNum the number of the scan to place the pointcloud in
	\param numDim number of dimensions the scan has
	\param numCh number of colour channels pointcloud has
	\param numPoints number of points in the cloud
	\param move the point cloud to use as a moving scan
*/
DllExport void setMoveScan(unsigned int scanNum, unsigned int numDim, unsigned int numCh, unsigned int numPoints, float* move);

//! Gets the location of the specified moving scans points
/*!
	/param scanNum the number of the moving scan
*/
DllExport float* getMoveLocs(unsigned int scanNum);
//! Gets the colour information of the specified moving scans points
/*!
	/param scanNum the number of the moving scan
*/
DllExport float* getMovePoints(unsigned int scanNum);
//! Gets the number of channels for the specified moving scan
/*!
	/param scanNum the number of the moving scan
*/
DllExport int getMoveNumCh(unsigned int scanNum);
//! Gets the number of dimensions of the specified moving scan
/*!
	/param scanNum the number of the moving scan
*/
DllExport int getMoveNumDim(unsigned int scanNum);
//! Gets the number of points of the specified moving scan
/*!
	/param scanNum the number of the moving scan
*/
DllExport int getMoveNumPoints(unsigned int scanNum);

//! Gets the number of dimensions of the specified base scan
/*!
	/param scanNum the number of the base scan
*/
DllExport int getBaseDim(unsigned int scanNum, unsigned int dim);
//! Gets the number of channels of the specified base scan
/*!
	/param scanNum the number of the base scan
*/
DllExport int getBaseNumCh(unsigned int scanNum);
//! Gets the image being used as the specified base scan
/*!
	/param scanNum the number of the base scan
*/
DllExport float* getBaseImage(unsigned int scanNum);

//! Allocates memory for camera matrix and links to camera transform
/*!
	\param panoramic 1 for panoramic camera model 0 for pin-point camera model
*/
DllExport void setupCamera(int panoramic);
//! Allocates memory for an affine transformation matrix
DllExport void setupTformAffine(void);
//! Allocates memory for a camera transformation matrix and binds to camera
DllExport void setupCameraTform(void);

//! Sets the camera matrix a 3 by 4 matrix given in column major order
/*!
	\param camMat a 3 by 4 camera matrix in column major order
*/
DllExport void setCameraMatrix(float* camMat);
//! Sets the n by n tform matrix in column major order
/*!
	\param tMat n by n transform matrix in column major order. Must be 3 by 3 for affine transforms and 4 by 4 for camera transforms.
*/
DllExport void setTformMatrix(float* tMat);

//! Transforms a moving images location and stores it in the SparseScan Gen
/*!
	/param imgNum the number of the moving scan to transform
*/
DllExport void transform(unsigned int imgNum);

//! Gets the location of the generated Scan gen
/*!
	\return gives an array of size numDim*numPoints holding the location values of gen in column major order
*/
DllExport float* getGenLocs(void);
//! Gets the intensity of points in the generated Scan gen
/*!
	\return gives an array of size numCh*numPoints holding the intensity values of gen in column major order
*/
DllExport float* getGenPoints(void);

//! Replaces the points in the specified moving scan with the current generated one
/*!
	NOTE this only replaces the points intensity, not their location
	Also note to operate efficiently this function SWAPS THE POINTERS OF MOVE AND GEN so cannot be called multiple times unless gen is regenerated first
*/
DllExport void replaceMovePoints(unsigned int scanNum);

//! Gets the number of channels the generated scan has
DllExport int getGenNumCh(void);
//! Gets the number of dimensions the generated scan has
DllExport int getGenNumDim(void);
//! Gets the number of points the generated scan has
DllExport int getGenNumPoints(void);

//! Gives the generated scan the same intensity values as the specified base scan
/*
	\param baseNum the number of the base scan
*/
DllExport void genBaseValues(unsigned int baseNum);

//! Sets up the SSD metric for use with scans
DllExport void setupSSDMetric(void);
//! Sets up the MI metric for use with scans
DllExport void setupMIMetric(unsigned int numBins);
//! Sets up the GOM metric for use with scans
DllExport void setupGOMMetric(void);
//! Sets up the Levinson method's metric for use with scans
DllExport void setupLIVMetric(float* avImg, unsigned int width, unsigned int height);

//! Gets the value of the metric when evaluated between the generated scan Gen and the specified moving scan.
DllExport float getMetricVal(unsigned int moveNum);

DllExport float* outputImage(unsigned int width, unsigned int height, unsigned int moveNum, unsigned int dilate);
DllExport float* outputImageGen(unsigned int width, unsigned int height, unsigned int dilate);

//! Grabs any errors generated by cuda
DllExport void checkCudaErrors(void);

//! Ensures CUDA capable devices are present
DllExport void setupCUDADevices(void);

#endif //MATLAB_CALLS_H
