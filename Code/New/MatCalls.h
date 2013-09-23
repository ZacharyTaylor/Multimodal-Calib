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
//! Clears the memory allocated for transforms
DllExport void clearTform(void);
//! Clears the memory allocated for extra parameters of inherited classes
DllExport void clearExtras(void);
//! Clears everything
DllExport void clearEverything(void);

//! Sets up things ready to perform camera calibration
DllExport void initalizeCamera(void);

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
DllExport void addBaseImage(float* baseIn, unsigned int height, unsigned int width, unsigned int depth, unsigned int tformIdx, unsigned int scanIdx);

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

//!Evalutaes and returns metric
DllExport float evalMetric(void);
#endif //MATLAB_CALLS_H
