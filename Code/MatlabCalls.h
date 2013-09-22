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

#endif //MATLAB_CALLS_H
