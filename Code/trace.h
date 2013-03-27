// TRACE macro for win32
#ifndef TRACE_H
#define TRACE_H

#include <crtdbg.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

//trace 0 = off, 1 = errors, 2 = warnings, 3 = info
#define DEBUG_TRACE 3

#ifdef DEBUG_TRACE 
	#if(DEBUG_TRACE > 0)

		#define FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

		__inline void TRACE_IN(const char* format, ...)
		{
			va_list argptr;
			va_start(argptr, format);
			vfprintf(stderr, format, argptr);
			va_end(argptr);

			printf("\n");
		}

		#define TRACE_ERROR printf("Error: %s(%d): ", FILE, __LINE__); TRACE_IN
	
		#if (DEBUG_TRACE > 1)
			#define TRACE_WARNING printf("Warning: %s(%d): ", FILE, __LINE__); TRACE_IN

		
			#if (DEBUG_TRACE > 2)
				#define TRACE_INFO printf("Info: %s(%d): ", FILE, __LINE__); TRACE_IN
			#else
				#define TRACE_INFO ((void)0)
			#endif
		#else
			#define TRACE_INFO ((void)0)
			#define TRACE_WARNING ((void)0)
		#endif
	#else
		// Remove for release mode
		#define TRACE_IN  ((void)0)
		#define TRACE_ERROR ((void)0)
		#define TRACE_INFO ((void)0)
		#define TRACE_WARNING ((void)0)
	#endif
#endif

#endif // TRACE_H