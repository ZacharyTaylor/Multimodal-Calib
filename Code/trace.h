/*! Trace is used for outputing information, errors and warnings to matlab from within C++ code. 
Identical behaviour is also implemented in .m files for use with matlab and .cu for use with CUDA. 

The trace can be set to 4 different levels
0 - no debug info
1 - errors only
2 - errors and warnings
3 - errors, warnings and information

The output of the trace states the file and line of the code that was triggered as well as the
laungage it is in and the type of error. For example
TRACE_ERROR("testing"); // would output the following
C++ Error at trace.h(13): testing
*/

#ifndef TRACE_H
#define TRACE_H

#include "common.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

void traceInfo(std::string in);

void traceWarning(std::string in);

void traceError(std::string in);

/*
#if(_WIN32)
	#include <crtdbg.h>
#endif

//! trace 0 = off, 1 = errors, 2 = warnings, 3 = info
#define DEBUG_TRACE 1

#define BUFF_SIZE 4096

#ifdef DEBUG_TRACE 

	#if(DEBUG_TRACE > 0)

		#if(_WIN32)
			inline void TRACE_IN(const char* format, ...)
			{
				va_list argptr;
				va_start(argptr, format);
	
				char buff[BUFF_SIZE];
				vsnprintf(buff,BUFF_SIZE, format, argptr);
				va_end(argptr);

				printf("%s",buff);
				printf("\n");
			}
		#else
			inline void TRACE_IN(const char* format, ...)
			{
				printf("%s",format);
				printf("\n");
			}

			inline void DO_NOTHING(const char* format, ...)
			{
				return;
			}
		#endif


		#define TRACE_ERROR printf("C++ Error at %s(%d): ", FILE, __LINE__); TRACE_IN

		#if (DEBUG_TRACE > 1)
			#define TRACE_WARNING printf("C++ Warning at %s(%d): ", FILE, __LINE__); TRACE_IN

	
			#if (DEBUG_TRACE > 2)
				#define TRACE_INFO printf("C++ Info at %s(%d): ", FILE, __LINE__); TRACE_IN
			#else
				#if(_WIN32)
					#define TRACE_INFO ((void)0)
				#else
					#define TRACE_INFO DO_NOTHING
				#endif
					
			#endif
		#else
			#if(_WIN32)
					#define TRACE_INFO ((void)0)
					#define TRACE_WARNING ((void)0)
			#else
					#define TRACE_INFO DO_NOTHING
					#define TRACE_WARNING DO_NOTHING
			#endif
		#endif
	#else
		#if(_WIN32)
			// Remove for release mode
			#define TRACE_IN  ((void)0)
			#define TRACE_ERROR ((void)0)
			#define TRACE_INFO ((void)0)
			#define TRACE_WARNING ((void)0)
		#else
			inline void DO_NOTHING(const char* format, ...)
			{
				return;
			}
			#define TRACE_ERROR DO_NOTHING
			#define TRACE_INFO DO_NOTHING
			#define TRACE_WARNING DO_NOTHING
		#endif
			
	#endif
#endif
*/
#endif // TRACE_H
