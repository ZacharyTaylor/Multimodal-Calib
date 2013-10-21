/* demonstrates using cout with MATLAB. this code was
* taken from sample code supplied with libg++:
*
* libg++_2.7.1/iostream_28.html#SEC28
*
* which demonstrated how to do simple redirection.
*
* author : Michael Schweisguth
*/

#include <iostream>
#include "mex.h"

typedef int (__cdecl * PrintFunc)(const char *, ...);

int __cdecl nullprintf(const char *, ...)
{
	//nothing is printed
	return 0;
}

template<int N, PrintFunc PRINTFUNC, int VERBOSE=0>
class matlab_streambuf : public std::streambuf
{
protected:

	char m_buffer[N];

public:

	// the constructor sets up the entire reserve
	// buffer to buffer output,... since there is
	// no input! ;-)

	matlab_streambuf() : std::streambuf(m_buffer,N)
	{
		// the entire buffer is devoted to the output.

		setp(m_buffer,m_buffer+N);
	}

	// outputs characters to the device via
	// PRINTFUNC. since there is no input,
	// there isnt really anything to sync!

	int sync()
	{
		int n = out_waiting();

		if (VERBOSE) {
			PRINTFUNC("n=%d\n", n);
		}

		xsputn(pbase(), n);

		pbump(-n);

		return 0;
	}

	// called when the associated buffer is
	// full:

	int overflow(int ch)
	{
		sync();

		if (VERBOSE) {
			PRINTFUNC("OF:%c", ch);
		} else {
			PRINTFUNC("%c", ch);
		}

		return 0;
	}

	// VisualC requires that this be defined.
	// since there is no input available, return
	// EOF.

	int underflow()
	{
		return EOF;
	}

	// prints a series of characters to the
	// screen:

	int xsputn(char *text, int n)
	{
		if (!n) {
			return 0;
		}

		char printf_fmt[16];

		sprintf(printf_fmt, "%%.%ds", n);

		if (VERBOSE) {
			PRINTFUNC("format = %s\n", printf_fmt);
		}

		PRINTFUNC(printf_fmt, text);

		return n;
	}
};