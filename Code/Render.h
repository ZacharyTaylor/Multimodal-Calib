#ifndef RENDER_H
#define RENDER_H

#include "common.h"
#include "trace.h"
#include "Scan.h"

class Render{
public:
	float* out_;
	Render(void);
	~Render(void);
	void GetImage(PointsList* points, PointsList* loc, size_t numPoints, size_t width, size_t height, size_t depth);
};

#endif //RENDER_H

/*#ifndef RENDER_H
#define RENDER_H

#include "common.h"
#include "trace.h"

// OpenGL Graphics includes
#include "inc/GL/glew.h"
#if defined (__APPLE__) || defined(MACOSX)
	#include "inc/GLUT/glut.h"
#else
	#include "inc/GL/freeglut.h"
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "inc/helper_functions.h"
#include "inc/timer.h"    

#define REFRESH_DELAY     10 //ms

class Render{
private:
	static const unsigned int window_width  = 512;
	static const unsigned int window_height = 512;

	// vbo variables
	static GLuint vbo;
	static struct cudaGraphicsResource *cuda_vbo_resource;
	static void *d_vbo_buffer = NULL;

	static float g_fAnim = 0.0;

	// mouse controls
	static int mouse_old_x, mouse_old_y;
	static int mouse_buttons = 0;
	static float rotate_x = 0.0, rotate_y = 0.0;
	static float translate_z = -3.0;

	static bool initGL(int *argc, char **argv);
	static void keyboard(unsigned char key, int , int);
	static void mouse(int button, int state, int x, int y);
	static void motion(int x, int y);
	static void display(void);
	static void timerEvent(int value);
	static void cleanup(void);

	static void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);
	static void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
};

#endif //RENDER_H*/