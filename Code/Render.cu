#include "Render.h"
#include "Metric.h"

Render::Render(void):out_(NULL){};

Render::~Render(void){
	delete[] out_;
}

void Render::GetImage(SparseScan* in, size_t width, size_t height){
	float* d_out;

	size_t depth = in->getNumCh();
	CudaSafeCall(cudaMalloc(&d_out, sizeof(float)*width*height*depth));
	CudaSafeCall(cudaMemset(d_out, 0, sizeof(float)*width*height*depth));

	if(out_ != NULL){
		delete[] out_;
		out_ = NULL;
	}
	out_ = new float[width*height*depth];

	for(size_t i = 0; i < depth; i++){
		generateOutputKernel<<<gridSize(in->getNumPoints()) ,BLOCK_SIZE>>>(
			&(((float*)in->GetLocation()->GetGpuPointer())[in->getNumPoints()*i]),
			&(((float*)in->getPoints()->GetGpuPointer())[in->getNumPoints()*i]), 
			&(d_out[width*height*i]), 
			width, 
			height, 
			in->getNumPoints());
		CudaCheckError();
	}

	CudaSafeCall(cudaMemcpy(out_, d_out, sizeof(float)*width*height*depth, cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaFree(d_out));
}

/*#include "Render.h"

bool Render::initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

	// initialize necessary OpenGL extensions
	glewInit();

	if (! glewIsSupported("GL_VERSION_2_0 "))
	{
		TRACE_ERROR("Support for necessary OpenGL extensions missing.");
		return false;
	}
	
	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

	SdkCheckErrorGL();

	return true;
}

void Render::mouse(int button, int state, int x, int y){
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1<<button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void Render::motion(int x, int y){
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void Render::keyboard(unsigned char key, int, int){
    switch (key)
    {
        case (27) :
            exit(EXIT_SUCCESS);
            break;
    }
}

void Render::display(){
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    //runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void Render::timerEvent(int value){
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void Render::cleanup(){
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}

void Render::createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

void Render::deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(vbo_res);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}*/