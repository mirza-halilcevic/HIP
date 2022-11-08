/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <EGL/egl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
                                       EGL_PBUFFER_BIT,
                                       EGL_BLUE_SIZE,
                                       8,
                                       EGL_GREEN_SIZE,
                                       8,
                                       EGL_RED_SIZE,
                                       8,
                                       EGL_DEPTH_SIZE,
                                       8,
                                       EGL_RENDERABLE_TYPE,
                                       EGL_OPENGL_BIT,
                                       EGL_NONE};

static const int pbufferWidth = 9;
static const int pbufferHeight = 9;

static const EGLint pbufferAttribs[] = {
    EGL_WIDTH, pbufferWidth, EGL_HEIGHT, pbufferHeight, EGL_NONE,
};

const uint width = 512, height = 512;

GLuint pbo;                                     // OpenGL pixel buffer object
struct cudaGraphicsResource* hip_pbo_resource;  // HIP Graphics Resource (to transfer PBO)

TEST_CASE("Blahem") {
  // 1. Initialize EGL
  EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  EGLint major, minor;

  REQUIRE(eglInitialize(eglDpy, &major, &minor));

  // 2. Select an appropriate configuration
  EGLint numConfigs;
  EGLConfig eglCfg;

  REQUIRE(eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs));

  // 3. Create a surface
  EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);

  // 4. Bind the API
  REQUIRE(eglBindAPI(EGL_OPENGL_API));

  // 5. Create a context and make it current
  EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);

  REQUIRE(eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx));

  // from now on use your OpenGL context

  // create pixel buffer object
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4, 0,
               GL_STREAM_DRAW_ARB);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  cudaFree(0);

  // register this buffer object with CUDA
  auto error = cudaGraphicsGLRegisterBuffer(&hip_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
  std::cout << cudaGetErrorString(error) << std::endl;

  error = cudaGraphicsUnregisterResource(hip_pbo_resource);
  std::cout << cudaGetErrorString(error) << std::endl;

  // 6. Terminate EGL when finished
  REQUIRE(eglTerminate(eglDpy));
}