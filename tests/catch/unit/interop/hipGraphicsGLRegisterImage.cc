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
#include <hip/hip_runtime_api.h>

#include "interop_common.hh"

#include "GLContextScopeGuard.hh"

namespace {
constexpr std::array<unsigned int, 5> kFlags{
    cudaGraphicsMapFlagsNone, cudaGraphicsMapFlagsReadOnly, cudaGraphicsMapFlagsWriteDiscard,
    cudaGraphicsRegisterFlagsSurfaceLoadStore, cudaGraphicsRegisterFlagsTextureGather};
}  // anonymous namespace

TEST_CASE("Unit_hipGraphicsGLRegisterImage_Positive_Basic") {
  GLContextScopeGuard gl_context;

  const auto flags = GENERATE(from_range(begin(kFlags), end(kFlags)));

  CreateGLImageObject();

  cudaGraphicsResource* tex_resource;

  REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D, flags) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(tex_resource) == CUDA_SUCCESS);
}

TEST_CASE("Unit_hipGraphicsGLRegisterImage_Positive_Register_Twice") {
  GLContextScopeGuard gl_context;

  CreateGLImageObject();

  cudaGraphicsResource *tex_resource_1, *tex_resource_2;

  REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource_1, tex, GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);
  REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource_2, tex, GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(tex_resource_1) == CUDA_SUCCESS);
  REQUIRE(cudaGraphicsUnregisterResource(tex_resource_2) == CUDA_SUCCESS);
}

TEST_CASE("Unit_hipGraphicsGLRegisterImage_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  CreateGLImageObject();

  cudaGraphicsResource* tex_resource;

  SECTION("resource == nullptr") {
    REQUIRE(cudaGraphicsGLRegisterImage(nullptr, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone) ==
            CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("invalid image") {
    REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, GLuint{}, GL_TEXTURE_2D,
                                        cudaGraphicsMapFlagsNone) == CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("invalid target") {
    REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, tex, GL_BUFFER, cudaGraphicsMapFlagsNone) ==
            CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("target does not match the object") {
    REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, tex, GL_RENDERBUFFER,
                                        cudaGraphicsMapFlagsNone) == CUDA_ERROR_INVALID_HANDLE);
  }

  SECTION("invalid flags") {
    REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D,
                                        static_cast<unsigned int>(-1)) == CUDA_ERROR_INVALID_VALUE);
  }
}