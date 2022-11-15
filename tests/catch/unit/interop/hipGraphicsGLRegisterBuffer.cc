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
constexpr std::array<unsigned int, 3> kFlags{cudaGraphicsRegisterFlagsNone,
                                             cudaGraphicsRegisterFlagsReadOnly,
                                             cudaGraphicsRegisterFlagsWriteDiscard};
}  // anonymous namespace

TEST_CASE("Unit_hipGraphicsGLRegisterBuffer_Positive_Basic") {
  GLContextScopeGuard gl_context;

  const auto flags = GENERATE(from_range(begin(kFlags), end(kFlags)));

  CreateGLBufferObject();

  cudaGraphicsResource* vbo_resource;

  REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, flags) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(vbo_resource) == CUDA_SUCCESS);
}

TEST_CASE("Unit_hipGraphicsGLRegisterBuffer_Positive_Register_Twice") {
  GLContextScopeGuard gl_context;

  CreateGLBufferObject();

  cudaGraphicsResource *vbo_resource_1, *vbo_resource_2;

  REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource_1, vbo, cudaGraphicsMapFlagsNone) ==
          CUDA_SUCCESS);
  REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource_2, vbo, cudaGraphicsMapFlagsNone) ==
          CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(vbo_resource_1) == CUDA_SUCCESS);
  REQUIRE(cudaGraphicsUnregisterResource(vbo_resource_2) == CUDA_SUCCESS);
}

TEST_CASE("Unit_hipGraphicsGLRegisterBuffer_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  CreateGLBufferObject();

  cudaGraphicsResource* vbo_resource;

  SECTION("resource == nullptr") {
    REQUIRE(cudaGraphicsGLRegisterBuffer(nullptr, vbo, cudaGraphicsMapFlagsNone) ==
            CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("invalid buffer") {
    REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, GLuint{}, cudaGraphicsMapFlagsNone) ==
            CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("invalid flags") {
    REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, static_cast<unsigned int>(-1)) ==
            CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("flags == hipGraphicsRegisterFlagsSurfaceLoadStore") {
    REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo,
                                         cudaGraphicsRegisterFlagsSurfaceLoadStore) ==
            CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("flags == hipGraphicsRegisterFlagsTextureGather") {
    REQUIRE(
        cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsRegisterFlagsTextureGather) ==
        CUDA_ERROR_INVALID_VALUE);
  }
}