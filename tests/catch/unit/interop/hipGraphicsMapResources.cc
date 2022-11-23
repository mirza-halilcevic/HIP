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

TEST_CASE("Unit_hipGraphicsMapResources_Positive_Basic") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;
  GLImageObject tex;

  std::array<cudaGraphicsResource_t, 2> resources;

  REQUIRE(cudaGraphicsGLRegisterBuffer(&resources.at(0), vbo, cudaGraphicsRegisterFlagsNone) ==
          CUDA_SUCCESS);
  REQUIRE(cudaGraphicsGLRegisterImage(&resources.at(1), tex, GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);

  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsMapResources(resources.size(), resources.data(), stream) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnmapResources(resources.size(), resources.data(), stream) == CUDA_SUCCESS);

  REQUIRE(cudaStreamDestroy(stream) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(resources.at(0)) == CUDA_SUCCESS);
  REQUIRE(cudaGraphicsUnregisterResource(resources.at(1)) == CUDA_SUCCESS);
}

TEST_CASE("Unit_hipGraphicsMapResources_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  cudaGraphicsResource* vbo_resource;

  REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsRegisterFlagsNone) ==
          CUDA_SUCCESS);

  SECTION("count == 0") {
    REQUIRE(cudaGraphicsMapResources(0, &vbo_resource, 0) == CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("resources == nullptr") {
    REQUIRE(cudaGraphicsMapResources(1, nullptr, 0) == CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("unregistered resource") {
    cudaGraphicsResource* unregistered_resource;
    REQUIRE(cudaGraphicsGLRegisterBuffer(&unregistered_resource, vbo,
                                         cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsUnregisterResource(unregistered_resource) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsMapResources(1, &unregistered_resource, 0) == CUDA_ERROR_INVALID_HANDLE);
  }

  SECTION("already mapped resource") {
    REQUIRE(cudaGraphicsMapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsMapResources(1, &vbo_resource, 0) == CUDA_ERROR_ALREADY_MAPPED);
    REQUIRE(cudaGraphicsUnmapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);
  }

  SECTION("invalid stream") {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaStreamDestroy(stream);
    REQUIRE(cudaGraphicsMapResources(1, &vbo_resource, stream) == CUDA_ERROR_CONTEXT_IS_DESTROYED);
  }

  REQUIRE(cudaGraphicsUnregisterResource(vbo_resource) == CUDA_SUCCESS);
}