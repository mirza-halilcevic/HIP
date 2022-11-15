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

TEST_CASE("Unit_hipGraphicsSubResourceGetMappedArray_Positive_Basic") {
  GLContextScopeGuard gl_context;

  CreateGLImageObject();

  cudaGraphicsResource* tex_resource;

  REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsMapResources(1, &tex_resource, 0) == CUDA_SUCCESS);

  cudaArray* image_devptr = nullptr;
  REQUIRE(cudaGraphicsSubResourceGetMappedArray(&image_devptr, tex_resource, 0, 0) == CUDA_SUCCESS);

  REQUIRE(image_devptr != nullptr);

  REQUIRE(cudaGraphicsUnmapResources(1, &tex_resource, 0) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(tex_resource) == CUDA_SUCCESS);
}

TEST_CASE("Unit_hipGraphicsSubResourceGetMappedArray_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  CreateGLImageObject();

  cudaGraphicsResource* tex_resource;

  REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsMapResources(1, &tex_resource, 0) == CUDA_SUCCESS);

  cudaArray* image_devptr = nullptr;

  SECTION("array == nullptr") {
    REQUIRE(cudaGraphicsSubResourceGetMappedArray(nullptr, tex_resource, 0, 0) == CUDA_SUCCESS);
  }

  SECTION("invalid resource") {
    cudaGraphicsResource* invalid_resource;
    REQUIRE(cudaGraphicsSubResourceGetMappedArray(&image_devptr, invalid_resource, 0, 0) ==
            CUDA_ERROR_INVALID_CONTEXT);
  }

  REQUIRE(cudaGraphicsUnmapResources(1, &tex_resource, 0) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(tex_resource) == CUDA_SUCCESS);
}