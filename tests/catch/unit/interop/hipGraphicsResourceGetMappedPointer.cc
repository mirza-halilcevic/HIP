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

TEST_CASE("Unit_hipGraphicsResourceGetMappedPointer_Positive_Basic") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  cudaGraphicsResource* vbo_resource;

  REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsRegisterFlagsNone) ==
          CUDA_SUCCESS);

  REQUIRE(cudaGraphicsMapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);

  float* buffer_devptr = nullptr;
  size_t size = 0;

  REQUIRE(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), &size,
                                               vbo_resource) == CUDA_SUCCESS);

  REQUIRE(buffer_devptr != nullptr);
  REQUIRE(size == vbo.kSize);

  REQUIRE(cudaGraphicsUnmapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(vbo_resource) == CUDA_SUCCESS);
}

TEST_CASE("Unit_hipGraphicsResourceGetMappedPointer_Positive_Parameters") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  cudaGraphicsResource* vbo_resource;

  REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsRegisterFlagsNone) ==
          CUDA_SUCCESS);

  REQUIRE(cudaGraphicsMapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);

  float* buffer_devptr = nullptr;
  size_t size = 0;

  SECTION("devPtr == nullptr") {
    REQUIRE(cudaGraphicsResourceGetMappedPointer(nullptr, &size, vbo_resource) == CUDA_SUCCESS);
    REQUIRE(size == vbo.kSize);
  }

  SECTION("size == nullptr") {
    REQUIRE(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), nullptr,
                                                 vbo_resource) == CUDA_SUCCESS);
    REQUIRE(buffer_devptr != nullptr);
  }

  REQUIRE(cudaGraphicsUnmapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(vbo_resource) == CUDA_SUCCESS);
}

TEST_CASE("Unit_hipGraphicsResourceGetMappedPointer_Negative_Parameters") {
  GLContextScopeGuard gl_context;

  GLBufferObject vbo;

  cudaGraphicsResource* vbo_resource;

  REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsRegisterFlagsNone) ==
          CUDA_SUCCESS);

  REQUIRE(cudaGraphicsMapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);

  float* buffer_devptr = nullptr;
  size_t size = 0;

  SECTION("non-pointer resource") {
    GLImageObject tex;
    cudaGraphicsResource* tex_resource;

    REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D,
                                        cudaGraphicsMapFlagsNone) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsMapResources(1, &tex_resource, 0) == CUDA_SUCCESS);

    REQUIRE(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), &size,
                                                 tex_resource) == CUDA_ERROR_NOT_MAPPED_AS_POINTER);

    REQUIRE(cudaGraphicsUnmapResources(1, &tex_resource, 0) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsUnregisterResource(tex_resource) == CUDA_SUCCESS);
  }

  SECTION("unregistered resource") {
    cudaGraphicsResource* unregistered_resource;
    REQUIRE(cudaGraphicsGLRegisterBuffer(&unregistered_resource, vbo,
                                         cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsUnregisterResource(unregistered_resource) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), &size,
                                                 unregistered_resource) ==
            CUDA_ERROR_CONTEXT_IS_DESTROYED);
  }

  SECTION("not mapped resource") {
    cudaGraphicsResource* not_mapped_resource;
    REQUIRE(cudaGraphicsGLRegisterBuffer(&not_mapped_resource, vbo,
                                         cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), &size,
                                                 not_mapped_resource) == CUDA_ERROR_NOT_MAPPED);
    REQUIRE(cudaGraphicsUnregisterResource(not_mapped_resource) == CUDA_SUCCESS);
  }

  SECTION("unmapped resource") {
    cudaGraphicsResource* unmapped_resource;

    REQUIRE(cudaGraphicsGLRegisterBuffer(&unmapped_resource, vbo, cudaGraphicsRegisterFlagsNone) ==
            CUDA_SUCCESS);

    REQUIRE(cudaGraphicsMapResources(1, &unmapped_resource, 0) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsUnmapResources(1, &unmapped_resource, 0) == CUDA_SUCCESS);

    REQUIRE(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&buffer_devptr), &size,
                                                 unmapped_resource) == CUDA_ERROR_NOT_MAPPED);

    REQUIRE(cudaGraphicsUnregisterResource(unmapped_resource) == CUDA_SUCCESS);
  }

  REQUIRE(cudaGraphicsUnmapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(vbo_resource) == CUDA_SUCCESS);
}