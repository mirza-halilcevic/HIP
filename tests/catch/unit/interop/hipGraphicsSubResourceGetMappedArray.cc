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

TEST_CASE("Unit_hipGraphicsSubResourceGetMappedArray_Positive_Basic") {
  GLContextScopeGuard gl_context;

  GLImageObject tex;

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

  GLImageObject tex;

  cudaGraphicsResource* tex_resource;

  REQUIRE(cudaGraphicsGLRegisterImage(&tex_resource, tex, GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsMapResources(1, &tex_resource, 0) == CUDA_SUCCESS);

  cudaArray* image_devptr = nullptr;

  SECTION("array == nullptr") {
    REQUIRE(cudaGraphicsSubResourceGetMappedArray(nullptr, tex_resource, 0, 0) == CUDA_SUCCESS);
  }

  SECTION("non-texture resource") {
    GLBufferObject vbo;
    cudaGraphicsResource* vbo_resource;

    REQUIRE(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsRegisterFlagsNone) ==
            CUDA_SUCCESS);
    REQUIRE(cudaGraphicsMapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);

    REQUIRE(cudaGraphicsSubResourceGetMappedArray(&image_devptr, vbo_resource, 0, 0) ==
            CUDA_ERROR_NOT_MAPPED_AS_ARRAY);

    REQUIRE(cudaGraphicsUnmapResources(1, &vbo_resource, 0) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsUnregisterResource(vbo_resource) == CUDA_SUCCESS);
  }

  SECTION("unregistered resource") {
    cudaGraphicsResource* unregistered_resource;
    REQUIRE(cudaGraphicsGLRegisterImage(&unregistered_resource, tex, GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsUnregisterResource(unregistered_resource) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsSubResourceGetMappedArray(&image_devptr, unregistered_resource, 0, 0) ==
            CUDA_ERROR_CONTEXT_IS_DESTROYED);
  }

  SECTION("not mapped resource") {
    cudaGraphicsResource* not_mapped_resource;
    REQUIRE(cudaGraphicsGLRegisterImage(&not_mapped_resource, tex, GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsSubResourceGetMappedArray(&image_devptr, not_mapped_resource, 0, 0) ==
            CUDA_ERROR_NOT_MAPPED);
    REQUIRE(cudaGraphicsUnregisterResource(not_mapped_resource) == CUDA_SUCCESS);
  }

  SECTION("unmapped resource") {
    cudaGraphicsResource* unmapped_resource;

    REQUIRE(cudaGraphicsGLRegisterImage(&unmapped_resource, tex, GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsNone) == CUDA_SUCCESS);

    REQUIRE(cudaGraphicsMapResources(1, &unmapped_resource, 0) == CUDA_SUCCESS);
    REQUIRE(cudaGraphicsUnmapResources(1, &unmapped_resource, 0) == CUDA_SUCCESS);

    REQUIRE(cudaGraphicsSubResourceGetMappedArray(&image_devptr, unmapped_resource, 0, 0) ==
            CUDA_ERROR_NOT_MAPPED);

    REQUIRE(cudaGraphicsUnregisterResource(unmapped_resource) == CUDA_SUCCESS);
  }

  SECTION("invalid arrayIndex") {
    REQUIRE(cudaGraphicsSubResourceGetMappedArray(&image_devptr, tex_resource,
                                                  std::numeric_limits<int>::max(),
                                                  0) == CUDA_ERROR_INVALID_VALUE);
  }

  SECTION("invalid mipLevel") {
    REQUIRE(cudaGraphicsSubResourceGetMappedArray(&image_devptr, tex_resource, 0,
                                                  std::numeric_limits<int>::max()) ==
            CUDA_ERROR_INVALID_VALUE);
  }

  REQUIRE(cudaGraphicsUnmapResources(1, &tex_resource, 0) == CUDA_SUCCESS);

  REQUIRE(cudaGraphicsUnregisterResource(tex_resource) == CUDA_SUCCESS);
}