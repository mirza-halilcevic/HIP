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

#include "vulkan_test.hh"

constexpr bool enable_validation = false;

TEST_CASE("Unit_hipImportExternalSemaphore_Negative_Parameters") {
  VulkanTest vkt(enable_validation);
  cudaExternalSemaphore_t ext_semaphore;

  SECTION("extSem_out == nullptr") {
    const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_TIMELINE);
    const auto handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_TIMELINE);
    REQUIRE(cudaImportExternalSemaphore(nullptr, &handle_desc) == cudaErrorInvalidValue);
  }

  SECTION("semHandleDesc == nullptr") {
    REQUIRE(cudaImportExternalSemaphore(&ext_semaphore, nullptr) == cudaErrorInvalidValue);
  }

  // TODO Uncomment and disable this for Linux in the JSON file
  //   SECTION("semHandleDesc.handle == NULL") {
  //     const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_TIMELINE);
  //     auto handle_desc = vkt.BuildSemaphoreDescriptor(semaphore,
  //     VK_SEMAPHORE_TYPE_TIMELINE); handle_desc.handle.win32.handle = NULL;
  //     cudaExternalSemaphore_t ext_semaphore;
  //     REQUIRE(cudaImportExternalSemaphore(&ext_semaphore, &handle_desc) ==
  //     cudaErrorInvalidValue);
  //   }

  SECTION("semHandleDesc.flags != 0") {
    const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_TIMELINE);
    auto handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_TIMELINE);
    handle_desc.flags = 1;
    REQUIRE(cudaImportExternalSemaphore(&ext_semaphore, &handle_desc) == cudaErrorInvalidValue);
  }

  SECTION("Invalid semHandleDesc.type") {
    const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_TIMELINE);
    auto handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_TIMELINE);
    handle_desc.type = static_cast<cudaExternalSemaphoreHandleType>(-1);
    REQUIRE(cudaImportExternalSemaphore(&ext_semaphore, &handle_desc) == cudaErrorInvalidValue);
  }
}