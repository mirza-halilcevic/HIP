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
#include <hip/hip_cooperative_groups.h>
#include <resource_guards.hh>
#include <utils.hh>

__global__ void kernel2() {}

__global__ void coop_kernel() {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  grid.sync();
}

TEST_CASE("Unit_hipLaunchCooperativeKernel_Positive_Basic") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxGridDimX, 0);
  const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxGridDimY, 0);
  const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxGridDimZ, 0);

  HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(coop_kernel), dim3{2, 2, 2},
                                       dim3{1, 1, 1}, nullptr, 0, nullptr));
  HIP_CHECK(hipDeviceSynchronize());


  //HIP_CHECK(hipExtLaunchKernel(reinterpret_cast<void*>(coop_kernel), dim3{x, y, z}, dim3{1, 1, 1},
  //                             nullptr, 0, nullptr, nullptr, nullptr, 0u));
  //HIP_CHECK(hipDeviceSynchronize());
}

TEST_CASE("Unit_hipLaunchCooperativeKernel_Positive_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                         dim3{x, 1, 1}, nullptr, 0, nullptr));
  }

  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                         dim3{y, 1, 1}, nullptr, 0, nullptr));
  }

  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                         dim3{z, 1, 1}, nullptr, 0, nullptr));
  }
}

TEST_CASE("Unit_hipLaunchCooperativeKernel_Negative_Parameters") {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HipTest::HIP_SKIP_TEST("CooperativeLaunch not supported");
    return;
  }

  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(static_cast<void*>(nullptr), dim3{1, 1, 1},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidSymbol);
  }

  SECTION("gridDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{0, 1, 1},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("gridDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 0, 1},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("gridDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 0},
                                               dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{0, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{1, 0, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{1, 1, 0}, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDim.x > maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{x, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.y > maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{1, y, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.z > maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{1, 1, z}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.x * blockDim.y * blockDim.z > maxThreadsPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0);
    const unsigned int dim = std::ceil(std::cbrt(max));
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{dim, dim, dim}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION(
      "gridDim.x * gridDim.y * gridDim.z > maxActiveBlocksPerMultiprocessor * "
      "multiProcessorCount") {
    int max_blocks;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
                                                           reinterpret_cast<void*>(kernel2), 1, 0));
    const unsigned int multiproc_count =
        GetDeviceAttribute(hipDeviceAttributeMultiprocessorCount, 0);
    const unsigned int dim = max_blocks * multiproc_count + 1;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{dim, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("sharedMemBytes > maxSharedMemoryPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{1, 1, 1}, nullptr, max, nullptr),
                    hipErrorCooperativeLaunchTooLarge);
  }

  SECTION("Invalid stream") {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipLaunchCooperativeKernel(reinterpret_cast<void*>(kernel2), dim3{1, 1, 1},
                                               dim3{1, 1, 1}, nullptr, 0, stream),
                    hipErrorInvalidValue);
  }
}