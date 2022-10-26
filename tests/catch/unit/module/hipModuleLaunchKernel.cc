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

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

static inline hipModule_t GetModule() {
  HIP_CHECK(hipFree(nullptr));
  static const auto mg = ModuleGuard::LoadModule("get_function_module.code");
  return mg.module();
}

static inline hipFunction_t GetKernel(const char* kname) {
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, GetModule(), kname));
  return kernel;
}

TEST_CASE("Unit_hipModuleLaunchKernel_Positive_Basic") {
  SECTION("Kernel with no arguments") {
    hipFunction_t f = GetKernel("global_kernel");
    HIP_CHECK(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    hipFunction_t f = GetKernel("kernel_42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    void* kernel_args[1] = {&result_ptr};
    HIP_CHECK(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, nullptr));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }

  SECTION("Kernel with arguments using extra") {
    hipFunction_t f = GetKernel("kernel_42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    size_t size = sizeof(result_ptr);
    // clang-format off
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &result_ptr, 
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, 
        HIP_LAUNCH_PARAM_END
    };
    // clang-format on
    HIP_CHECK(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, extra));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }
}

TEST_CASE("Unit_hipModuleLaunchKernel_Positive_Parameters") {
  constexpr auto LaunchNOPKernel = [](unsigned int blockDimX, unsigned int blockDimY,
                                      unsigned int blockDimZ) {
    hipFunction_t f = GetKernel("global_kernel");
    HIP_CHECK(hipModuleLaunchKernel(f, 1, 1, 1, blockDimX, blockDimY, blockDimZ, 0, nullptr,
                                    nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  };

  SECTION("blockDimX == maxblockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    LaunchNOPKernel(x, 1, 1);
  }

  SECTION("blockDimY == maxblockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    LaunchNOPKernel(1, y, 1);
  }

  SECTION("blockDimZ == maxblockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    LaunchNOPKernel(1, 1, z);
  }
}

TEST_CASE("Unit_hipModuleLaunchKernel_Negative_Parameters") {
  hipFunction_t f = GetKernel("global_kernel");

  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(hipModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidImage);
  }

  SECTION("gridDimX == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 0, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("gridDimY == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 0, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("gridDimZ == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 0, 1, 1, 1, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimX == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 0, 1, 1, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimY == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 1, 0, 1, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimZ == 0") {
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("blockDimX > maxblockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0) + 1u;
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, x, 1, 1, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDimY > maxblockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0) + 1u;
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 1, y, 1, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDimZ > maxblockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0) + 1u;
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, z, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDimX * blockDimY * blockDimZ > MaxThreadsPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0);
    const unsigned int dim = std::ceil(std::cbrt(max));
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, dim, dim, dim, 0, nullptr, nullptr, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("sharedMemBytes > max shared memory per block") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0) + 1u;
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 1, max, nullptr, nullptr, nullptr),
                    hipErrorOutOfMemory);
  }

  SECTION("Invalid stream") {
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 0, 0, stream, nullptr, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Passing kernel_args and extra simultaneously") {
    hipFunction_t f = GetKernel("kernel_42");
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    int* result_ptr = result_dev.ptr();
    size_t size = sizeof(result_ptr);
    void* kernel_args[1] = {&result_ptr};
    // clang-format off
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &result_ptr, 
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, 
        HIP_LAUNCH_PARAM_END
    };
    // clang-format on
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, kernel_args, extra),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid extra") {
    hipFunction_t f = GetKernel("kernel_42");
    void* extra[0] = {};
    HIP_CHECK_ERROR(hipModuleLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, extra),
                    hipErrorNotInitialized);
  }
}