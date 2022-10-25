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

__global__ void SetAttributeTestKernel() {}

TEST_CASE("Unit_hipFuncSetAttribute_Positive_MaxDynamicSharedMemorySize") {
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(SetAttributeTestKernel),
                                hipFuncAttributeMaxDynamicSharedMemorySize, 1024));

  hipFuncAttributes attributes;
  HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<void*>(SetAttributeTestKernel)));

  REQUIRE(attributes.maxDynamicSharedSizeBytes == 1024);
}

TEST_CASE("Unit_hipFuncSetAttribute_Positive_PreferredSharedMemoryCarveout") {
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<void*>(SetAttributeTestKernel),
                                hipFuncAttributePreferredSharedMemoryCarveout, 50));

  hipFuncAttributes attributes;
  HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<void*>(SetAttributeTestKernel)));

  REQUIRE(attributes.preferredShmemCarveout == 50);
}

TEST_CASE("Unit_hipFuncSetAttribute_Negative_Parameters") {
  SECTION("func == nullptr") {
    HIP_CHECK_ERROR(hipFuncSetAttribute(nullptr, hipFuncAttributePreferredSharedMemoryCarveout, 50),
                    hipErrorInvalidDeviceFunction);
  }
  SECTION("invalid attribute") {
    HIP_CHECK_ERROR(hipFuncSetAttribute(reinterpret_cast<void*>(SetAttributeTestKernel),
                                        static_cast<hipFuncAttribute>(-1), 50),
                    hipErrorInvalidValue);
  }
  SECTION("hipFuncAttributeMaxDynamicSharedMemorySize invalid value") {
    // The sum of this value and the function attribute sharedSizeBytes cannot exceed the device
    // attribute cudaDevAttrMaxSharedMemoryPerBlockOptin
    int value;
    HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));

    hipFuncAttributes attributes;
    HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<void*>(SetAttributeTestKernel)));

    HIP_CHECK_ERROR(hipFuncSetAttribute(reinterpret_cast<void*>(SetAttributeTestKernel),
                                        hipFuncAttributeMaxDynamicSharedMemorySize,
                                        value - attributes.sharedSizeBytes + 1),
                    hipErrorInvalidValue);
  }
  SECTION("hipFuncAttributePreferredSharedMemoryCarveout invalid value") {
    HIP_CHECK_ERROR(hipFuncSetAttribute(reinterpret_cast<void*>(SetAttributeTestKernel),
                                        hipFuncAttributePreferredSharedMemoryCarveout, 101),
                    hipErrorInvalidValue);
  }
}