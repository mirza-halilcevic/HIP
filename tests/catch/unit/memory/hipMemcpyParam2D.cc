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

#include "memcpy2d_tests_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

static constexpr auto MemTypeHost() {
#if HT_AMD
  return hipMemoryTypeHost;
#else
  return CU_MEMORYTYPE_HOST;
#endif
}

static constexpr auto MemTypeDevice() {
#if HT_AMD
  return hipMemoryTypeDevice;
#else
  return CU_MEMORYTYPE_DEVICE;
#endif
}

template <hipMemcpyKind kind> static constexpr auto MemcpyParam2DAdapter() {
  return [](void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
            hipMemcpyKind) {
    hip_Memcpy2D params = {0};
    if constexpr (kind == hipMemcpyDeviceToHost) {
      params.dstMemoryType = MemTypeHost();
      params.dstHost = dst;
      params.srcMemoryType = MemTypeDevice();
      params.srcDevice = reinterpret_cast<hipDeviceptr_t>(src);
    } else if constexpr (kind == hipMemcpyDeviceToDevice) {
      params.dstMemoryType = MemTypeDevice();
      params.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst);
      params.srcMemoryType = MemTypeDevice();
      params.srcDevice = reinterpret_cast<hipDeviceptr_t>(src);
    } else if constexpr (kind == hipMemcpyHostToDevice) {
      params.dstMemoryType = MemTypeDevice();
      params.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst);
      params.srcMemoryType = MemTypeHost();
      params.srcHost = src;
    } else if constexpr (kind == hipMemcpyHostToHost) {
      params.dstMemoryType = MemTypeHost();
      params.dstHost = dst;
      params.srcMemoryType = MemTypeHost();
      params.srcHost = src;
    } else {
      static_assert(sizeof(kind), "Invalid hipMemcpyKind enumerator");
    }

    params.dstPitch = dpitch;
    params.srcPitch = spitch;
    params.WidthInBytes = width;
    params.Height = height;
    return hipMemcpyParam2D(&params);
  };
}

TEST_CASE("Unit_hipMemcpyParam2D_Positive_Basic") {
  SECTION("Device to Host") {
    Memcpy2DDeviceToHostShell<false>(MemcpyParam2DAdapter<hipMemcpyDeviceToHost>());
  }

  SECTION("Device to Device") {
    constexpr auto f = MemcpyParam2DAdapter<hipMemcpyDeviceToDevice>();
    SECTION("Peer access disabled") { Memcpy2DDeviceToDeviceShell<false, false>(f); }
    SECTION("Peer access enabled") { Memcpy2DDeviceToDeviceShell<false, true>(f); }
  }

  SECTION("Host to Device") {
    Memcpy2DHostToDeviceShell<false>(MemcpyParam2DAdapter<hipMemcpyHostToDevice>());
  }

  SECTION("Host to Host") {
    Memcpy2DHostToHostShell<false>(MemcpyParam2DAdapter<hipMemcpyHostToHost>());
  }
}

TEST_CASE("Unit_hipMemcpyParam2D_Positive_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());
  SECTION("Host to Device") {
    Memcpy2DHtoDSyncBehavior(MemcpyParam2DAdapter<hipMemcpyHostToDevice>(), true);
  }

  SECTION("Device to Host") {
    constexpr auto f = MemcpyParam2DAdapter<hipMemcpyDeviceToHost>();
    Memcpy2DDtoHPageableSyncBehavior(f, true);
    Memcpy2DDtoHPageableSyncBehavior(f, true);
  }

  SECTION("Device to Device") {
    Memcpy2DDtoDSyncBehavior(MemcpyParam2DAdapter<hipMemcpyDeviceToDevice>(), false);
  }

  SECTION("Host to Host") {
    Memcpy2DHtoHSyncBehavior(MemcpyParam2DAdapter<hipMemcpyHostToHost>(), true);
  }
}