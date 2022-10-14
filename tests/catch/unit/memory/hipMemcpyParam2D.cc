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

TEST_CASE("Unit_hipMemcpyParam2D_Positive_Basic") {
  constexpr bool sync = false;
  SECTION("Device to Host") {
    Memcpy2DDeviceToHostShell<sync>(MemcpyParam2DAdapter<hipMemcpyDeviceToHost>());
  }

  SECTION("Device to Device") {
    const auto f = MemcpyParam2DAdapter<hipMemcpyDeviceToDevice>();
    SECTION("Peer access disabled") { Memcpy2DDeviceToDeviceShell<sync, false>(f); }
    SECTION("Peer access enabled") { Memcpy2DDeviceToDeviceShell<sync, true>(f); }
  }

  SECTION("Host to Device") {
    Memcpy2DHostToDeviceShell<sync>(MemcpyParam2DAdapter<hipMemcpyHostToDevice>());
  }

  SECTION("Host to Host") {
    Memcpy2DHostToHostShell<sync>(MemcpyParam2DAdapter<hipMemcpyHostToHost>());
  }
}

TEST_CASE("Unit_hipMemcpyParam2D_Positive_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());
  SECTION("Host to Device") {
    Memcpy2DHtoDSyncBehavior(MemcpyParam2DAdapter<hipMemcpyHostToDevice>(), true);
  }

  SECTION("Device to Host") {
    const auto f = MemcpyParam2DAdapter<hipMemcpyDeviceToHost>();
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

TEST_CASE("Unit_hipMemcpyParam2D_Positive_Parameters") {
  const auto f = [](void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                    size_t height, hipMemcpyKind kind) {
    return MemcpyParam2DAdapter<false>(kind)(dst, dpitch, src, spitch, width, height, kind);
  };
  Memcpy2DZeroWidthHeight<false>(f);
}

TEST_CASE("Unit_hipMemcpyParam2D_Negative_Parameters") {
  constexpr size_t cols = 128;
  constexpr size_t rows = 128;

  constexpr auto NegativeTests = [](void* dst, size_t dpitch, const void* src, size_t spitch,
                                    size_t width, size_t height, hipMemcpyKind kind) {
    SECTION("dst == nullptr") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind)(nullptr, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("src == nullptr") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind)(dst, dpitch, nullptr, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("dstPitch < WithInBytes") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind)(dst, width - 1, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("srcPitch < WidthInBytes") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind)(dst, dpitch, src, width - 1, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("dstPitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind)(dst, static_cast<size_t>(attr) + 1, src, spitch,
                                                 width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("srcPitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind)(dst, dpitch, src, static_cast<size_t>(attr) + 1,
                                                 width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("WidthInBytes + srcXInBytes > srcPitch") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind, nullptr, make_hipExtent(spitch - width + 1, 0, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("WidthInBytes + dstXInBytes > dstPitch") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind, nullptr, make_hipExtent(0, 0, 0),
                                           make_hipExtent(dpitch - width + 1, 0, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("srcY out of bounds") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind, nullptr, make_hipExtent(0, 1, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
    SECTION("dstY out of bounds") {
      HIP_CHECK_ERROR(MemcpyParam2DAdapter(kind, nullptr, make_hipExtent(0, 0, 0),
                                           make_hipExtent(0, 1, 0))(
                          dst, dpitch, src, spitch, width, height, kind),
                      hipErrorInvalidValue);
    }
  };

  SECTION("Host to device") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(device_alloc.ptr(), device_alloc.pitch(), host_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyHostToDevice);
  }

  SECTION("Device to host") {
    LinearAllocGuard2D<int> device_alloc(cols, rows);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, device_alloc.pitch() * rows);
    NegativeTests(host_alloc.ptr(), device_alloc.pitch(), device_alloc.ptr(), device_alloc.pitch(),
                  device_alloc.width(), device_alloc.height(), hipMemcpyDeviceToHost);
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, cols * rows * sizeof(int));
    NegativeTests(dst_alloc.ptr(), cols * sizeof(int), src_alloc.ptr(), cols * sizeof(int),
                  cols * sizeof(int), rows, hipMemcpyHostToHost);
  }

  SECTION("Device to device") {
    LinearAllocGuard2D<int> src_alloc(cols, rows);
    LinearAllocGuard2D<int> dst_alloc(cols, rows);
    NegativeTests(dst_alloc.ptr(), dst_alloc.pitch(), src_alloc.ptr(), src_alloc.pitch(),
                  dst_alloc.width(), dst_alloc.height(), hipMemcpyDeviceToDevice);
  }
}