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

TEST_CASE("Unit_hipMemcpyParam2DAsync_Positive_Basic") {
  using namespace std::placeholders;
  constexpr bool async = true;

  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  SECTION("Device to Host") {
    Memcpy2DDeviceToHostShell<async>(std::bind(MemcpyParam2DAdapter<hipMemcpyDeviceToHost, async>(),
                                               _1, _2, _3, _4, _5, _6, _7, stream));
  }

  SECTION("Device to Device") {
    SECTION("Peer access disabled") {
      Memcpy2DDeviceToDeviceShell<async, false>(
          std::bind(MemcpyParam2DAdapter<hipMemcpyDeviceToDevice, async>(), _1, _2, _3, _4, _5, _6,
                    _7, stream));
    }
    SECTION("Peer access enabled") {
      Memcpy2DDeviceToDeviceShell<async, true>(
          std::bind(MemcpyParam2DAdapter<hipMemcpyDeviceToDevice, async>(), _1, _2, _3, _4, _5, _6,
                    _7, stream));
    }
  }

  SECTION("Host to Device") {
    Memcpy2DHostToDeviceShell<async>(std::bind(MemcpyParam2DAdapter<hipMemcpyHostToDevice, async>(),
                                               _1, _2, _3, _4, _5, _6, _7, stream));
  }

  SECTION("Host to Host") {
    Memcpy2DHostToHostShell<async>(std::bind(MemcpyParam2DAdapter<hipMemcpyHostToHost, async>(), _1,
                                             _2, _3, _4, _5, _6, _7, stream));
  }
}