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

template <typename T>
void MemcpyArrayCompare(T* const expected, T* const actual, const size_t num_elements) {
  const auto ret = std::mismatch(expected, expected + num_elements, actual);
  if (ret.first != expected + num_elements) {
    const auto idx = std::distance(expected, ret.first);
    INFO("Value mismatch at index: " << idx);
    REQUIRE(expected[idx] == actual[idx]);
  }
}


enum class LinearAllocations {
  malloc,
  mallocAndRegister,
  hipHostMalloc,
  hipMalloc,
  hipMallocManaged,
};

template <typename T> class LinearAllocationGuard {
 public:
  LinearAllocationGuard(const LinearAllocations allocation_type, const size_t size,
                        const unsigned int flags = 0u)
      : allocation_type_{allocation_type} {
    switch (allocation_type_) {
      case LinearAllocations::malloc:
        ptr_ = host_ptr_ = reinterpret_cast<T*>(malloc(size));
        break;
      case LinearAllocations::mallocAndRegister:
        host_ptr_ = reinterpret_cast<T*>(malloc(size));
        HIP_CHECK(hipHostRegister(host_ptr_, size, flags));
        HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&ptr_), host_ptr_, 0u));
        break;
      case LinearAllocations::hipHostMalloc:
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ptr_), size, flags));
        host_ptr_ = ptr_;
        break;
      case LinearAllocations::hipMalloc:
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr_), size));
        break;
      case LinearAllocations::hipMallocManaged:
        HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&ptr_), size, flags ? flags : 1u));
        host_ptr_ = ptr_;
    }
  }

  LinearAllocationGuard(const LinearAllocationGuard&) = delete;
  LinearAllocationGuard(LinearAllocationGuard&&) = delete;

  ~LinearAllocationGuard() {
    // No Catch macros, don't want to possibly throw in the destructor
    switch (allocation_type_) {
      case LinearAllocations::malloc:
        free(ptr_);
        break;
      case LinearAllocations::mallocAndRegister:
        hipHostUnregister(host_ptr_);
        free(host_ptr_);
        break;
      case LinearAllocations::hipHostMalloc:
        hipHostFree(ptr_);
        break;
      case LinearAllocations::hipMalloc:
      case LinearAllocations::hipMallocManaged:
        hipFree(ptr_);
    }
  }

  T* ptr() { return ptr_; };
  T* const ptr() const { return ptr_; };
  T* host_ptr() { return host_ptr_; }
  T* const host_ptr() const { return host_ptr(); }

 private:
  const LinearAllocations allocation_type_;
  T* ptr_ = nullptr;
  T* host_ptr_ = nullptr;
};

enum class Streams { nullstream, perThread, created };

class StreamGuard {
 public:
  StreamGuard(const Streams stream_type) : stream_type_{stream_type} {
    switch (stream_type_) {
      case Streams::nullstream:
        stream_ = nullptr;
        break;
      case Streams::perThread:
        stream_ = hipStreamPerThread;
        break;
      case Streams::created:
        HIP_CHECK(hipStreamCreate(&stream_));
    }
  }

  StreamGuard(const StreamGuard&) = delete;
  StreamGuard(StreamGuard&&) = delete;

  ~StreamGuard() {
    if (stream_type_ == Streams::created) {
      hipStreamDestroy(stream_);
    }
  }

  hipStream_t stream() const { return stream_; }

 private:
  const Streams stream_type_;
  hipStream_t stream_;
};

unsigned int GenerateLinearAllocationFlagCombinations(const LinearAllocations allocation_type) {
  switch (allocation_type) {
    case LinearAllocations::mallocAndRegister:
      // TODO
      return 0;
    case LinearAllocations::hipHostMalloc:
      // TODO
      return 0;
    case LinearAllocations::hipMallocManaged:
      // TODO
      return 1u;
    case LinearAllocations::malloc:
    case LinearAllocations::hipMalloc:
      return 0u;
  }
}

template <typename T>
__global__ void VectorIncrement(T* const vec, const T increment_value, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    vec[i] += increment_value;
  }
}

template <typename T> __global__ void VectorSet(T* const vec, const T value, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    vec[i] = value;
  }
}

template <typename T>
void ArrayFindIfNot(T* const array, const T expected_value, const size_t num_elements) {
  const auto it = std::find_if_not(array, array + num_elements, [expected_value](const int elem) {
    return expected_value == elem;
  });

  if (it != array + num_elements) {
    const auto idx = std::distance(array, it);
    INFO("Value mismatch at index " << idx);
    REQUIRE(expected_value == array[idx]);
  }
}

namespace {
constexpr size_t kPageSize = 4096;
}  // anonymous namespace

template <typename F>
void MemcpyDeviceToHostShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocations;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto host_allocation_type = GENERATE(LA::mallocAndRegister, LA::hipHostMalloc);
  const auto host_allocation_flags = GenerateLinearAllocationFlagCombinations(host_allocation_type);

  LinearAllocationGuard<int> host_allocation(host_allocation_type, allocation_size,
                                             host_allocation_flags);
  LinearAllocationGuard<int> device_allocation(LA::hipMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*device_allocation.ptr());
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 42;
  VectorSet<<<block_count, thread_count, 0, kernel_stream>>>(device_allocation.ptr(),
                                                             expected_value, element_count);
  HIP_CHECK(hipGetLastError());

  memcpy_func(host_allocation.host_ptr(), device_allocation.ptr(), allocation_size);

  ArrayFindIfNot(host_allocation.host_ptr(), expected_value, element_count);
}

template <typename F>
void MemcpyHostToDeviceShell(F memcpy_func, const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocations;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto host_allocation_type = GENERATE(LA::mallocAndRegister, LA::hipHostMalloc);
  const auto host_allocation_flags = GenerateLinearAllocationFlagCombinations(host_allocation_type);

  LinearAllocationGuard<int> host_allocation(host_allocation_type, allocation_size,
                                             host_allocation_flags);
  LinearAllocationGuard<int> device_allocation(LA::hipMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*device_allocation.ptr());
  constexpr int fill_value = 41;
  std::fill_n(host_allocation.host_ptr(), element_count, fill_value);

  memcpy_func(device_allocation.ptr(), host_allocation.host_ptr(), allocation_size);

  constexpr int increment_value = 1;
  constexpr int thread_count = 1024;
  const int block_count = element_count / thread_count + 1;
  VectorIncrement<<<block_count, thread_count, 0, kernel_stream>>>(device_allocation.ptr(),
                                                                   increment_value, element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(host_allocation.host_ptr(), device_allocation.ptr(), allocation_size,
                      hipMemcpyDeviceToHost));

  ArrayFindIfNot(host_allocation.host_ptr(), fill_value + increment_value, element_count);
}


int foo() {
  int i = GENERATE(1, 2, 3);
  int j = GENERATE(10, 20, 30);
  return i + j;
}

TEST_CASE("Blahem") {
  HIP_CHECK(hipSetDevice(0));
  int *p0;
  size_t size = 1024 * sizeof(*p0);
  HIP_CHECK(hipMalloc(&p0, size));
  HIP_CHECK(hipSetDevice(1));
  int *p1;
  HIP_CHECK(hipMalloc(&p1, size));
  HIP_CHECK(hipSetDevice(0));
  VectorSet<<<1, 1024>>>(p0, 42, 1024);
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemcpy(p1, p0, size, hipMemcpyDeviceToDevice));
  int *h;
  HIP_CHECK(hipHostMalloc(&h, size, 0));
  HIP_CHECK(hipMemcpy(h, p1, size, hipMemcpyDeviceToHost));
  ArrayFindIfNot(h, 42, 1024);
}

// A memory copy will succeed even if it is issued to a stream that is not associated to the current
// device

/*------------------------------------------------------------------------------------------------*/
// hipMemcpy
TEST_CASE("Unit_hipMemcpy_Basic") {
  SECTION("Device to host") {
    MemcpyDeviceToHostShell([](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyDeviceToHost));
      HIP_CHECK(hipStreamQuery(nullptr));
    });
  }

  SECTION("Device to host with default kind") {
    MemcpyDeviceToHostShell([](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyDefault));
      HIP_CHECK(hipStreamQuery(nullptr));
    });
  }

  SECTION("Host to device") {
    MemcpyHostToDeviceShell([](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyHostToDevice));
    });
  }

  SECTION("Host to device with default kind") {
    MemcpyHostToDeviceShell([](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpy(dst, src, count, hipMemcpyDefault));
    });
  }
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyAsync
TEST_CASE("Unit_hipMemcpyAsync_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  SECTION("Device to host") {
    const auto f = [stream](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    };
    MemcpyDeviceToHostShell(f, stream);
  }

  SECTION("Device to host with default kind") {
    const auto f = [stream](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpyAsync(dst, src, count, hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    };
    MemcpyDeviceToHostShell(f, stream);
  }

  SECTION("Host to device") {
    const auto f = [stream](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream));
    };
    MemcpyHostToDeviceShell(f, stream);
  }

  SECTION("Host to device with default kind") {
    const auto f = [stream](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpyAsync(dst, src, count, hipMemcpyDefault, stream));
    };
    MemcpyHostToDeviceShell(f, stream);
  }
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyWithStream
TEST_CASE("Unit_hipMemcpyWithStream_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  SECTION("Device to host") {
    const auto f = [stream](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpyWithStream(dst, src, count, hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamQuery(stream));
    };
    MemcpyDeviceToHostShell(f, stream);
  }

  SECTION("Device to host with default kind") {
    const auto f = [stream](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpyWithStream(dst, src, count, hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamQuery(stream));
    };
    MemcpyDeviceToHostShell(f, stream);
  }

  SECTION("Host to device") {
    const auto f = [stream](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpyWithStream(dst, src, count, hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipStreamQuery(stream));
    };
    MemcpyHostToDeviceShell(f, stream);
  }

  SECTION("Host to device with default kind") {
    const auto f = [stream](void* dst, void* src, size_t count) {
      HIP_CHECK(hipMemcpyWithStream(dst, src, count, hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamQuery(stream));
    };
    MemcpyHostToDeviceShell(f, stream);
  }
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyDtoH
TEST_CASE("Unit_hipMemcpyDtoH_Basic") {
  MemcpyDeviceToHostShell([](void* dst, void* src, size_t count) {
    HIP_CHECK(hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count));
    HIP_CHECK(hipStreamQuery(nullptr));
  });
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyHtoD
TEST_CASE("Unit_hipMemcpyHtoD_Basic") {
  MemcpyHostToDeviceShell([](void* dst, void* src, size_t count) {
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count));
  });
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyDtoD
TEST_CASE("Unit_hipMemcpyDtoD_Basic") {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto device_count = HipTest::getDeviceCount();
  const auto src_device = GENERATE_COPY(0);
  const auto dst_device = GENERATE_COPY(1);

  HIP_CHECK(hipSetDevice(src_device));
  LinearAllocationGuard<int> src_allocation(LinearAllocations::hipMalloc, allocation_size);
  HIP_CHECK(hipSetDevice(dst_device));
  LinearAllocationGuard<int> dst_allocation(LinearAllocations::hipMalloc, allocation_size);
  HIP_CHECK(hipSetDevice(src_device));
  LinearAllocationGuard<int> result(LinearAllocations::hipHostMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*src_allocation.ptr());
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 42;
  VectorSet<<<block_count, thread_count>>>(src_allocation.ptr(), expected_value, element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipSetDevice(dst_device));
  HIP_CHECK(hipMemcpy(dst_allocation.ptr(), src_allocation.ptr(), allocation_size, hipMemcpyDeviceToDevice));

  HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_allocation.ptr(), allocation_size, hipMemcpyDeviceToHost));

  ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyDtoHAsync
TEST_CASE("Unit_hipMemcpyDtoHAsync_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  const auto f = [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
    HIP_CHECK(hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  };
  MemcpyDeviceToHostShell(f, stream_guard.stream());
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyHtoDAsync
TEST_CASE("Unit_hipMemcpyHtoDAsync_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  const auto f = [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
    HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, stream));
  };
  MemcpyHostToDeviceShell(f, stream_guard.stream());
}
/*------------------------------------------------------------------------------------------------*/