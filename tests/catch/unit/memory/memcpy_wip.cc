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

#include <chrono>
#include <functional>

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


enum class LinearAllocs {
  malloc,
  mallocAndRegister,
  hipHostMalloc,
  hipMalloc,
  hipMallocManaged,
};

template <typename T> class LinearAllocGuard {
 public:
  LinearAllocGuard(const LinearAllocs allocation_type, const size_t size,
                   const unsigned int flags = 0u)
      : allocation_type_{allocation_type} {
    switch (allocation_type_) {
      case LinearAllocs::malloc:
        ptr_ = host_ptr_ = reinterpret_cast<T*>(malloc(size));
        break;
      case LinearAllocs::mallocAndRegister:
        host_ptr_ = reinterpret_cast<T*>(malloc(size));
        HIP_CHECK(hipHostRegister(host_ptr_, size, flags));
        HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&ptr_), host_ptr_, 0u));
        break;
      case LinearAllocs::hipHostMalloc:
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&ptr_), size, flags));
        host_ptr_ = ptr_;
        break;
      case LinearAllocs::hipMalloc:
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr_), size));
        break;
      case LinearAllocs::hipMallocManaged:
        HIP_CHECK(hipMallocManaged(reinterpret_cast<void**>(&ptr_), size, flags ? flags : 1u));
        host_ptr_ = ptr_;
    }
  }

  LinearAllocGuard(const LinearAllocGuard&) = delete;
  LinearAllocGuard(LinearAllocGuard&&) = delete;

  ~LinearAllocGuard() {
    // No Catch macros, don't want to possibly throw in the destructor
    switch (allocation_type_) {
      case LinearAllocs::malloc:
        free(ptr_);
        break;
      case LinearAllocs::mallocAndRegister:
        hipHostUnregister(host_ptr_);
        free(host_ptr_);
        break;
      case LinearAllocs::hipHostMalloc:
        hipHostFree(ptr_);
        break;
      case LinearAllocs::hipMalloc:
      case LinearAllocs::hipMallocManaged:
        hipFree(ptr_);
    }
  }

  T* ptr() { return ptr_; };
  T* const ptr() const { return ptr_; };
  T* host_ptr() { return host_ptr_; }
  T* const host_ptr() const { return host_ptr(); }

 private:
  const LinearAllocs allocation_type_;
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

unsigned int GenerateLinearAllocationFlagCombinations(const LinearAllocs allocation_type) {
  switch (allocation_type) {
    case LinearAllocs::mallocAndRegister:
      // TODO
      return 0;
    case LinearAllocs::hipHostMalloc:
      return GENERATE(hipHostMallocDefault, hipHostMallocPortable, hipHostMallocMapped,
                      hipHostMallocWriteCombined);
    case LinearAllocs::hipMallocManaged:
      // TODO
      return 1u;
    case LinearAllocs::malloc:
    case LinearAllocs::hipMalloc:
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

// Will execute for atleast interval milliseconds
__global__ void Delay(uint32_t interval, const uint32_t ticks_per_ms) {
  while (interval--) {
    uint64_t start = clock();
    while (clock() - start < ticks_per_ms) {
    }
  }
}

void LaunchDelayKernel(const std::chrono::milliseconds interval, const hipStream_t stream) {
  int ticks_per_ms = 0;
  // Clock rate is in kHz => number of clock ticks in a millisecond
  HIP_CHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeClockRate, 0));
  Delay<<<1, 1, 0, stream>>>(interval.count(), ticks_per_ms);
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
void MemcpyDeviceToHostShell(F memcpy_func, const bool should_synchronize,
                             const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto host_allocation_type = GENERATE(LA::mallocAndRegister, LA::hipHostMalloc);
  const auto host_allocation_flags = GenerateLinearAllocationFlagCombinations(host_allocation_type);

  LinearAllocGuard<int> host_allocation(host_allocation_type, allocation_size,
                                        host_allocation_flags);
  LinearAllocGuard<int> device_allocation(LA::hipMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*device_allocation.ptr());
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 42;
  VectorSet<<<block_count, thread_count, 0, kernel_stream>>>(device_allocation.ptr(),
                                                             expected_value, element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(memcpy_func(host_allocation.host_ptr(), device_allocation.ptr(), allocation_size));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  ArrayFindIfNot(host_allocation.host_ptr(), expected_value, element_count);
}

template <typename F>
void MemcpyHostToDeviceShell(F memcpy_func, const bool should_synchronize,
                             const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto host_allocation_type = GENERATE(LA::mallocAndRegister, LA::hipHostMalloc);
  const auto host_allocation_flags = GenerateLinearAllocationFlagCombinations(host_allocation_type);

  LinearAllocGuard<int> host_allocation(host_allocation_type, allocation_size,
                                        host_allocation_flags);
  LinearAllocGuard<int> device_allocation(LA::hipMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*device_allocation.ptr());
  constexpr int fill_value = 41;
  std::fill_n(host_allocation.host_ptr(), element_count, fill_value);

  HIP_CHECK(memcpy_func(device_allocation.ptr(), host_allocation.host_ptr(), allocation_size));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

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

template <typename F>
void MemcpyHostToHostShell(F memcpy_func, const bool should_synchronize,
                           const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto src_allocation_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto dst_allocation_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto src_allocation_flags = GenerateLinearAllocationFlagCombinations(src_allocation_type);
  const auto dst_allocation_flags = GenerateLinearAllocationFlagCombinations(dst_allocation_type);

  LinearAllocGuard<int> src_allocation(src_allocation_type, allocation_size, src_allocation_flags);
  LinearAllocGuard<int> dst_allocation(dst_allocation_type, allocation_size, dst_allocation_flags);

  const auto element_count = allocation_size / sizeof(*src_allocation.host_ptr());
  constexpr auto expected_value = 42;
  std::fill_n(src_allocation.host_ptr(), element_count, expected_value);

  HIP_CHECK(memcpy_func(dst_allocation.host_ptr(), src_allocation.host_ptr(), allocation_size));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  ArrayFindIfNot(dst_allocation.host_ptr(), expected_value, element_count);
}

template <typename F>
void MemcpyDeviceToDeviceShell(F memcpy_func, const bool should_synchronize,
                               const hipStream_t kernel_stream = nullptr) {
  const auto allocation_size = GENERATE(kPageSize / 2, kPageSize, kPageSize * 2);
  const auto device_count = HipTest::getDeviceCount();
  // Waiting to figure out what the issue is when devices are different
  const auto src_device = GENERATE_COPY(0);
  const auto dst_device = GENERATE_COPY(0);

  HIP_CHECK(hipSetDevice(src_device));
  LinearAllocGuard<int> src_allocation(LinearAllocs::hipMalloc, allocation_size);
  LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, allocation_size, hipHostMallocPortable);
  HIP_CHECK(hipSetDevice(dst_device));
  LinearAllocGuard<int> dst_allocation(LinearAllocs::hipMalloc, allocation_size);

  const auto element_count = allocation_size / sizeof(*src_allocation.ptr());
  constexpr auto thread_count = 1024;
  const auto block_count = element_count / thread_count + 1;
  constexpr int expected_value = 42;
  // Consider situations when this is dst_device instead
  HIP_CHECK(hipSetDevice(src_device));
  VectorSet<<<block_count, thread_count, 0, kernel_stream>>>(src_allocation.ptr(), expected_value,
                                                             element_count);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(memcpy_func(dst_allocation.ptr(), src_allocation.ptr(), allocation_size));
  if (should_synchronize) {
    HIP_CHECK(hipStreamSynchronize(kernel_stream));
  }

  HIP_CHECK(
      hipMemcpy(result.host_ptr(), dst_allocation.ptr(), allocation_size, hipMemcpyDeviceToHost));

  ArrayFindIfNot(result.host_ptr(), expected_value, element_count);
}

template <typename F>
void MemcpyWithDirectionCommonTests(F memcpy_func, const bool should_synchronize) {
  using namespace std::placeholders;
  SECTION("Device to host") {
    MemcpyDeviceToHostShell(std::bind(memcpy_func, _1, _2, _3, hipMemcpyDeviceToHost),
                            should_synchronize);
  }

  SECTION("Device to host with default kind") {
    MemcpyDeviceToHostShell(std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault),
                            should_synchronize);
  }

  SECTION("Host to device") {
    MemcpyHostToDeviceShell(std::bind(memcpy_func, _1, _2, _3, hipMemcpyHostToDevice),
                            should_synchronize);
  }

  SECTION("Host to device with default kind") {
    MemcpyHostToDeviceShell(std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault),
                            should_synchronize);
  }

  SECTION("Host to host") {
    MemcpyHostToHostShell(std::bind(memcpy_func, _1, _2, _3, hipMemcpyHostToHost),
                          should_synchronize);
  }

  SECTION("Host to host with default kind") {
    MemcpyHostToHostShell(std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault), should_synchronize);
  }

  SECTION("Device to device") {
    MemcpyDeviceToDeviceShell(std::bind(memcpy_func, _1, _2, _3, hipMemcpyDeviceToDevice),
                              should_synchronize);
  }

  SECTION("Device to device with default kind") {
    MemcpyDeviceToDeviceShell(std::bind(memcpy_func, _1, _2, _3, hipMemcpyDefault),
                              should_synchronize);
  }
}

template <typename F>
void MemcpySyncBehaviorCheck(F memcpy_func, const bool should_sync,
                             const hipStream_t kernel_stream) {
  LaunchDelayKernel(std::chrono::milliseconds{100}, kernel_stream);
  HIP_CHECK(memcpy_func());
  if (should_sync) {
    HIP_CHECK(hipStreamQuery(kernel_stream));
  } else {
    HIP_CHECK_ERROR(hipStreamQuery(kernel_stream), hipErrorNotReady);
  }
}

template <typename F>
void MemcpyHtoDSyncBehavior(F memcpy_func, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto host_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  LinearAllocGuard<int> host_alloc(host_alloc_type, kPageSize);
  LinearAllocGuard<int> device_alloc(LA::hipMalloc, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, device_alloc.ptr(), host_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F>
void MemcpyDtoHPageableSyncBehavior(F memcpy_func, const bool should_sync,
                                    const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard<int> host_alloc(LinearAllocs::malloc, kPageSize);
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, host_alloc.ptr(), device_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F>
void MemcpyDtoHPinnedSyncBehavior(F memcpy_func, const bool should_sync,
                                  const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, host_alloc.ptr(), device_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F>
void MemcpyDtoDSyncBehavior(F memcpy_func, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, dst_alloc.ptr(), src_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F>
void MemcpyHtoHSyncBehavior(F memcpy_func, const bool should_sync,
                            const hipStream_t kernel_stream = nullptr) {
  using LA = LinearAllocs;
  const auto src_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);
  const auto dst_alloc_type = GENERATE(LA::malloc, LA::hipHostMalloc);

  LinearAllocGuard<int> src_alloc(src_alloc_type, kPageSize);
  LinearAllocGuard<int> dst_alloc(dst_alloc_type, kPageSize);
  MemcpySyncBehaviorCheck(std::bind(memcpy_func, dst_alloc.ptr(), src_alloc.ptr(), kPageSize),
                          should_sync, kernel_stream);
}

template <typename F> void MemcpyCommonNegativeTests(F f, void* dst, void* src, size_t count) {
  SECTION("dst == nullptr") { HIP_CHECK_ERROR(f(nullptr, src, count), hipErrorInvalidValue); }
  SECTION("src == nullptr") { HIP_CHECK_ERROR(f(dst, nullptr, count), hipErrorInvalidValue); }
}

TEST_CASE("D2D_Fail") {
  // Adapted from hipMemcpy.cc:Unit_hipMemcpy_H2H-H2D-D2H-H2PinMem:H2D-D2D-D2H-PeerGPU
  constexpr auto num_elements = 1024;
  constexpr auto size = num_elements * sizeof(int);
  HIP_CHECK(hipSetDevice(0));
  int* h = reinterpret_cast<int*>(malloc(size));
  std::fill_n(h, num_elements, 42);
  int* r = reinterpret_cast<int*>(malloc(size));
  std::fill_n(r, num_elements, 0);
  REQUIRE(0 == r[0]);
  int* p0 = nullptr;
  HIP_CHECK(hipMalloc(&p0, size));

  HIP_CHECK(hipSetDevice(1));
  int* p1 = nullptr;
  HIP_CHECK(hipMalloc(&p1, size));

  HIP_CHECK(hipMemcpy(p0, h, size, hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(p1, p0, size, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipMemcpy(r, p1, size, hipMemcpyDeviceToHost));

  REQUIRE(42 == r[0]);
}

// A memory copy will succeed even if it is issued to a stream that is not associated to the current
// device

/*------------------------------------------------------------------------------------------------*/
// hipMemcpy
TEST_CASE("Unit_hipMemcpy_Basic") { MemcpyWithDirectionCommonTests(hipMemcpy, false); }

TEST_CASE("Unit_hipMemcpy_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host memory to device memory") {
    // For transfers from pageable host memory to device memory, a stream sync is performed before
    // the copy is initiated. The function will return once the pageable buffer has been copied to
    // the staging memory for DMA transfer to device memory, but the DMA to final destination may
    // not have completed.
    // For transfers from pinned host memory to device memory, the function is synchronous with
    // respect to the host
    MemcpyHtoDSyncBehavior(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToDevice), true);
  }

  // For transfers from device to either pageable or pinned host memory, the function returns only
  // once the copy has completed
  SECTION("Device memory to host memory") {
    const auto f = std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToHost);
    MemcpyDtoHPageableSyncBehavior(f, true);
    MemcpyDtoHPinnedSyncBehavior(f, true);
  }

  // For transfers from device memory to device memory, no host-side synchronization is performed.
  SECTION("Device memory to device memory") {
    // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
    // respect to the host
#if HT_AMD
    HipTest::HIP_SKIP_TEST(
        "EXSWCPHIPT-127 - Memcpy from device to device memory behavior differs on AMD and Nvidia");
    return;
#endif
    MemcpyDtoDSyncBehavior(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToDevice), false);
  }

  // For transfers from any host memory to any host memory, the function is fully synchronous with
  // respect to the host
  SECTION("Host memory to host memory") {
    MemcpyHtoHSyncBehavior(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToHost), true);
  }
}


TEST_CASE("Unit_hipMemcpy_Negative_Parameters") {
  using namespace std::placeholders;

  SECTION("Host to device") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToDevice),
                              device_alloc.ptr(), host_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpy(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                static_cast<hipMemcpyKind>(-1)),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Device to host") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToHost),
                              host_alloc.ptr(), device_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpy(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                static_cast<hipMemcpyKind>(-1)),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyHostToHost),
                              dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(
          hipMemcpy(dst_alloc.ptr(), src_alloc.ptr(), kPageSize, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidValue);
    }
  }

  SECTION("Device to device") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpy, _1, _2, _3, hipMemcpyDeviceToDevice),
                              dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(
          hipMemcpy(dst_alloc.ptr(), src_alloc.ptr(), kPageSize, static_cast<hipMemcpyKind>(-1)),
          hipErrorInvalidValue);
    }
  }
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyAsync
TEST_CASE("Unit_hipMemcpyAsync_Basic") {
  using namespace std::placeholders;
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  MemcpyWithDirectionCommonTests(std::bind(hipMemcpyAsync, _1, _2, _3, _4, stream), true);
}


TEST_CASE("Unit_hipMemcpyAsync_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host memory to device memory") {
    // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
    // respect to the host
#if HT_AMD
    HipTest::HIP_SKIP_TEST(
        "EXSWCPHIPT-127 - MemcpyAsync from host to device memory behavior differs on AMD and "
        "Nvidia");
    return;
#endif
    MemcpyHtoDSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr),
                           false);
  }

  // For transfers from device memory to pageable host memory, the function will return only once
  // the copy has completed
  SECTION("Device memory to pageable host memory") {
    MemcpyDtoHPageableSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr), true);
  }

  // For all other transfers, the function is fully asynchronous. If pageable memory must first be
  // staged to pinned memory, this will be handled asynchronously with a worker thread
  SECTION("Device memory to pinned host memory") {
    MemcpyDtoHPinnedSyncBehavior(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr), false);
  }

  SECTION("Device memory to device memory") {
    MemcpyDtoDSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr),
                           false);
  }

  // For transfers from any host memory to any host memory, the function is fully synchronous with
  // respect to the host
  SECTION("Host memory to host memory") {
    MemcpyHtoHSyncBehavior(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                           true);
  }
}

TEST_CASE("Unit_hipMemcpyAsync_Negative_Parameters") {
  using namespace std::placeholders;

  SECTION("Host to device") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToDevice, nullptr),
                              device_alloc.ptr(), host_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyAsync(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                     hipMemcpyHostToDevice, stream),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Device to host") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToHost, nullptr),
                              host_alloc.ptr(), device_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyAsync(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                     hipMemcpyDeviceToHost, stream),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                              dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(
          hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize, hipMemcpyHostToHost, stream),
          hipErrorInvalidValue);
    }
  }

  SECTION("Device to device") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    MemcpyCommonNegativeTests(
        std::bind(hipMemcpyAsync, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr), dst_alloc.ptr(),
        src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyAsync(src_alloc.ptr(), dst_alloc.ptr(), kPageSize,
                                     static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyAsync(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                     hipMemcpyDeviceToDevice, stream),
                      hipErrorInvalidValue);
    }
  }
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyWithStream
TEST_CASE("Unit_hipMemcpyWithStream_Basic") {
  using namespace std::placeholders;
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);
  const hipStream_t stream = stream_guard.stream();

  MemcpyWithDirectionCommonTests(std::bind(hipMemcpyWithStream, _1, _2, _3, _4, stream), false);
}

TEST_CASE("Unit_hipMemcpyWithStream_Synchronization_Behavior") {
  using namespace std::placeholders;
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host memory to device memory") {
    MemcpyHtoDSyncBehavior(
        std::bind(hipMemcpyWithStream, _1, _2, _3, hipMemcpyHostToDevice, nullptr), true);
  }

  SECTION("Device memory to host memory") {
    const auto f = std::bind(hipMemcpyWithStream, _1, _2, _3, hipMemcpyDeviceToHost, nullptr);
    MemcpyDtoHPageableSyncBehavior(f, true);
    MemcpyDtoHPinnedSyncBehavior(f, true);
  }

  SECTION("Device memory to device memory") {
    // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
    // respect to the host
#if HT_AMD
    HipTest::HIP_SKIP_TEST(
        "EXSWCPHIPT-127 - Memcpy from device to device memory behavior differs on AMD and Nvidia");
    return;
#endif
    MemcpyDtoDSyncBehavior(
        std::bind(hipMemcpyWithStream, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr), false);
  }

  SECTION("Host memory to host memory") {
    MemcpyHtoHSyncBehavior(std::bind(hipMemcpyWithStream, _1, _2, _3, hipMemcpyHostToHost, nullptr),
                           true);
  }
}

TEST_CASE("Unit_hipMemcpyWithStream_Negative_Parameters") {
  using namespace std::placeholders;

  SECTION("Host to device") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(
        std::bind(hipMemcpyWithStream, _1, _2, _3, hipMemcpyHostToDevice, nullptr),
        device_alloc.ptr(), host_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyWithStream(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                          static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyWithStream(device_alloc.ptr(), host_alloc.ptr(), kPageSize,
                                          hipMemcpyHostToDevice, stream),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Device to host") {
    LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(
        std::bind(hipMemcpyWithStream, _1, _2, _3, hipMemcpyDeviceToHost, nullptr),
        host_alloc.ptr(), device_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyWithStream(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                          static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyWithStream(host_alloc.ptr(), device_alloc.ptr(), kPageSize,
                                          hipMemcpyDeviceToHost, stream),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Host to host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc, kPageSize);

    MemcpyCommonNegativeTests(
        std::bind(hipMemcpyWithStream, _1, _2, _3, hipMemcpyHostToHost, nullptr), dst_alloc.ptr(),
        src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyWithStream(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                          static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyWithStream(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                          hipMemcpyHostToHost, stream),
                      hipErrorInvalidValue);
    }
  }

  SECTION("Device to device") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

    MemcpyCommonNegativeTests(
        std::bind(hipMemcpyWithStream, _1, _2, _3, hipMemcpyDeviceToDevice, nullptr),
        dst_alloc.ptr(), src_alloc.ptr(), kPageSize);

    SECTION("Invalid MemcpyKind") {
      HIP_CHECK_ERROR(hipMemcpyWithStream(src_alloc.ptr(), dst_alloc.ptr(), kPageSize,
                                          static_cast<hipMemcpyKind>(-1), nullptr),
                      hipErrorInvalidValue);
    }

    SECTION("Invalid stream") {
      hipStream_t stream;
      HIP_CHECK_ERROR(hipMemcpyWithStream(dst_alloc.ptr(), src_alloc.ptr(), kPageSize,
                                          hipMemcpyDeviceToDevice, stream),
                      hipErrorInvalidValue);
    }
  }
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyDtoH
TEST_CASE("Unit_hipMemcpyDtoH_Basic") {
  MemcpyDeviceToHostShell(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      false);
}

TEST_CASE("Unit_hipMemcpyDtoH_Synchronization_Behavior") {
  const auto f = [](void* dst, void* src, size_t count) {
    return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
  };
  MemcpyDtoHPageableSyncBehavior(f, true);
  MemcpyDtoHPinnedSyncBehavior(f, true);
}

TEST_CASE("Unit_hipMemcpyDtoH_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoH(dst, reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      host_alloc.ptr(), device_alloc.ptr(), kPageSize);
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyHtoD
TEST_CASE("Unit_hipMemcpyHtoD_Basic") {
  MemcpyHostToDeviceShell(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      false);
}

TEST_CASE("Unit_hipMemcpyHtoD_Synchronization_Behavior") {
  MemcpyHtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      true);
}

TEST_CASE("Unit_hipMemcpyHtoD_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(dst), src, count);
      },
      device_alloc.ptr(), host_alloc.ptr(), kPageSize);
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyDtoD
TEST_CASE("Unit_hipMemcpyDtoD_Basic") {
  SECTION("Device to device") {
    MemcpyDeviceToDeviceShell(
        [](void* dst, void* src, size_t count) {
          return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                               reinterpret_cast<hipDeviceptr_t>(src), count);
        },
        false);
  }
}

TEST_CASE("Unit_hipMemcpyDtoD_Synchronization_Behavior") {
  // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
  // respect to the host
#if HT_AMD
  HipTest::HIP_SKIP_TEST(
      "EXSWCPHIPT-127 - Memcpy from device to device memory behavior differs on AMD and Nvidia");
  return;
#endif
  MemcpyDtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                             reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      false);
}

TEST_CASE("Unit_hipMemcpyDtoD_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(dst),
                             reinterpret_cast<hipDeviceptr_t>(src), count);
      },
      dst_alloc.ptr(), src_alloc.ptr(), kPageSize);
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyDtoHAsync
TEST_CASE("Unit_hipMemcpyDtoHAsync_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  const auto f = [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
    return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, stream);
  };
  MemcpyDeviceToHostShell(f, true, stream_guard.stream());
}

TEST_CASE("Unit_hipMemcpyDtoHAsync_Synchronization_Behavior") {
  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Device memory to pageable host memory") {
    MemcpyDtoHPageableSyncBehavior(
        [](void* dst, void* src, size_t count) {
          return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
        },
        true);
  }

  SECTION("Device memory to pinned host memory") {
    MemcpyDtoHPinnedSyncBehavior(
        [](void* dst, void* src, size_t count) {
          return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
        },
        false);
  }
}

TEST_CASE("Unit_hipMemcpyDtoHAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoHAsync(dst, reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      host_alloc.ptr(), device_alloc.ptr(), kPageSize);
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyHtoDAsync
TEST_CASE("Unit_hipMemcpyHtoDAsync_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  const auto f = [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
    return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, stream);
  };
  MemcpyHostToDeviceShell(f, true, stream_guard.stream());
}

TEST_CASE("Unit_hipMemcpyHtoDAsync_Synchronization_Behavior") {
  // This behavior differs on NVIDIA and AMD, on AMD the hipMemcpy calls is synchronous with
  // respect to the host
#if HT_AMD
  HipTest::HIP_SKIP_TEST(
      "EXSWCPHIPT-127 - MemcpyAsync from host to device memory behavior differs on AMD and "
      "Nvidia");
  return;
#endif
  MemcpyHtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, nullptr);
      },
      false);
}

TEST_CASE("Unit_hipMemcpyHtoDAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> device_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> host_alloc(LinearAllocs::hipHostMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst), src, count, nullptr);
      },
      device_alloc.ptr(), host_alloc.ptr(), kPageSize);
}
/*------------------------------------------------------------------------------------------------*/


/*------------------------------------------------------------------------------------------------*/
// hipMemcpyDtoDAsync
TEST_CASE("Unit_hipMemcpyDtoDAsync_Basic") {
  const auto stream_type = GENERATE(Streams::nullstream, Streams::perThread, Streams::created);
  const StreamGuard stream_guard(stream_type);

  SECTION("Device to device") {
    MemcpyDeviceToDeviceShell(
        [stream = stream_guard.stream()](void* dst, void* src, size_t count) {
          return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                    reinterpret_cast<hipDeviceptr_t>(src), count, stream);
        },
        true);
  }
}

TEST_CASE("Unit_hipMemcpyDtoDAsync_Synchronization_Behavior") {
  MemcpyDtoDSyncBehavior(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                  reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      false);
}

TEST_CASE("Unit_hipMemcpyDtoDAsync_Negative_Parameters") {
  using namespace std::placeholders;
  LinearAllocGuard<int> src_alloc(LinearAllocs::hipMalloc, kPageSize);
  LinearAllocGuard<int> dst_alloc(LinearAllocs::hipMalloc, kPageSize);

  MemcpyCommonNegativeTests(
      [](void* dst, void* src, size_t count) {
        return hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(dst),
                                  reinterpret_cast<hipDeviceptr_t>(src), count, nullptr);
      },
      dst_alloc.ptr(), src_alloc.ptr(), kPageSize);
}
/*------------------------------------------------------------------------------------------------*/