/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Testcase Description:
// 1) Verifies the working of Memcpy2D API negative scenarios by
//    Pass NULL to destination pointer
//    Pass NULL to Source pointer
//    Pass width greater than spitch/dpitch
// 2) Verifies hipMemcpy2D API by
//    pass 0 to destionation pitch
//    pass 0 to source pitch
//    pass 0 to width
//    pass 0 to height
// 3) Verifies working of Memcpy2D API on host memory and pinned host memory by
//    performing D2H, D2D and H2D memory kind copies on same GPU
// 4) Verifies working of Memcpy2D API for the following scenarios
//      H2D-D2D-D2H on host and device memory
//      H2D-D2D-D2H on pinned host and device memory
//      H2D-D2D-D2H functionalities where memory is allocated in GPU-0
//      and API is triggered from GPU-1

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <resource_guards.hh>

static constexpr auto NUM_W{16};
static constexpr auto NUM_H{16};
static constexpr auto COLUMNS{8};
static constexpr auto ROWS{8};

/*
This testcases performs the following scenarios of hipMemcpy2D API on same GPU
1. H2D-D2D-D2H for Host Memory<-->Device Memory
2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory

Input : "A_h" initialized based on data type
         "A_h" --> "A_d" using H2D copy
         "A_d" --> "B_d" using D2D copy
         "B_d" --> "B_h" using D2H copy
Output: Validating A_h with B_h both should be equal for
        the number of COLUMNS and ROWS copied
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpy2D_H2D-D2D-D2H", ""
                   , int, float, double) {
  // 1 refers to pinned host memory
  auto mem_type = GENERATE(0, 1);
  HIP_CHECK(hipSetDevice(0));
  TestType  *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr},
            *B_d{nullptr};
  size_t pitch_A, pitch_B;
  size_t width{NUM_W * sizeof(TestType)};

  // Allocating memory
  if (mem_type) {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, &B_h, &C_h, NUM_W*NUM_H, true);
  } else {
    HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
                                  &A_h, &B_h, &C_h, NUM_W*NUM_H, false);
  }
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
                          &pitch_A, width, NUM_H));
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&B_d),
                          &pitch_B, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, B_h, C_h);

  // Host to Device
  HIP_CHECK(hipMemcpy2D(A_d, pitch_A, A_h, COLUMNS*sizeof(TestType),
                        COLUMNS*sizeof(TestType), ROWS, hipMemcpyHostToDevice));

  // Performs D2D on same GPU device
  HIP_CHECK(hipMemcpy2D(B_d, pitch_B, A_d,
                        pitch_A, COLUMNS*sizeof(TestType),
                        ROWS, hipMemcpyDeviceToDevice));

  // hipMemcpy2D Device to Host
  HIP_CHECK(hipMemcpy2D(B_h, COLUMNS*sizeof(TestType), B_d, pitch_B,
                        COLUMNS*sizeof(TestType), ROWS,
                        hipMemcpyDeviceToHost));

  // Validating the result
  REQUIRE(HipTest::checkArray<TestType>(A_h, B_h, COLUMNS, ROWS) == true);


  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  if (mem_type) {
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                  A_h, B_h, C_h, true);
  } else {
    HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
                                  A_h, B_h, C_h, false);
  }
}


template <typename T> __global__ void blahem(T* const out, size_t pitch, size_t w, size_t h, size_t d) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  const auto z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < w && y < h && z < d) {
    char *const slice = reinterpret_cast<char*>(out) + pitch * h * z;
    char* const row = slice + pitch * y;
    reinterpret_cast<T*>(row)[x] = z * w * h + y * w + x;
  }
}

TEST_CASE("Moo") {
  constexpr size_t width = 2048;
  constexpr size_t height = width;
  constexpr size_t depth = 5;
  hipPitchedPtr pitched_ptr;
  const hipExtent extent{width * sizeof(int), height, depth};
  HIP_CHECK(hipMalloc3D(&pitched_ptr, extent));
  const dim3 threads_per_block(32, 32);
  const dim3 blocks(width / threads_per_block.x + 1, height / threads_per_block.y + 1, depth);
  blahem<<<blocks, threads_per_block>>>(reinterpret_cast<int*>(pitched_ptr.ptr), pitched_ptr.pitch,
                                        width, height, depth);
  HIP_CHECK(hipGetLastError());
  std::vector<int> arr(width * height * depth);
  hipMemcpy3DParms params = {0};
  params.dstPtr = make_hipPitchedPtr(arr.data(), width * sizeof(int), width * sizeof(int), height);
  params.srcPtr = pitched_ptr;
  params.extent = extent;
  params.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&params));
  for (int i = 0; i < arr.size(); ++i) {
    if (arr[i] != i) {
      std::cout << "Wraeng at index: " << i << std::endl;
    }
  }
}

TEST_CASE("Mlahem") {
  constexpr size_t N = 2048;
  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, N*sizeof(int));
  const auto threads_per_block = 1024;
  const auto blocks = N / threads_per_block + 1;
  blahem<<<blocks, threads_per_block>>>(alloc.ptr(), N*sizeof(int), N, 1, 1);
  HIP_CHECK(hipGetLastError());
  std::vector<int> arr(N);
  HIP_CHECK(hipMemcpy(arr.data(), alloc.ptr(), N*sizeof(int), hipMemcpyDeviceToHost));
  for(int i = 0; i < arr.size(); ++i) {
    if(arr[i] != i) {
      std::cout << "Wraeng at index: " << i << std::endl;
    }
  }
}

TEST_CASE("Blahem") {
  int* ptr = nullptr;
  size_t pitch = 0;
  constexpr size_t width = 2048;
  constexpr size_t height = width;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, width * sizeof(*ptr), height));
  dim3 threads_per_block(32, 32);
  dim3 blocks(width / threads_per_block.x + 1, height / threads_per_block.y + 1);
  blahem<<<blocks, threads_per_block>>>(ptr, pitch, width, height, 1);
  HIP_CHECK(hipGetLastError());
  std::vector<int> arr(width*height);
  HIP_CHECK(hipMemcpy2D(arr.data(), width * sizeof(*ptr), ptr, pitch, width * sizeof(*ptr), height,
                        hipMemcpyDeviceToHost));
  for(int i = 0; i < arr.size(); ++i) {
    if(arr[i] != i) {
      std::cout << "Wraeng at index: " << i << std::endl;
    }
  }
  // int one[2][6] = {{0, 1, 2, 3, -1, -1}, {4, 5, 6, 7, -1, -1}};
  // int two [8];
  // HIP_CHECK(hipMemcpy2D(two, 4*sizeof(int), one, 6*sizeof(int), 4*sizeof(int), 2, hipMemcpyHostToHost));
  // for(int i = 0; i < 8; ++i) {
  //   std::cout << two[i] << " ";
  // }
  // std::cout << std::endl;
}

/*
This testcases performs the following scenarios of hipMemcpy2D API on Peer GPU
1. H2D-D2D-D2H for Host Memory<-->Device Memory
2. H2D-D2D-D2H for Pinned Host Memory<-->Device Memory
3. Device context change where memory is allocated in GPU-0
   and API is trigerred from GPU-1

Input : "A_h" initialized based on data type
         "A_h" --> "A_d" using H2D copy
         "A_d" --> "X_d" using D2D copy
         "X_d" --> "B_h" using D2H copy
Output: Validating A_h with B_h both should be equal for
        the number of COLUMNS and ROWS copied
*/
TEMPLATE_TEST_CASE("Unit_hipMemcpy2D_multiDevice-D2D", ""
                   , int, float, double) {
  auto mem_type = GENERATE(0, 1);
  int numDevices = 0;
  int canAccessPeer = 0;
  TestType* A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(TestType)};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer) {
      HIP_CHECK(hipSetDevice(0));

      // Allocating memory
      if (mem_type) {
        HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
            &A_h, &B_h, &C_h, NUM_W*NUM_H, true);
      } else {
        HipTest::initArrays<TestType>(nullptr, nullptr, nullptr,
            &A_h, &B_h, &C_h, NUM_W*NUM_H, false);
      }
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
            &pitch_A, width, NUM_H));

      // Initialize the data
      HipTest::setDefaultData<TestType>(NUM_W*NUM_H, A_h, B_h, C_h);

      char *X_d{nullptr};
      size_t pitch_X;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&X_d),
                               &pitch_X, width, NUM_H));

      // Change device
      HIP_CHECK(hipSetDevice(1));

      // Host to Device
      std::cout << COLUMNS * sizeof(TestType) << std::endl;
      std::cout << pitch_A << std::endl;
      HIP_CHECK(hipMemcpy2D(A_d, pitch_A, A_h, COLUMNS*sizeof(TestType),
            COLUMNS*sizeof(TestType), ROWS, hipMemcpyHostToDevice));

      // Device to Device
      HIP_CHECK(hipMemcpy2D(X_d, pitch_X, A_d,
            pitch_A, COLUMNS*sizeof(TestType),
            ROWS, hipMemcpyDeviceToDevice));

      // Device to Host
      HIP_CHECK(hipMemcpy2D(B_h, COLUMNS*sizeof(TestType), X_d,
            pitch_X, COLUMNS*sizeof(TestType), ROWS, hipMemcpyDeviceToHost));

      // Validating the result
      REQUIRE(HipTest::checkArray<TestType>(A_h, B_h, COLUMNS, ROWS) == true);

      // DeAllocating the memory
      HIP_CHECK(hipFree(A_d));
      if (mem_type) {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
            A_h, B_h, C_h, true);
      } else {
        HipTest::freeArrays<TestType>(nullptr, nullptr, nullptr,
            A_h, B_h, C_h, false);
      }
      HIP_CHECK(hipFree(X_d));
    } else {
      SUCCEED("Machine does not seem to have P2P");
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}

/*
This Testcase verifies the null size checks of hipMemcpy2D API
*/
TEST_CASE("Unit_hipMemcpy2D_SizeCheck") {
  HIP_CHECK(hipSetDevice(0));
  int* A_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(int)};

  // Allocating memory
  HipTest::initArrays<int>(nullptr, nullptr, nullptr,
      &A_h, nullptr, nullptr, NUM_W*NUM_H);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
        &pitch_A, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<int>(NUM_W*NUM_H, A_h, nullptr, nullptr);

  SECTION("hipMemcpy2D API where Source Pitch is zero") {
    REQUIRE(hipMemcpy2D(A_h, 0, A_d,
            pitch_A, NUM_W, NUM_H,
            hipMemcpyDeviceToHost) != hipSuccess);
  }

  SECTION("hipMemcpy2D API where Destination Pitch is zero") {
    REQUIRE(hipMemcpy2D(A_h, width, A_d,
            0, NUM_W, NUM_H,
            hipMemcpyDeviceToHost) != hipSuccess);
  }

  SECTION("hipMemcpy2D API where height is zero") {
    REQUIRE(hipMemcpy2D(A_h, width, A_d,
            pitch_A, NUM_W, 0,
            hipMemcpyDeviceToHost) == hipSuccess);
  }

  SECTION("hipMemcpy2D API where width is zero") {
    REQUIRE(hipMemcpy2D(A_h, width, A_d,
            pitch_A, 0, NUM_H,
            hipMemcpyDeviceToHost) == hipSuccess);
  }

  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

/*
This Testcase verifies all the negative scenarios of hipMemcpy2D API
*/
TEST_CASE("Unit_hipMemcpy2D_Negative") {
  HIP_CHECK(hipSetDevice(0));
  int* A_h{nullptr}, *A_d{nullptr};
  size_t pitch_A;
  size_t width{NUM_W * sizeof(int)};

  // Allocating memory
  HipTest::initArrays<int>(nullptr, nullptr, nullptr,
      &A_h, nullptr, nullptr, NUM_W*NUM_H);
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d),
        &pitch_A, width, NUM_H));

  // Initialize the data
  HipTest::setDefaultData<int>(NUM_W*NUM_H, A_h, nullptr, nullptr);

  SECTION("hipMemcpy2D API by Passing nullptr to destination") {
    REQUIRE(hipMemcpy2D(nullptr, width, A_d,
          pitch_A, COLUMNS*sizeof(int), ROWS,
          hipMemcpyDeviceToHost) != hipSuccess);
  }

  SECTION("hipMemcpy2D API by Passing nullptr to destination") {
    REQUIRE(hipMemcpy2D(nullptr, width, nullptr,
          pitch_A, COLUMNS*sizeof(int), ROWS,
          hipMemcpyDeviceToHost) != hipSuccess);
  }

  SECTION("hipMemcpy2D API where width is greater than destination pitch") {
    REQUIRE(hipMemcpy2D(A_h, 10, A_d, pitch_A,
          COLUMNS*sizeof(int), ROWS,
          hipMemcpyDeviceToHost) != hipSuccess);
  }

  // DeAllocating the memory
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}
