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

#include <experimental/filesystem>
#include <experimental/string_view>
#include <fstream>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

// Load module into buffer instead of mapping file to avoid platform specific mechanisms
static std::vector<char> LoadModuleIntoBuffer(const std::experimental::string_view path_string) {
  std::experimental::filesystem::path p(path_string.data());
  const auto file_size = std::experimental::filesystem::file_size(p);
  std::ifstream f(p, std::ios::binary | std::ios::in);
  REQUIRE(f);
  std::vector<char> empty_module(file_size);
  REQUIRE(f.read(empty_module.data(), file_size));
  return empty_module;
}

static std::vector<char> CreateRTCCharArray(const std::experimental::string_view src) {
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, src.data(), "foo", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(prog, 0, nullptr));
  size_t code_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size, '\0');
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return code;
}

TEST_CASE("Unit_hipModuleLoadData_Positive_Basic") {
  hipModule_t module = nullptr;

  SECTION("Load compiled module from file") {
    const auto loaded_module = LoadModuleIntoBuffer("empty_module.code");
    HIP_CHECK(hipModuleLoadData(&module, loaded_module.data()));
    REQUIRE(module != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }

  SECTION("Load RTCd module") {
    const auto rtc = CreateRTCCharArray(R"(extern "C" __global__ void kernel() {})");
    HIP_CHECK(hipModuleLoadData(&module, rtc.data()));
    REQUIRE(module != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }
}

TEST_CASE("Unit_hipModuleLoadData_Negative_Parameters") {
  hipModule_t module;

  SECTION("module == nullptr") {
    const auto loaded_module = LoadModuleIntoBuffer("empty_module.code");
    HIP_CHECK_ERROR(hipModuleLoadData(nullptr, loaded_module.data()), hipErrorInvalidValue);
    LoadModuleIntoBuffer("empty_module.code");
  }

  SECTION("image == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoadData(&module, nullptr), hipErrorInvalidValue);
  }

  SECTION("image == empty string") {
    HIP_CHECK_ERROR(hipModuleLoadData(&module, ""), hipErrorInvalidKernelFile);
  }
}