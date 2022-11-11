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

#include "vulkan_test.hh"

constexpr bool enable_validation = true;

template <typename T> __global__ void Set(T* ptr, const T val) { ptr[threadIdx.x] = val; }

TEST_CASE("Blahem") {
  VulkanTest vkt(enable_validation);
  using type = uint8_t;

  constexpr uint32_t count = 3;

  const auto vk_storage =
      vkt.CreateMappedStorage<type>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);

  const auto desc = vkt.BuildMemoryDescriptor(vk_storage.memory, vk_storage.size);

  cudaExternalMemory_t ext_memory;
  E(cudaImportExternalMemory(&ext_memory, &desc));

  cudaExternalMemoryBufferDesc external_mem_buffer_desc = {};
  external_mem_buffer_desc.size = vk_storage.size;

  type* dev_ptr = nullptr;
  E(cudaExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&dev_ptr), ext_memory,
                                      &external_mem_buffer_desc));

  Set<<<1, count>>>(dev_ptr, static_cast<type>(42));
  cudaDeviceSynchronize();
  REQUIRE(vk_storage.host_ptr[1] == static_cast<type>(42));
  E(cudaFree(dev_ptr));
  for(int i = 0; i < count; ++i) {
    std::cout << static_cast<int>(vk_storage.host_ptr[i]) << std::endl; 
  }
  E(cudaDestroyExternalMemory(ext_memory));
}