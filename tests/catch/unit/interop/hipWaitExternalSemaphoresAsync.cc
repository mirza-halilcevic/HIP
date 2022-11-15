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

constexpr bool enable_validation = false;

TEST_CASE("Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Binary_Semaphore") {
  VulkanTest vkt(enable_validation);

  constexpr uint32_t count = 1;
  const auto src_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  const auto dst_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  const auto command_buffer = vkt.GetCommandBuffer();

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &begin_info));
  VkBufferCopy buffer_copy = {};
  buffer_copy.size = count * sizeof(*src_storage.host_ptr);
  vkCmdCopyBuffer(command_buffer, src_storage.buffer, dst_storage.buffer, 1, &buffer_copy);
  VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer));

  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  const auto cuda_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);
  cudaExternalSemaphore_t cuda_ext_semaphore;
  E(cudaImportExternalSemaphore(&cuda_ext_semaphore, &cuda_sem_handle_desc));

  cudaExternalSemaphoreWaitParams cuda_ext_semaphore_wait_params = {};
  cuda_ext_semaphore_wait_params.flags = 0;
  cuda_ext_semaphore_wait_params.params.fence.value = 0;
  E(cudaWaitExternalSemaphoresAsync(&cuda_ext_semaphore, &cuda_ext_semaphore_wait_params, 1,
                                    nullptr));
  PollStream(nullptr, cudaErrorNotReady);

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &semaphore;

  *src_storage.host_ptr = 42;

  const auto fence = vkt.CreateFence();
  VK_CHECK_RESULT(vkQueueSubmit(vkt.GetQueue(), 1, &submit_info, fence));
  VK_CHECK_RESULT(
      vkWaitForFences(vkt.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000 /*5 seconds*/));

  PollStream(nullptr, cudaSuccess);

  REQUIRE(42 == *dst_storage.host_ptr);

  E(cudaDestroyExternalSemaphore(cuda_ext_semaphore));
}

TEST_CASE("Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Timeline_Semaphore") {
  VulkanTest vkt(enable_validation);

  const auto [wait_value, signal_value] =
      GENERATE(std::make_pair(2, 2), std::make_pair(2, 3), std::make_pair(3, 2));
  INFO("Wait value: " << wait_value << ", signal value: " << signal_value);

  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_TIMELINE);
  const auto cuda_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_TIMELINE);
  cudaExternalSemaphore_t cuda_ext_semaphore;
  E(cudaImportExternalSemaphore(&cuda_ext_semaphore, &cuda_sem_handle_desc));

  cudaExternalSemaphoreWaitParams cuda_ext_semaphore_wait_params = {};
  cuda_ext_semaphore_wait_params.flags = 0;
  cuda_ext_semaphore_wait_params.params.fence.value = wait_value;
  E(cudaWaitExternalSemaphoresAsync(&cuda_ext_semaphore, &cuda_ext_semaphore_wait_params, 1,
                                    nullptr));
  PollStream(nullptr, cudaErrorNotReady);

  VkSemaphoreSignalInfo signal_info = {};
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.semaphore = semaphore;
  signal_info.value = signal_value;
  VK_CHECK_RESULT(vkSignalSemaphore(vkt.GetDevice(), &signal_info));
  if (wait_value > signal_value) {
    PollStream(nullptr, cudaErrorNotReady);
    signal_info.value = wait_value;
    VK_CHECK_RESULT(vkSignalSemaphore(vkt.GetDevice(), &signal_info));
  }
  PollStream(nullptr, cudaSuccess);

  E(cudaDestroyExternalSemaphore(cuda_ext_semaphore));
}

TEST_CASE("Unit_hipWaitExternalSemaphoresAsync_Vulkan_Positive_Multiple_Semaphores") {
  VulkanTest vkt(enable_validation);

#if HT_AMD
  constexpr auto second_semaphore_type = VK_SEMAPHORE_TYPE_BINARY;
#else
  constexpr auto second_semaphore_type = VK_SEMAPHORE_TYPE_TIMELINE;
#endif

  constexpr uint32_t count = 1;
  const auto src_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  const auto dst_storage = vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  const auto command_buffer = vkt.GetCommandBuffer();

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &begin_info));
  VkBufferCopy buffer_copy = {};
  buffer_copy.size = count * sizeof(*src_storage.host_ptr);
  vkCmdCopyBuffer(command_buffer, src_storage.buffer, dst_storage.buffer, 1, &buffer_copy);
  VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer));

  const auto binary_semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  const auto cuda_binary_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(binary_semaphore, VK_SEMAPHORE_TYPE_BINARY);
  cudaExternalSemaphore_t cuda_binary_ext_semaphore;
  E(cudaImportExternalSemaphore(&cuda_binary_ext_semaphore, &cuda_binary_sem_handle_desc));

  const auto timeline_semaphore = vkt.CreateExternalSemaphore(second_semaphore_type);
  const auto cuda_timeline_sem_handle_desc =
      vkt.BuildSemaphoreDescriptor(timeline_semaphore, second_semaphore_type);
  cudaExternalSemaphore_t cuda_timeline_ext_semaphore;
  E(cudaImportExternalSemaphore(&cuda_timeline_ext_semaphore, &cuda_timeline_sem_handle_desc));

  cudaExternalSemaphoreWaitParams binary_semaphore_wait_params = {};
  binary_semaphore_wait_params.params.fence.value = 0;

  cudaExternalSemaphoreWaitParams timeline_semaphore_wait_params = {};
  timeline_semaphore_wait_params.params.fence.value =
      second_semaphore_type == VK_SEMAPHORE_TYPE_TIMELINE ? 1 : 0;

  cudaExternalSemaphore_t ext_semaphores[] = {cuda_binary_ext_semaphore,
                                              cuda_timeline_ext_semaphore};
  cudaExternalSemaphoreWaitParams wait_params[] = {binary_semaphore_wait_params,
                                                   timeline_semaphore_wait_params};
  E(cudaWaitExternalSemaphoresAsync(ext_semaphores, wait_params, 2));

  PollStream(nullptr, cudaErrorNotReady);

  if (second_semaphore_type == VK_SEMAPHORE_TYPE_TIMELINE) {
    VkSemaphoreSignalInfo signal_info = {};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = timeline_semaphore;
    signal_info.value = 1;
    VK_CHECK_RESULT(vkSignalSemaphore(vkt.GetDevice(), &signal_info));

    PollStream(nullptr, cudaErrorNotReady);
  }

  VkSubmitInfo submit_info = {};
  VkSemaphore signal_semaphores[] = {binary_semaphore, timeline_semaphore};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  submit_info.signalSemaphoreCount = second_semaphore_type == VK_SEMAPHORE_TYPE_TIMELINE ? 1 : 2;
  submit_info.pSignalSemaphores =
      second_semaphore_type == VK_SEMAPHORE_TYPE_MAX_ENUM ? &binary_semaphore : signal_semaphores;

  const auto fence = vkt.CreateFence();
  VK_CHECK_RESULT(vkQueueSubmit(vkt.GetQueue(), 1, &submit_info, fence));
  VK_CHECK_RESULT(
      vkWaitForFences(vkt.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000 /*5 seconds*/));

  PollStream(nullptr, cudaSuccess);

  E(cudaDestroyExternalSemaphore(cuda_timeline_ext_semaphore));
  E(cudaDestroyExternalSemaphore(cuda_binary_ext_semaphore));
}

TEST_CASE("Unit_hipWaitExternalSemaphoresAsync_Vulkan_Negative_Parameters") {
  VulkanTest vkt(enable_validation);
  cudaExternalSemaphoreWaitParams wait_params = {};
  wait_params.params.fence.value = 1;

  SECTION("extSemArray == nullptr") {
    REQUIRE(cudaWaitExternalSemaphoresAsync(nullptr, &wait_params, 1) == cudaErrorInvalidValue);
  }

  SECTION("paramsArray == nullptr") {
    const auto cuda_ext_semaphore = ImportBinarySemaphore(vkt);
    REQUIRE(cudaWaitExternalSemaphoresAsync(&cuda_ext_semaphore, nullptr, 1) ==
            cudaErrorInvalidValue);
    E(cudaDestroyExternalSemaphore(cuda_ext_semaphore));
  }

  SECTION("Wait params flag != 0") {
    const auto cuda_ext_semaphore = ImportBinarySemaphore(vkt);
    wait_params.flags = 1;
    REQUIRE(cudaWaitExternalSemaphoresAsync(&cuda_ext_semaphore, &wait_params, 1) ==
            cudaErrorInvalidValue);
    E(cudaDestroyExternalSemaphore(cuda_ext_semaphore));
  }

  SECTION("Invalid stream") {
    const auto cuda_ext_semaphore = ImportBinarySemaphore(vkt);
    cudaStream_t stream = nullptr;
    E(cudaStreamCreate(&stream));
    E(cudaStreamDestroy(stream));
    // Doesn't want to compile for some reason
    // REQUIRE(cudaWaitExternalSemaphoresAsync(&cuda_ext_semaphore, &wait_params, 1, stream) ==
    //         cudaErrorDeviceUninitilialized);
    E(cudaDestroyExternalSemaphore(cuda_ext_semaphore));
  }
}