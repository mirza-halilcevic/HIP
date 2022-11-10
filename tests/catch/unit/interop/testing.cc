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

void SingleSemaphoreTest(bool timeline_semaphore) {
  VulkanTest vkt(enable_validation);

  constexpr uint32_t count = 1;
  const auto [src_buffer, src_host] =
      vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  const auto [dst_buffer, dst_host] =
      vkt.CreateMappedStorage<int>(count, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  const auto command_buffer = vkt.GetCommandBuffer();
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(command_buffer, &begin_info));
  VkBufferCopy buffer_copy = {};
  buffer_copy.size = count * sizeof(*src_host);
  vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &buffer_copy);
  VK_CHECK_RESULT(vkEndCommandBuffer(command_buffer));

  const auto semaphore = vkt.CreateExternalSemaphore(
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, timeline_semaphore);

  const auto cuda_sem_handle_desc = vkt.BuildSemaphoreDescriptor(
      semaphore, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, timeline_semaphore);

  cudaExternalSemaphore_t cuda_ext_semaphore;
  E(cudaImportExternalSemaphore(&cuda_ext_semaphore, &cuda_sem_handle_desc));

  cudaExternalSemaphoreWaitParams cuda_ext_semaphore_wait_params = {};
  cuda_ext_semaphore_wait_params.flags = 0;
  cuda_ext_semaphore_wait_params.params.fence.value = timeline_semaphore ? 1 : 0;
  E(cudaWaitExternalSemaphoresAsync(&cuda_ext_semaphore, &cuda_ext_semaphore_wait_params, 1,
                                    nullptr));
  REQUIRE(cudaStreamQuery(nullptr) == cudaErrorNotReady);

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;
  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = &semaphore;

  *src_host = 42;

  const auto fence = vkt.CreateFence();
  VK_CHECK_RESULT(vkQueueSubmit(vkt.GetQueue(), 1, &submit_info, fence));
  VK_CHECK_RESULT(
      vkWaitForFences(vkt.GetDevice(), 1, &fence, VK_TRUE, 5'000'000'000 /*5 seconds*/));

  // Sometimes in CUDA the stream is not immediately ready after the fence wait is finished
  cudaError_t query_result = cudaErrorNotReady;
  for (auto _ = 0; _ < 5; ++_) {
    if ((query_result = cudaStreamQuery(nullptr)) != cudaSuccess) {
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    }
  }
  REQUIRE(cudaSuccess == query_result);

  REQUIRE(42 == *dst_host);
}


TEST_CASE("Binary_Semaphore") { SingleSemaphoreTest(false); }

TEST_CASE("Timeline_Semaphore") { SingleSemaphoreTest(true); }