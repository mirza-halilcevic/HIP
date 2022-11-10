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

#pragma once

#include <cuda.h>
#include <vulkan/vulkan.h>
#include <vector>

#ifdef _WIN64
#include <VersionHelpers.h>
#endif

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

#define E(expr)                                                                                    \
  {                                                                                                \
    cudaError_t err = (expr);                                                                      \
    UNSCOPED_INFO("Cuda error: " << cudaGetErrorString(err));                                      \
    REQUIRE(cudaSuccess == err);                                                                   \
  }

#define VK_CHECK_RESULT(code)                                                                      \
  {                                                                                                \
    VkResult res = (code);                                                                         \
    UNSCOPED_INFO("Vulkan error: " << std::to_string(res));                                        \
    REQUIRE(VK_SUCCESS == res);                                                                    \
  }

class VulkanTest {
 public:
  VulkanTest(bool enable_validation)
      : _enable_validation{enable_validation}, _sem_handle_type{GetVKSemHandlePlatformType()} {
    if (_enable_validation) {
      _required_instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    CreateInstance();
    CreateDevice();
    CreateCommandBuffer();
  }

  ~VulkanTest() {
    for (const auto s : _semaphores) {
      vkDestroySemaphore(_device, s, nullptr);
    }

    for (const auto f : _fences) {
      vkDestroyFence(_device, f, nullptr);
    }

    for (const auto& s : _stores) {
      vkUnmapMemory(_device, s.memory);
      vkDestroyBuffer(_device, s.buffer, nullptr);
      vkFreeMemory(_device, s.memory, nullptr);
    }

    if (_command_buffer != VK_NULL_HANDLE)
      vkFreeCommandBuffers(_device, _command_pool, 1, &_command_buffer);

    if (_command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(_device, _command_pool, nullptr);

    if (_device != VK_NULL_HANDLE) vkDestroyDevice(_device, nullptr);

    if (_instance != VK_NULL_HANDLE) vkDestroyInstance(_instance, nullptr);
  }

  VulkanTest(const VulkanTest&) = delete;

  VulkanTest(VulkanTest&&) = delete;

  template <typename T> struct MappedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    T* host_ptr = nullptr;
  };

  enum class SemaphoreType { Timeline, Binary };


  template <typename T>
  MappedBuffer<T> CreateMappedStorage(uint32_t count, VkBufferUsageFlagBits transfer_flags);

  VkFence CreateFence();

  VkSemaphore CreateExternalSemaphore(SemaphoreType sem_type, uint64_t initial_value = 0);

  cudaExternalSemaphoreHandleDesc BuildSemaphoreDescriptor(VkSemaphore vk_sem,
                                                           SemaphoreType sem_type);


  VkDevice GetDevice() const { return _device; }

  VkCommandBuffer GetCommandBuffer() const { return _command_buffer; }

  VkQueue GetQueue() const { return _queue; }

 private:
  void CreateInstance();

  void CreateDevice();

  void CreateCommandBuffer();

  bool CheckExtensionSupport(std::vector<const char*> expected_extensions);

  void EnableValidationLayer();

  VkDebugUtilsMessengerCreateInfoEXT BuildDebugCreateInfo();

  uint32_t GetComputeQueueFamilyIndex();

  void FindPhysicalDevice();

  uint32_t FindMemoryType(uint32_t memory_type_bits, VkMemoryPropertyFlags properties);

  cudaExternalSemaphoreHandleType VulkanHandleTypeToCudaHandleType(SemaphoreType sem_type);

#ifdef _WIN64
  HANDLE
  GetSemaphoreHandle(VkSemaphore semaphore);
#else
  int GetSemaphoreHandle(VkSemaphore semaphore);
#endif

  VkExternalSemaphoreHandleTypeFlagBits GetVKSemHandlePlatformType() const;

  struct Storage {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t size = 0u;
  };

 private:
  const bool _enable_validation = false;
  const VkExternalSemaphoreHandleTypeFlagBits _sem_handle_type;
  VkInstance _instance = VK_NULL_HANDLE;
  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  VkDevice _device = VK_NULL_HANDLE;
  VkQueue _queue = VK_NULL_HANDLE;
  VkCommandPool _command_pool = VK_NULL_HANDLE;
  VkCommandBuffer _command_buffer = VK_NULL_HANDLE;
  uint32_t _compute_family_queue_idx = 0u;
  std::vector<const char*> _enabled_layers;

  std::vector<VkSemaphore> _semaphores;
  std::vector<VkFence> _fences;
  std::vector<Storage> _stores;

  std::vector<const char*> _required_instance_extensions{
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME};
  std::vector<const char*> _required_device_extensions{VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                                                       VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};
};

template <typename T>
VulkanTest::MappedBuffer<T> VulkanTest::CreateMappedStorage(uint32_t count,
                                                            VkBufferUsageFlagBits transfer_flags) {
  Storage storage;
  storage.size = count * sizeof(T);

  VkBufferCreateInfo buffer_create_info = {};
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.size = sizeof(int);
  buffer_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | transfer_flags;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK_RESULT(vkCreateBuffer(_device, &buffer_create_info, nullptr, &storage.buffer));

  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(_device, storage.buffer, &memory_requirements);

  VkMemoryAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.allocationSize = memory_requirements.size;
  allocate_info.memoryTypeIndex =
      FindMemoryType(memory_requirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  REQUIRE(allocate_info.memoryTypeIndex != VK_MAX_MEMORY_TYPES);

  VK_CHECK_RESULT(vkAllocateMemory(_device, &allocate_info, nullptr, &storage.memory));
  VK_CHECK_RESULT(vkBindBufferMemory(_device, storage.buffer, storage.memory, 0));

  T* host_ptr = nullptr;
  VK_CHECK_RESULT(vkMapMemory(_device, storage.memory, 0, storage.size, 0,
                              reinterpret_cast<void**>(&host_ptr)));

  _stores.push_back(storage);
  return MappedBuffer<T>{storage.buffer, host_ptr};
}