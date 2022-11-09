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

#include <cuda.h>
#include <iostream>
#include <vulkan/vulkan.h>
#include <vector>
#include <algorithm>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

#define E(expr)                                                                                    \
  {                                                                                                \
    cudaError_t err = (expr);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      UNSCOPED_INFO("Cuda error: " << cudaGetErrorString(err));                                    \
    }                                                                                              \
    REQUIRE(cudaSuccess == err);                                                                   \
  }

#define VK_CHECK_RESULT(code)                                                                      \
  {                                                                                                \
    VkResult res = (code);                                                                         \
    if (res != VK_SUCCESS) {                                                                       \
      UNSCOPED_INFO("Vulkan error" << std::to_string(res));                                        \
    }                                                                                              \
    REQUIRE(VK_SUCCESS == res);                                                                    \
  }

inline static cudaExternalSemaphoreHandleType VulkanHandleTypeToCudaHandleType(
    const VkExternalSemaphoreHandleTypeFlagBits handle_type, bool is_timeline_semaphore) {
  if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    return is_timeline_semaphore ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
                                 : cudaExternalSemaphoreHandleTypeOpaqueWin32;
  } else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    return is_timeline_semaphore ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
                                 : cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
  } else if (handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
    return is_timeline_semaphore ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
                                 : cudaExternalSemaphoreHandleTypeOpaqueFd;
  }

  throw std::invalid_argument("Invalid vulkan semaphore handle type");
}

#ifdef _WIN64
HANDLE
GetSemaphoreHandle(VkSemaphore semaphore, const VkExternalSemaphoreHandleTypeFlagBits handle_type,
                   VkDevice device) {
  HANDLE handle = 0;

  VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR = {};
  semaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  semaphoreGetWin32HandleInfoKHR.pNext = NULL;
  semaphoreGetWin32HandleInfoKHR.semaphore = semaphore;
  semaphoreGetWin32HandleInfoKHR.handleType = handle_type;

  PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
  fpGetSemaphoreWin32HandleKHR =
      (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
  if (!fpGetSemaphoreWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetSemaphoreWin32HandleKHR");
  }
  if (fpGetSemaphoreWin32HandleKHR(device, &semaphoreGetWin32HandleInfoKHR, &handle) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }
  return handle;
}
#else
int GetSemaphoreHandle(VkSemaphore semaphore,
                       const VkExternalSemaphoreHandleTypeFlagBits handle_type, VkDevice device) {
  int fd;

  VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
  semaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  semaphoreGetFdInfoKHR.pNext = NULL;
  semaphoreGetFdInfoKHR.semaphore = semaphore;
  semaphoreGetFdInfoKHR.handleType = handle_type;

  PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
  fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
  if (!fpGetSemaphoreFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetSemaphoreFdKHR");
  }
  if (fpGetSemaphoreFdKHR(device, &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve semaphore handle");
  }

  return fd;
}
#endif

VkSemaphore CreateTimelineSemaphore(VkExternalSemaphoreHandleTypeFlagBits handle_type,
                                    VkDevice device) {
  VkSemaphoreTypeCreateInfo timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_create_info.pNext = nullptr;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue = 0;

  VkExportSemaphoreCreateInfoKHR export_sem_create_info = {};
  export_sem_create_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
  export_sem_create_info.pNext = &timeline_create_info;
  export_sem_create_info.handleTypes = handle_type;

  VkSemaphoreCreateInfo sem_info = {};
  sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  sem_info.pNext = &export_sem_create_info;

  VkSemaphore semaphore;
  if (vkCreateSemaphore(device, &sem_info, nullptr, &semaphore) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create timeline semaphore");
  }
  return semaphore;
}

cudaExternalSemaphoreHandleDesc BuildSemaphoreDescriptor(
    cudaExternalSemaphore_t& cuda_sem, VkSemaphore& vk_sem,
    VkExternalSemaphoreHandleTypeFlagBits handle_type, VkDevice device) {
  cudaExternalSemaphoreHandleDesc sem_handle_desc = {};
  sem_handle_desc.type =
      VulkanHandleTypeToCudaHandleType(handle_type, true /*Needs to be passed as argument*/);
#ifdef _WIN64
  sem_handle_desc.handle.win32.handle = GetSemaphoreHandle(vk_sem, handle_type, device);
#else
  sem_handle_desc.handle.fd = GetSemaphoreHandle(vk_sem, handle_type, device);
#endif
  sem_handle_desc.flags = 0;

  return sem_handle_desc;
}


static VKAPI_ATTR VkBool32 VKAPI_CALL
DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
              const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void*) {
  std::cerr << "Validation layer: " << callback_data->pMessage << std::endl;
  return VK_FALSE;
}

class VulkanTestBase {
 public:
  VulkanTestBase(bool enable_validation) : _enable_validation{enable_validation} {
    if (_enable_validation) {
      _required_instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    CreateInstance();
    FindPhysicalDevice();
    CreateDevice();
    CreateCommandBuffer();
    CreateInBuffer();
    CreateOutBuffer();
    Blahem();
  }

 private:
  bool CheckExtensionSupport(std::vector<const char*> expected_extensions) {
    uint32_t extension_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> extension_properties(extension_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extension_properties.data());

    std::vector<const char*> supported_extensions;
    supported_extensions.reserve(extension_count);
    std::transform(extension_properties.begin(), extension_properties.end(),
                   std::back_inserter(supported_extensions),
                   [](const auto& p) { return p.extensionName; });

    constexpr auto p = [](const char* l, const char* r) { return strcmp(l, r) < 0; };
    std::sort(expected_extensions.begin(), expected_extensions.end(), p);
    std::sort(supported_extensions.begin(), supported_extensions.end(), p);

    return std::includes(supported_extensions.begin(), supported_extensions.end(),
                         expected_extensions.begin(), expected_extensions.end(),
                         [](const char* l, const char* r) { return strcmp(l, r) == 0; });
  }

  void EnableValidationLayer() {
    uint32_t layer_count = 0;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> layer_properties(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layer_properties.data());
    const bool found_val_layer =
        std::any_of(layer_properties.cbegin(), layer_properties.cend(), [](const auto& props) {
          return strcmp(props.layerName, "VK_LAYER_KHRONOS_validation") == 0;
        });


    if (found_val_layer) {
      _enabled_layers.push_back("VK_LAYER_KHRONOS_validation");
    } else {
        UNSCOPED_INFO("Validation was requested, but the validation layer could not be located");
        REQUIRE(found_val_layer);
    }
  }

  VkDebugUtilsMessengerCreateInfoEXT BuildDebugCreateInfo() {
    VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {};

    debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_create_info.pfnUserCallback = DebugCallback;

    return debug_create_info;
  }

  void CreateInstance() {
    UNSCOPED_INFO("Not all of the required instance extensions are supported");
    REQUIRE(CheckExtensionSupport(_required_instance_extensions));

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledExtensionCount = static_cast<uint32_t>(_required_instance_extensions.size());
    create_info.ppEnabledExtensionNames = _required_instance_extensions.data();
    create_info.enabledLayerCount = static_cast<uint32_t>(_enabled_layers.size());
    create_info.ppEnabledLayerNames = _enabled_layers.data();

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {};
    if (_enable_validation) {
      EnableValidationLayer();
      debug_create_info = BuildDebugCreateInfo();
      create_info.pNext = &debug_create_info;
    }

    VK_CHECK_RESULT(vkCreateInstance(&create_info, nullptr, &_instance));
  }

  void FindPhysicalDevice() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(_instance, &device_count, nullptr);
    REQUIRE(device_count != 0u);

    std::vector<VkPhysicalDevice> physical_devices(device_count);
    vkEnumeratePhysicalDevices(_instance, &device_count, physical_devices.data());

    _physical_device = physical_devices[0];
  }

  uint32_t GetComputeQueueFamilyIndex() {
    uint32_t queue_family_count = 0u;

    vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &queue_family_count,
                                             queue_families.data());

    const auto it =
        std::find_if(queue_families.cbegin(), queue_families.cend(), [](const auto& props) {
          return props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT);
        });
    REQUIRE(it != queue_families.cend());

    return std::distance(queue_families.cbegin(), it);
  }

  void CreateDevice() {
    UNSCOPED_INFO("Not all of the required device extensions are supported");
    REQUIRE(CheckExtensionSupport(_required_device_extensions));

    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = _compute_family_queue_idx = GetComputeQueueFamilyIndex();
    queue_create_info.queueCount = 1;
    float queue_priorities = 1.0;
    queue_create_info.pQueuePriorities = &queue_priorities;

    VkDeviceCreateInfo device_create_info = {};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.enabledLayerCount = _enabled_layers.size();
    device_create_info.ppEnabledLayerNames = _enabled_layers.data();
    device_create_info.enabledExtensionCount = _required_device_extensions.size();
    device_create_info.ppEnabledExtensionNames = _required_device_extensions.data();
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.queueCreateInfoCount = 1;

    VK_CHECK_RESULT(vkCreateDevice(_physical_device, &device_create_info, nullptr, &_device));
    vkGetDeviceQueue(_device, _compute_family_queue_idx, 0, &_queue);
  }

  void CreateCommandBuffer() {
    VkCommandPoolCreateInfo command_pool_create_info = {};
    command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_create_info.flags = 0;
    command_pool_create_info.queueFamilyIndex = _compute_family_queue_idx;
    VK_CHECK_RESULT(
        vkCreateCommandPool(_device, &command_pool_create_info, nullptr, &_command_pool));

    VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
    command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.commandPool = _command_pool;
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = 1;
    VK_CHECK_RESULT(
        vkAllocateCommandBuffers(_device, &command_buffer_allocate_info, &_command_buffer));
  }

  uint32_t FindMemoryType(uint32_t memory_type_bits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(_physical_device, &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((memory_type_bits & (1 << i)) &&
          ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties))
        return i;
    }
    return static_cast<uint32_t>(-1);
  }

  void CreateInBuffer() {
    VkBufferCreateInfo buffer_create_info = {};
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.size = sizeof(int);
    buffer_create_info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(_device, &buffer_create_info, nullptr, &_in_buffer));

    VkMemoryRequirements memory_requiremenets;
    vkGetBufferMemoryRequirements(_device, _in_buffer, &memory_requiremenets);

    VkMemoryAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requiremenets.size;
    allocate_info.memoryTypeIndex =
        FindMemoryType(memory_requiremenets.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(_device, &allocate_info, nullptr, &_in_buffer_memory));
    VK_CHECK_RESULT(vkBindBufferMemory(_device, _in_buffer, _in_buffer_memory, 0));
  }

  void CreateOutBuffer() {
    VkBufferCreateInfo buffer_create_info = {};
    buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_create_info.size = sizeof(int);
    buffer_create_info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(_device, &buffer_create_info, nullptr, &_out_buffer));

    VkMemoryRequirements memory_requiremenets;
    vkGetBufferMemoryRequirements(_device, _out_buffer, &memory_requiremenets);

    VkMemoryAllocateInfo allocate_info = {};
    allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate_info.allocationSize = memory_requiremenets.size;
    allocate_info.memoryTypeIndex =
        FindMemoryType(memory_requiremenets.memoryTypeBits,
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(_device, &allocate_info, nullptr, &_out_buffer_memory));
    VK_CHECK_RESULT(vkBindBufferMemory(_device, _out_buffer, _out_buffer_memory, 0));
  }

  void Blahem() {
    VkBufferCopy buffer_copy = {};
    buffer_copy.srcOffset = 0;
    buffer_copy.dstOffset = 0;
    buffer_copy.size = sizeof(int);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(_command_buffer, &begin_info));
    vkCmdCopyBuffer(_command_buffer, _in_buffer, _out_buffer, 1, &buffer_copy);
    VK_CHECK_RESULT(vkEndCommandBuffer(_command_buffer));

    VkFence fence;
    VkFenceCreateInfo fence_create_info = {};
    fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_create_info.flags = 0;
    VK_CHECK_RESULT(vkCreateFence(_device, &fence_create_info, nullptr, &fence));

    VkExportSemaphoreCreateInfoKHR export_semaphore_info = {};
    export_semaphore_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
    export_semaphore_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo semaphore_create_info = {};
    semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_create_info.pNext = &export_semaphore_info;

    VkSemaphore semaphore;
    VK_CHECK_RESULT(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &semaphore));

    int semaphore_handle =
        GetSemaphoreHandle(semaphore, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, _device);
    cudaExternalSemaphoreHandleDesc cuda_sem_handle_desc = {};
    cuda_sem_handle_desc.type =
        VulkanHandleTypeToCudaHandleType(VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, false);
    cuda_sem_handle_desc.handle.fd = semaphore_handle;
    cuda_sem_handle_desc.flags = 0;
    cudaExternalSemaphore_t cuda_ext_semaphore;
    E(cudaImportExternalSemaphore(&cuda_ext_semaphore, &cuda_sem_handle_desc));
    cudaExternalSemaphoreWaitParams cuda_ext_semaphore_wait_params = {};
    cuda_ext_semaphore_wait_params.flags = 0;
    cuda_ext_semaphore_wait_params.params.fence.value = 0;
    E(cudaWaitExternalSemaphoresAsync(&cuda_ext_semaphore, &cuda_ext_semaphore_wait_params, 1,
                                      nullptr));
    std::cout << cudaGetErrorName(cudaStreamQuery(nullptr)) << std::endl;

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &_command_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &semaphore;

    int* in_data;
    VK_CHECK_RESULT(vkMapMemory(_device, _in_buffer_memory, 0, sizeof(int), 0,
                                reinterpret_cast<void**>(&in_data)));
    *in_data = 42;
    vkUnmapMemory(_device, _in_buffer_memory);

    VK_CHECK_RESULT(vkQueueSubmit(_queue, 1, &submit_info, fence));
    VK_CHECK_RESULT(vkWaitForFences(_device, 1, &fence, VK_TRUE, 10'000'000'000));
    std::cout << cudaGetErrorName(cudaStreamQuery(nullptr)) << std::endl;
    std::cout << "Blahem" << std::endl;

    int* out_data;
    VK_CHECK_RESULT(vkMapMemory(_device, _out_buffer_memory, 0, sizeof(int), 0,
                                reinterpret_cast<void**>(&out_data)));
    std::cout << *out_data << std::endl;
    vkUnmapMemory(_device, _out_buffer_memory);
  }

  VkInstance _instance = VK_NULL_HANDLE;
  VkPhysicalDevice _physical_device = VK_NULL_HANDLE;
  VkDevice _device = VK_NULL_HANDLE;
  VkQueue _queue = VK_NULL_HANDLE;
  VkPipeline _pipeline = VK_NULL_HANDLE;
  VkPipelineLayout _pipeline_Layout = VK_NULL_HANDLE;
  VkCommandPool _command_pool = VK_NULL_HANDLE;
  VkCommandBuffer _command_buffer = VK_NULL_HANDLE;
  VkBuffer _in_buffer = VK_NULL_HANDLE;
  VkBuffer _out_buffer = VK_NULL_HANDLE;
  VkDeviceMemory _in_buffer_memory = VK_NULL_HANDLE;
  VkDeviceMemory _out_buffer_memory = VK_NULL_HANDLE;
  uint32_t _compute_family_queue_idx = 0u;
  bool _enable_validation = false; 

  std::vector<const char*> _required_instance_extensions{
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME};
  std::vector<const char*> _required_device_extensions{VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
                                                      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};
  std::vector<const char*> _enabled_layers;
};

TEST_CASE("Blahem") { VulkanTestBase app(true); }